import os, logging, json
from tqdm import tqdm
import torch
from torch import nn, optim
from .data_loader import SentenceRELoader
from .utils import AverageMeter

class SentenceRE(nn.Module):

    def __init__(self,
                 model,
                 train_path,
                 val_path,
                 test_path,
                 ckpt,
                 batch_size=32,
                 max_epoch=100,
                 lr=0.1,
                 weight_decay=1e-5,
                 warmup_step=300,
                 opt='sgd',
                 use_class_weights=False):
    
        super().__init__()
        self.max_epoch = max_epoch
        # Load data
        if train_path != None:
            self.train_loader = SentenceRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                True)

        if val_path != None:
            self.val_loader = SentenceRELoader(
                val_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False)
        
        if test_path != None:
            self.test_loader = SentenceRELoader(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False
            )
        # Model
        self.model = model

        # CRITICAL FIX: Only use DataParallel with multiple GPUs
        # DataParallel with single GPU can cause issues with model replication
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        logging.info(f"Number of available GPUs: {gpu_count}")

        if gpu_count > 1:
            logging.info("Using DataParallel for multi-GPU training")
            self.parallel_model = nn.DataParallel(self.model)
        else:
            logging.info("Using single GPU or CPU, skipping DataParallel")
            self.parallel_model = self.model
        # Criterion - optionally use class weights to address class imbalance
        if use_class_weights and train_path is not None:
            logging.info("Calculating class weights from training data...")
            class_counts = {}
            for item in self.train_loader.dataset.data:
                rel = item['relation']
                class_counts[rel] = class_counts.get(rel, 0) + 1

            # Create weight tensor
            num_classes = len(model.rel2id)
            class_weights = torch.ones(num_classes)

            # Calculate inverse frequency weights
            total_samples = sum(class_counts.values())
            for rel, count in class_counts.items():
                rel_id = model.rel2id[rel]
                # Inverse frequency weight
                class_weights[rel_id] = total_samples / (num_classes * count)

            # Special handling for 'Na' class - reduce its weight to encourage predicting relations
            if 'Na' in model.rel2id:
                na_id = model.rel2id['Na']
                # Scale down Na weight by 10x to penalize false negatives on actual relations
                class_weights[na_id] = class_weights[na_id] * 0.1

            # Normalize weights so they average to 1.0
            class_weights = class_weights / class_weights.mean()

            logging.info(f"Class weights calculated. Na weight: {class_weights[model.rel2id['Na']]:.4f}, " +
                        f"Mean non-Na weight: {class_weights[[i for i in range(num_classes) if i != model.rel2id.get('Na', -1)]].mean():.4f}")

            if torch.cuda.is_available():
                class_weights = class_weights.cuda()

            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            # Use standard unweighted loss
            if use_class_weights:
                logging.info("Class weights requested but no training data available, using unweighted loss")
            else:
                logging.info("Using standard unweighted CrossEntropyLoss")
            self.criterion = nn.CrossEntropyLoss()
        # Params and optimizer
        params = self.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw': # Optimizer for BERT
            # Use transformers AdamW for BERT fine-tuning (has correct_bias parameter)
            try:
                from transformers import AdamW
            except ImportError:
                # Fallback to torch AdamW if transformers not available
                from torch.optim import AdamW

            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            # Use correct_bias=False for BERT (original BERT implementation behavior)
            try:
                self.optimizer = AdamW(grouped_params, correct_bias=False)
            except TypeError:
                # torch.optim.AdamW doesn't have correct_bias parameter
                self.optimizer = AdamW(grouped_params)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # Warmup
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step, num_training_steps=training_steps)
        else:
            self.scheduler = None
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def train_model(self, metric='acc'):
        best_metric = 0
        global_step = 0
        for epoch in range(self.max_epoch):
            self.train()
            logging.info("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            t = tqdm(self.train_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                args = data[1:]

                logits = self.parallel_model(*args)

                loss = self.criterion(logits, label)

                score, pred = logits.max(-1) # (B)
                acc = float((pred == label).long().sum()) / label.size(0)

                # Log
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)

                # Optimize
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1
            # Val 
            logging.info("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader) 
            logging.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
            if result[metric] > best_metric:
                logging.info("Best ckpt and saved.")
                folder_path = '/'.join(self.ckpt.split('/')[:-1])
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
                best_metric = result[metric]
        logging.info("Best %s on val set: %f" % (metric, best_metric))

    def eval_model(self, eval_loader):
        self.eval()
        avg_acc = AverageMeter()
        pred_result = []
        with torch.no_grad():
            t = tqdm(eval_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                args = data[1:]        
                logits = self.parallel_model(*args)
                score, pred = logits.max(-1) # (B)
                # Save result
                for i in range(pred.size(0)):
                    pred_result.append(pred[i].item())
                # Log
                acc = float((pred == label).long().sum()) / label.size(0)
                avg_acc.update(acc, pred.size(0))
                t.set_postfix(acc=avg_acc.avg)
        result = eval_loader.dataset.eval(pred_result)
        return result

    def get_predictions(self, eval_loader):
        """
        Get raw predictions without evaluation metrics.
        Returns list of predicted label IDs.
        """
        self.eval()
        pred_result = []
        with torch.no_grad():
            t = tqdm(eval_loader, desc='Getting predictions')
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                args = data[1:]
                logits = self.parallel_model(*args)
                score, pred = logits.max(-1) # (B)
                # Save result
                for i in range(pred.size(0)):
                    pred_result.append(pred[i].item())
        return pred_result

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

