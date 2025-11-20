import os, logging, json
from tqdm import tqdm
import torch
from torch import nn, optim
from .data_loader import SentenceRELoader
from .utils import AverageMeter
from .amtl_loss import AMTLLoss

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
                 use_class_weights=False,
                 use_amtl=False,
                 amtl_num_segments=4,
                 amtl_lambda=None,
                 negative_ratio=None):
    
        super().__init__()
        self.max_epoch = max_epoch
        # Load data
        if train_path != None:
            self.train_loader = SentenceRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                True,
                negative_ratio=negative_ratio)

        if val_path != None:
            self.val_loader = SentenceRELoader(
                val_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False,
                negative_ratio=None)  # Don't sample validation set

        if test_path != None:
            self.test_loader = SentenceRELoader(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False,
                negative_ratio=None  # Don't sample test set
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
        # Criterion - optionally use AMTL or class weights to address class imbalance
        if use_amtl and train_path is not None:
            logging.info("Using AMTL (Adaptive Multi-Threshold Loss) for class imbalance")
            logging.info("IMPLEMENTATION: Following Xu et al. (2025) exact formulation (Equations 2-4)")

            # Calculate class frequencies for AMTL
            class_freq = {}
            for item in self.train_loader.dataset.data:
                rel = item['relation']
                rel_id = model.rel2id[rel]
                class_freq[rel_id] = class_freq.get(rel_id, 0) + 1

            num_classes = len(model.rel2id)

            # Initialize AMTL loss with correct paper parameters
            self.criterion = AMTLLoss(
                class_freq=class_freq,
                num_classes=num_classes,
                num_segments=amtl_num_segments,
                lambda_smooth=amtl_lambda  # Uses n-0.5 by default (paper's optimal)
            )

            if torch.cuda.is_available():
                self.criterion = self.criterion.cuda()

        elif use_class_weights and train_path is not None:
            import math
            logging.info("Calculating class weights from training data...")
            class_counts = {}
            for item in self.train_loader.dataset.data:
                rel = item['relation']
                class_counts[rel] = class_counts.get(rel, 0) + 1

            # Create weight tensor
            num_classes = len(model.rel2id)
            class_weights = torch.ones(num_classes)

            # TWO-TIER WEIGHTING STRATEGY:
            # 1. Separate Na from positive relations
            # 2. Apply sqrt inverse frequency to positive relations
            # 3. Set Na to a fixed lower weight (tunable)

            # Find Na class
            na_rel = None
            for name in ['NA', 'na', 'Na', 'no_relation', 'Other', 'Others']:
                if name in model.rel2id:
                    na_rel = name
                    break

            # Separate Na from other relations
            non_na_counts = {k: v for k, v in class_counts.items() if k != na_rel}
            non_na_total = sum(non_na_counts.values())
            non_na_classes = len(non_na_counts)

            logging.info(f"Class distribution: {len(class_counts)} total classes")
            logging.info(f"  Na class: {class_counts.get(na_rel, 0)} samples ({100*class_counts.get(na_rel, 0)/sum(class_counts.values()):.1f}%)")
            logging.info(f"  Positive relations: {non_na_total} samples across {non_na_classes} classes")

            # Calculate weights for non-Na classes using SQRT inverse frequency
            # This is less aggressive than pure inverse frequency
            temp_weights = {}
            for rel, count in non_na_counts.items():
                rel_id = model.rel2id[rel]
                # Square root makes weights less extreme for rare classes
                temp_weights[rel_id] = math.sqrt(non_na_total / (non_na_classes * count))

            # Normalize non-Na weights to average to target value
            # Higher target = more emphasis on positive relations vs Na
            target_non_na_avg = 5.0
            if temp_weights:
                mean_non_na = sum(temp_weights.values()) / len(temp_weights)
                for rel_id in temp_weights:
                    class_weights[rel_id] = temp_weights[rel_id] / mean_non_na * target_non_na_avg

            # Set Na to fixed lower weight
            # Lower value = less penalty for misclassifying Na (encourages predicting relations)
            # Higher value = more penalty for misclassifying Na (more conservative)
            # Recommended range: 0.3 - 1.0
            na_weight = 0.5
            if na_rel and na_rel in model.rel2id:
                na_id = model.rel2id[na_rel]
                class_weights[na_id] = na_weight

            # Final normalization so weights average to 1.0
            class_weights = class_weights / class_weights.mean()

            # Log weight statistics
            na_final_weight = class_weights[model.rel2id[na_rel]].item() if na_rel else 0
            non_na_indices = [i for i in range(num_classes) if i != model.rel2id.get(na_rel, -1)]
            mean_non_na_weight = class_weights[non_na_indices].mean().item() if non_na_indices else 0

            logging.info(f"Class weights calculated:")
            logging.info(f"  Na weight: {na_final_weight:.4f}")
            logging.info(f"  Mean non-Na weight: {mean_non_na_weight:.4f}")
            logging.info(f"  Weight ratio (non-Na/Na): {mean_non_na_weight/na_final_weight:.2f}x")
            logging.info(f"  Weight range: [{class_weights.min().item():.4f}, {class_weights.max().item():.4f}]")

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

            # Track per-class correct predictions for macro accuracy (computed once at end)
            from collections import defaultdict
            per_class_correct = defaultdict(int)
            per_class_total = defaultdict(int)

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

                # Track per-class statistics (compute macro at end for efficiency)
                for i in range(label.size(0)):
                    class_id = label[i].item()
                    per_class_total[class_id] += 1
                    if pred[i] == label[i]:
                        per_class_correct[class_id] += 1

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

            # Compute epoch-level macro accuracy
            class_accs = []
            for class_id in per_class_total:
                if per_class_total[class_id] > 0:
                    class_accs.append(per_class_correct[class_id] / per_class_total[class_id])
            epoch_macro_acc = sum(class_accs) / len(class_accs) if class_accs else 0
            logging.info("Epoch %d training complete: loss=%.4f, acc=%.4f, macro_acc=%.4f" %
                        (epoch, avg_loss.avg, avg_acc.avg, epoch_macro_acc))
            # Val
            logging.info("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader)
            logging.info('Early stopping metric: {} | Current: {:.4f} | Best: {:.4f}'.format(
                metric, result[metric], best_metric))
            if result[metric] > best_metric:
                logging.info("🎯 New best model! Saving checkpoint...")
                folder_path = '/'.join(self.ckpt.split('/')[:-1])
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
                best_metric = result[metric]
            else:
                logging.info("No improvement.")
        logging.info("=" * 80)
        logging.info("Training complete! Best %s on val set: %.4f" % (metric, best_metric))
        logging.info("=" * 80)

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

