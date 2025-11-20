# coding:utf-8
import torch
import numpy as np
import json
import opennre
from opennre import encoder, model, framework
import sys
import os
import argparse
import logging
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_path', default='bert-base-uncased', 
            help='Pre-trained ckpt path / model name (hugginface)')
    parser.add_argument('--ckpt', default='', 
            help='Checkpoint name')
    parser.add_argument('--pooler', default='entity', choices=['cls', 'entity'], 
            help='Sentence representation pooler')
    parser.add_argument('--only_test', action='store_true', 
            help='Only run test')
    parser.add_argument('--mask_entity', action='store_true', 
            help='Mask entity mentions')

    # Data
    parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
            help='Metric for picking up best checkpoint')
    parser.add_argument('--dataset', default='none', choices=['none', 'semeval', 'wiki80', 'tacred'], 
            help='Dataset. If not none, the following args can be ignored')
    parser.add_argument('--train_file', default='', type=str,
            help='Training data file')
    parser.add_argument('--val_file', default='', type=str,
            help='Validation data file')
    parser.add_argument('--test_file', default='', type=str,
            help='Test data file')
    parser.add_argument('--rel2id_file', default='', type=str,
            help='Relation to ID file')

    # Hyper-parameters
    parser.add_argument('--batch_size', default=64, type=int,
            help='Batch size')
    parser.add_argument('--lr', default=2e-5, type=float,
            help='Learning rate')
    parser.add_argument('--max_length', default=128, type=int,
            help='Maximum sentence length')
    parser.add_argument('--max_epoch', default=3, type=int,
            help='Max number of training epochs')
    parser.add_argument('--use_class_weights', action='store_true',
            help='Use class-weighted loss to address class imbalance (reduces Na over-prediction)')

    # Seed
    parser.add_argument('--seed', default=42, type=int,
            help='Seed')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Some basic settings
    root_path = '.'
    sys.path.append(root_path)
    if not os.path.exists('ckpt'):
        os.mkdir('ckpt')
    if len(args.ckpt) == 0:
        args.ckpt = '{}_{}_{}'.format(args.dataset, args.pretrain_path, args.pooler)
    ckpt = 'ckpt/{}.pth.tar'.format(args.ckpt)

    if args.dataset != 'none':
        opennre.download(args.dataset, root_path=root_path)
        args.train_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_train.txt'.format(args.dataset))
        args.val_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_val.txt'.format(args.dataset))
        args.test_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_test.txt'.format(args.dataset))
        if not os.path.exists(args.test_file):
            logging.warn("Test file {} does not exist! Use val file instead".format(args.test_file))
            args.test_file = args.val_file
        args.rel2id_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_rel2id.json'.format(args.dataset))
        if args.dataset == 'wiki80':
            args.metric = 'acc'
        else:
            args.metric = 'micro_f1'
    else:
        if not (os.path.exists(args.train_file) and os.path.exists(args.val_file) and os.path.exists(args.test_file) and os.path.exists(args.rel2id_file)):
            raise Exception('--train_file, --val_file, --test_file and --rel2id_file are not specified or files do not exist. Or specify --dataset')

    logging.info('Arguments:')
    for arg in vars(args):
        logging.info('    {}: {}'.format(arg, getattr(args, arg)))

    rel2id = json.load(open(args.rel2id_file))

    # Define the sentence encoder
    if args.pooler == 'entity':
        sentence_encoder = opennre.encoder.BERTEntityEncoder(
            max_length=args.max_length, 
            pretrain_path=args.pretrain_path,
            mask_entity=args.mask_entity
        )
    elif args.pooler == 'cls':
        sentence_encoder = opennre.encoder.BERTEncoder(
            max_length=args.max_length, 
            pretrain_path=args.pretrain_path,
            mask_entity=args.mask_entity
        )
    else:
        raise NotImplementedError

    # Define the model
    model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

    # Define the whole training framework
    framework = opennre.framework.SentenceRE(
        train_path=args.train_file,
        val_path=args.val_file,
        test_path=args.test_file,
        model=model,
        ckpt=ckpt,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        lr=args.lr,
        opt='adamw',
        use_class_weights=args.use_class_weights
    )

    # Train the model
    if not args.only_test:
        framework.train_model('macro_f1')

    # Test
    framework.load_state_dict(torch.load(ckpt)['state_dict'])
    result = framework.eval_model(framework.test_loader)

    # Print the result
    logging.info('=' * 80)
    logging.info('TEST SET RESULTS')
    logging.info('=' * 80)
    logging.info('Overall Accuracy: {:.4f}'.format(result['acc']))
    logging.info('')
    logging.info('Micro metrics (non-Na only):')
    logging.info('  Precision: {:.4f}'.format(result['micro_p']))
    logging.info('  Recall: {:.4f}'.format(result['micro_r']))
    logging.info('  F1: {:.4f}'.format(result['micro_f1']))
    logging.info('')
    logging.info('Macro metrics (all classes including Na):')
    logging.info('  Accuracy: {:.4f}'.format(result['macro_acc']))
    logging.info('  Precision: {:.4f}'.format(result['macro_p']))
    logging.info('  Recall: {:.4f}'.format(result['macro_r']))
    logging.info('  F1: {:.4f}'.format(result['macro_f1']))
    logging.info('=' * 80)

    # Save predictions to file for inspection
    logging.info('Saving predictions to predictions.json...')
    predictions = framework.get_predictions(framework.test_loader)

    # Create id2rel mapping (reverse of rel2id)
    id2rel = {v: k for k, v in model.rel2id.items()}

    # Save predictions with original text for inspection
    import json
    with open('predictions.json', 'w', encoding='utf-8') as f:
        for i, pred_id in enumerate(predictions):
            item = framework.test_loader.dataset.data[i]
            pred_relation = id2rel[pred_id]
            gold_relation = item['relation']

            output = {
                'text': item['text'],
                'head': item['h'],
                'tail': item['t'],
                'gold_relation': gold_relation,
                'predicted_relation': pred_relation,
                'correct': gold_relation == pred_relation
            }
            f.write(json.dumps(output, ensure_ascii=False) + '\n')

    logging.info('Predictions saved to predictions.json')

    # Print some example predictions
    logging.info('\n=== Sample Predictions ===')
    with open('predictions.json', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10:  # Show first 10
                break
            pred = json.loads(line)
            status = '✓' if pred['correct'] else '✗'
            logging.info(f"\n{status} Text: {pred['text'][:100]}...")
            logging.info(f"  Head: {pred['head']['name']} ({pred['head']['type']})")
            logging.info(f"  Tail: {pred['tail']['name']} ({pred['tail']['type']})")
            logging.info(f"  Gold: {pred['gold_relation']}")
            logging.info(f"  Pred: {pred['predicted_relation']}")
