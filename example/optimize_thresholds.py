# coding:utf-8
"""
Optimize per-class thresholds after training.

Usage:
    python example/optimize_thresholds.py \
        --ckpt ckpt/nerel_ncrl.pth.tar \
        --pretrain_path DeepPavlov/rubert-base-cased \
        --val_file data/NEREL-sent/dev.txt \
        --test_file data/NEREL-sent/test.txt \
        --rel2id_file data/NEREL-sent/nerel-rel2id.json

This will:
1. Load trained model
2. Optimize thresholds on validation set
3. Evaluate on test set with optimized thresholds
4. Save thresholds to file
"""

import torch
import numpy as np
import json
import opennre
from opennre import encoder, model, framework
import sys
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True,
            help='Path to trained model checkpoint')
    parser.add_argument('--pretrain_path', default='bert-base-uncased',
            help='Pre-trained model name')
    parser.add_argument('--pooler', default='entity', choices=['cls', 'entity'],
            help='Sentence representation pooler')

    # Data
    parser.add_argument('--val_file', required=True, type=str,
            help='Validation data file')
    parser.add_argument('--test_file', required=True, type=str,
            help='Test data file')
    parser.add_argument('--rel2id_file', required=True, type=str,
            help='Relation to ID file')

    # Optimization
    parser.add_argument('--search_steps', default=100, type=int,
            help='Number of threshold values to search (default: 100)')
    parser.add_argument('--batch_size', default=64, type=int,
            help='Batch size')
    parser.add_argument('--max_length', default=128, type=int,
            help='Maximum sentence length')

    # Output
    parser.add_argument('--save_thresholds', default='',
            help='Path to save optimized thresholds (e.g., thresholds.npy)')

    args = parser.parse_args()

    # Load rel2id
    rel2id = json.load(open(args.rel2id_file))
    logging.info(f"Loaded {len(rel2id)} relations")

    # Define the sentence encoder
    if args.pooler == 'entity':
        sentence_encoder = opennre.encoder.BERTEntityEncoder(
            max_length=args.max_length,
            pretrain_path=args.pretrain_path
        )
    elif args.pooler == 'cls':
        sentence_encoder = opennre.encoder.BERTEncoder(
            max_length=args.max_length,
            pretrain_path=args.pretrain_path
        )
    else:
        raise NotImplementedError

    # Define the model
    model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

    # Load trained model
    logging.info(f"Loading model from {args.ckpt}")
    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    logging.info("Model loaded successfully")

    # Create data loaders (no negative sampling for evaluation)
    val_loader = opennre.framework.SentenceRELoader(
        args.val_file,
        rel2id,
        sentence_encoder.tokenize,
        args.batch_size,
        False,  # no shuffle
        negative_ratio=None  # no sampling
    )

    test_loader = opennre.framework.SentenceRELoader(
        args.test_file,
        rel2id,
        sentence_encoder.tokenize,
        args.batch_size,
        False,  # no shuffle
        negative_ratio=None  # no sampling
    )

    # Optimize thresholds on validation set
    logging.info("=" * 80)
    logging.info("OPTIMIZING THRESHOLDS ON VALIDATION SET")
    logging.info("=" * 80)

    threshold_optimizer = opennre.framework.PerClassThresholdOptimizer(
        num_classes=len(rel2id),
        search_steps=args.search_steps
    )

    optimized_thresholds = threshold_optimizer.optimize(model, val_loader)

    # Save thresholds if requested
    if args.save_thresholds:
        threshold_optimizer.save_thresholds(args.save_thresholds)

    # Evaluate on validation set with optimized thresholds
    logging.info("=" * 80)
    logging.info("VALIDATION SET RESULTS (with optimized thresholds)")
    logging.info("=" * 80)

    val_predictions = []
    with torch.no_grad():
        for data in val_loader:
            if torch.cuda.is_available():
                for i in range(len(data)):
                    try:
                        data[i] = data[i].cuda()
                    except:
                        pass

            args_data = data[1:]
            logits = model(*args_data)

            # Use optimized thresholds
            preds = threshold_optimizer.predict(logits)

            for i in range(preds.size(0)):
                val_predictions.append(preds[i].item())

    val_result = val_loader.dataset.eval(val_predictions)

    logging.info('Validation Results (optimized thresholds):')
    logging.info('  Accuracy: {:.4f}'.format(val_result['acc']))
    logging.info('  Micro F1: {:.4f}'.format(val_result['micro_f1']))
    logging.info('  Macro F1: {:.4f}'.format(val_result['macro_f1']))

    # Evaluate on test set with optimized thresholds
    logging.info("=" * 80)
    logging.info("TEST SET RESULTS (with optimized thresholds)")
    logging.info("=" * 80)

    test_predictions = []
    with torch.no_grad():
        for data in test_loader:
            if torch.cuda.is_available():
                for i in range(len(data)):
                    try:
                        data[i] = data[i].cuda()
                    except:
                        pass

            args_data = data[1:]
            logits = model(*args_data)

            # Use optimized thresholds
            preds = threshold_optimizer.predict(logits)

            for i in range(preds.size(0)):
                test_predictions.append(preds[i].item())

    test_result = test_loader.dataset.eval(test_predictions)

    logging.info('Test Results (optimized thresholds):')
    logging.info('  Accuracy: {:.4f}'.format(test_result['acc']))
    logging.info('  Micro F1: {:.4f}'.format(test_result['micro_f1']))
    logging.info('  Macro F1: {:.4f}'.format(test_result['macro_f1']))

    # For comparison, also evaluate with standard argmax
    logging.info("=" * 80)
    logging.info("TEST SET RESULTS (standard argmax for comparison)")
    logging.info("=" * 80)

    test_predictions_baseline = []
    with torch.no_grad():
        for data in test_loader:
            if torch.cuda.is_available():
                for i in range(len(data)):
                    try:
                        data[i] = data[i].cuda()
                    except:
                        pass

            args_data = data[1:]
            logits = model(*args_data)

            # Standard argmax
            _, preds = logits.max(dim=-1)

            for i in range(preds.size(0)):
                test_predictions_baseline.append(preds[i].item())

    test_result_baseline = test_loader.dataset.eval(test_predictions_baseline)

    logging.info('Test Results (standard argmax):')
    logging.info('  Accuracy: {:.4f}'.format(test_result_baseline['acc']))
    logging.info('  Micro F1: {:.4f}'.format(test_result_baseline['micro_f1']))
    logging.info('  Macro F1: {:.4f}'.format(test_result_baseline['macro_f1']))

    # Compare
    logging.info("=" * 80)
    logging.info("IMPROVEMENT FROM THRESHOLD OPTIMIZATION")
    logging.info("=" * 80)
    logging.info('Accuracy: {:.4f} -> {:.4f} ({:+.4f})'.format(
        test_result_baseline['acc'], test_result['acc'],
        test_result['acc'] - test_result_baseline['acc']))
    logging.info('Micro F1: {:.4f} -> {:.4f} ({:+.4f})'.format(
        test_result_baseline['micro_f1'], test_result['micro_f1'],
        test_result['micro_f1'] - test_result_baseline['micro_f1']))
    logging.info('Macro F1: {:.4f} -> {:.4f} ({:+.4f})'.format(
        test_result_baseline['macro_f1'], test_result['macro_f1'],
        test_result['macro_f1'] - test_result_baseline['macro_f1']))
    logging.info("=" * 80)
