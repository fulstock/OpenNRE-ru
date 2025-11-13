#!/usr/bin/env python3
"""
Predict relations using a trained OpenNRE model.

Usage:
    python predict.py --model_path ckpt/bert_entity.pth.tar \
                      --test_file test.txt \
                      --rel2id_file nerel-rel2id.json \
                      --output predictions.json \
                      --pretrain_path DeepPavlov/rubert-base-cased

This will load the trained model and make predictions on the test file,
saving them to a JSON file with the original text and entities.
"""

import argparse
import json
import logging
import torch
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import opennre

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='ckpt/bert_entity.pth.tar',
                        help='Path to trained model checkpoint')
    parser.add_argument('--test_file', default='test.txt',
                        help='Test file in TACRED format')
    parser.add_argument('--rel2id_file', default='nerel-rel2id.json',
                        help='Relation to ID mapping file')
    parser.add_argument('--output', default='predictions.json',
                        help='Output file for predictions')
    parser.add_argument('--pretrain_path', default='DeepPavlov/rubert-base-cased',
                        help='Pre-trained BERT model path')
    parser.add_argument('--pooler', default='entity', choices=['cls', 'entity'],
                        help='Pooling method')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for prediction')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Check if model exists
    if not os.path.exists(args.model_path):
        logging.error(f"Model file not found: {args.model_path}")
        logging.error("Please train a model first using train_supervised_bert.py")
        return

    # Check if test file exists
    if not os.path.exists(args.test_file):
        logging.error(f"Test file not found: {args.test_file}")
        return

    # Load relation mapping
    logging.info(f"Loading relation mapping from {args.rel2id_file}")
    with open(args.rel2id_file, 'r', encoding='utf-8') as f:
        rel2id = json.load(f)

    id2rel = {v: k for k, v in rel2id.items()}

    # Create model
    logging.info(f"Creating model with {args.pooler} pooler")
    if args.pooler == 'entity':
        sentence_encoder = opennre.encoder.BERTEntityEncoder(
            max_length=args.max_length,
            pretrain_path=args.pretrain_path
        )
    else:
        sentence_encoder = opennre.encoder.BERTEncoder(
            max_length=args.max_length,
            pretrain_path=args.pretrain_path
        )

    model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

    # Load trained weights
    logging.info(f"Loading model from {args.model_path}")
    if torch.cuda.is_available():
        state_dict = torch.load(args.model_path)['state_dict']
        model.cuda()
    else:
        state_dict = torch.load(args.model_path, map_location='cpu')['state_dict']

    model.load_state_dict(state_dict)
    model.eval()

    # Create data loader
    logging.info(f"Loading test data from {args.test_file}")
    test_loader = opennre.framework.SentenceRELoader(
        args.test_file,
        rel2id,
        sentence_encoder.tokenize,
        args.batch_size,
        False  # Don't shuffle for prediction
    )

    # Make predictions
    logging.info("Making predictions...")
    pred_result = []
    with torch.no_grad():
        t = tqdm(test_loader, desc='Predicting')
        for data in t:
            if torch.cuda.is_available():
                for i in range(len(data)):
                    try:
                        data[i] = data[i].cuda()
                    except:
                        pass
            label = data[0]
            args_data = data[1:]
            logits = model(*args_data)
            score, pred = logits.max(-1)
            for i in range(pred.size(0)):
                pred_result.append(pred[i].item())

    # Save predictions with original text
    logging.info(f"Saving predictions to {args.output}")
    correct_count = 0
    total_with_gold = 0

    with open(args.output, 'w', encoding='utf-8') as f:
        for i, pred_id in enumerate(pred_result):
            item = test_loader.dataset.data[i]
            pred_relation = id2rel[pred_id]
            gold_relation = item['relation']
            output = {
                'text': item['text'],
                'head': item['h'],
                'tail': item['t'],
<<<<<<< HEAD
                'predicted_relation': pred_relation
            }

            # Only include gold_relation and correct if gold is not 'Na'
            if gold_relation != 'Na':
                is_correct = gold_relation == pred_relation
                output['gold_relation'] = gold_relation
                output['correct'] = is_correct

                if is_correct:
                    correct_count += 1
                total_with_gold += 1

            f.write(json.dumps(output, ensure_ascii=False) + '\n')

    logging.info(f"Predictions saved to {args.output}")
    if total_with_gold > 0:
        accuracy = correct_count / total_with_gold
        logging.info(f"Accuracy on examples with gold labels: {accuracy:.4f} ({correct_count}/{total_with_gold})")
    else:
        logging.info("No gold labels found (inference on new data)")

    # Show some example predictions
    logging.info('\n=== Sample Predictions ===')
    with open(args.output, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10:  # Show first 10
                break
            pred = json.loads(line)

            # Check if gold labels exist
            if 'gold_relation' in pred:
                status = '✓' if pred['correct'] else '✗'
            else:
                status = '•'  # No gold label
            text = pred['text']
            if len(text) > 100:
                text = text[:100] + '...'
            logging.info(f"\n{status} Text: {text}")
            logging.info(f"  Head: {pred['head']['name']} ({pred['head']['type']})")
            logging.info(f"  Tail: {pred['tail']['name']} ({pred['tail']['type']})")

            if 'gold_relation' in pred:
                logging.info(f"  Gold: {pred['gold_relation']}")
            logging.info(f"  Pred: {pred['predicted_relation']}")

    # Show error analysis (only for predictions with gold labels)
    if total_with_gold > 0:
        logging.info('\n=== Error Analysis ===')
        errors_by_relation = {}
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                pred = json.loads(line)
                if 'correct' in pred and not pred['correct']:
                    gold = pred['gold_relation']
                    pred_rel = pred['predicted_relation']
                    key = f"{gold} → {pred_rel}"
                    errors_by_relation[key] = errors_by_relation.get(key, 0) + 1

        # Show top 10 most common errors
        sorted_errors = sorted(errors_by_relation.items(), key=lambda x: x[1], reverse=True)
        logging.info("Top 10 most common prediction errors:")
        for i, (error_type, count) in enumerate(sorted_errors[:10]):
            logging.info(f"  {i+1}. {error_type}: {count} times")

if __name__ == "__main__":
    main()
