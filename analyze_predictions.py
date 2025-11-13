#!/usr/bin/env python3
"""
Analyze predictions to understand model performance.

Usage:
    python analyze_predictions.py predictions.json

This script provides detailed analysis of the model's predictions including:
- Per-relation performance metrics
- Confusion matrix
- Error patterns
- Example correct/incorrect predictions for each relation
"""

import argparse
import json
import logging
from collections import defaultdict, Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions_file', help='Path to predictions JSON file')
    parser.add_argument('--show_examples', type=int, default=5,
                        help='Number of examples to show per relation')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    # Load predictions
    logging.info(f"Loading predictions from {args.predictions_file}")
    predictions = []
    with open(args.predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(json.loads(line))

    # Calculate per-relation metrics
    relation_stats = defaultdict(lambda: {
        'tp': 0,  # True positives
        'fp': 0,  # False positives
        'fn': 0,  # False negatives
        'total_gold': 0,  # Total gold instances
        'total_pred': 0,  # Total predictions
        'correct_examples': [],
        'incorrect_examples': []
    })

    confusion = defaultdict(lambda: defaultdict(int))

    # Separate predictions with and without gold labels
    preds_with_gold = [p for p in predictions if 'gold_relation' in p]
    preds_without_gold = [p for p in predictions if 'gold_relation' not in p]

    print(f"Total predictions: {len(predictions)}")
    print(f"  With gold labels: {len(preds_with_gold)}")
    print(f"  Without gold labels (new predictions): {len(preds_without_gold)}\n")

    # Only analyze predictions with gold labels
    for pred in preds_with_gold:
        gold = pred['gold_relation']
        pred_rel = pred['predicted_relation']
        is_correct = pred['correct']

        # Update gold relation stats
        relation_stats[gold]['total_gold'] += 1
        relation_stats[pred_rel]['total_pred'] += 1

        if is_correct:
            # True positive
            relation_stats[gold]['tp'] += 1
            if len(relation_stats[gold]['correct_examples']) < args.show_examples:
                relation_stats[gold]['correct_examples'].append(pred)
        else:
            # False negative for gold, false positive for predicted
            relation_stats[gold]['fn'] += 1
            relation_stats[pred_rel]['fp'] += 1
            if len(relation_stats[gold]['incorrect_examples']) < args.show_examples:
                relation_stats[gold]['incorrect_examples'].append(pred)

        # Update confusion matrix
        confusion[gold][pred_rel] += 1

    # Calculate overall metrics (only for predictions with gold labels)
    total = len(preds_with_gold)
    correct = sum(1 for p in preds_with_gold if p['correct'])
    accuracy = correct / total

    # Calculate overall TP, FP, FN for micro averages (INCLUDING Na)
    total_tp = sum(stats['tp'] for stats in relation_stats.values())
    total_fp = sum(stats['fp'] for stats in relation_stats.values())
    total_fn = sum(stats['fn'] for stats in relation_stats.values())

    # Micro averages (INCLUDING Na)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    # Macro averages (INCLUDING Na)
    valid_relations = [rel for rel, stats in relation_stats.items() if stats['total_gold'] > 0]
    precisions = []
    recalls = []
    f1s = []

    for relation in valid_relations:
        stats = relation_stats[relation]
        precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
        recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_precision = sum(precisions) / len(precisions) if precisions else 0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0

    # ===== Calculate metrics EXCLUDING Na (the important metrics for relation extraction) =====
    # Filter out Na from calculations
    non_na_stats = {rel: stats for rel, stats in relation_stats.items() if rel != 'Na'}

    # Calculate non-Na TP, FP, FN for micro averages
    non_na_tp = sum(stats['tp'] for stats in non_na_stats.values())
    non_na_fp = sum(stats['fp'] for stats in non_na_stats.values())
    non_na_fn = sum(stats['fn'] for stats in non_na_stats.values())

    # Non-Na micro averages
    non_na_micro_precision = non_na_tp / (non_na_tp + non_na_fp) if (non_na_tp + non_na_fp) > 0 else 0
    non_na_micro_recall = non_na_tp / (non_na_tp + non_na_fn) if (non_na_tp + non_na_fn) > 0 else 0
    non_na_micro_f1 = 2 * non_na_micro_precision * non_na_micro_recall / (non_na_micro_precision + non_na_micro_recall) if (non_na_micro_precision + non_na_micro_recall) > 0 else 0

    # Non-Na macro averages (only non-Na relations)
    non_na_valid_relations = [rel for rel, stats in non_na_stats.items() if stats['total_gold'] > 0]
    non_na_precisions = []
    non_na_recalls = []
    non_na_f1s = []

    for relation in non_na_valid_relations:
        stats = non_na_stats[relation]
        precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
        recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        non_na_precisions.append(precision)
        non_na_recalls.append(recall)
        non_na_f1s.append(f1)

    non_na_macro_precision = sum(non_na_precisions) / len(non_na_precisions) if non_na_precisions else 0
    non_na_macro_recall = sum(non_na_recalls) / len(non_na_recalls) if non_na_recalls else 0
    non_na_macro_f1 = sum(non_na_f1s) / len(non_na_f1s) if non_na_f1s else 0

    # Count non-Na predictions
    non_na_preds = [p for p in preds_with_gold if p['gold_relation'] != 'Na']
    non_na_total = len(non_na_preds)
    non_na_correct = sum(1 for p in non_na_preds if p['correct'])
    non_na_accuracy = non_na_correct / non_na_total if non_na_total > 0 else 0

    logging.info("=" * 100)
    logging.info("OVERALL METRICS (INCLUDING Na)")
    logging.info("=" * 100)
    logging.info(f"Total predictions: {total}")
    logging.info(f"Correct predictions: {correct}")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info("")
    logging.info(f"Micro Precision: {micro_precision:.4f}")
    logging.info(f"Micro Recall: {micro_recall:.4f}")
    logging.info(f"Micro F1: {micro_f1:.4f}")
    logging.info("")
    logging.info(f"Macro Precision: {macro_precision:.4f}")
    logging.info(f"Macro Recall: {macro_recall:.4f}")
    logging.info(f"Macro F1: {macro_f1:.4f}")

    logging.info("\n" + "=" * 100)
    logging.info("RELATION EXTRACTION METRICS (EXCLUDING Na) ⭐")
    logging.info("=" * 100)
    logging.info(f"Total non-Na predictions: {non_na_total}")
    logging.info(f"Correct non-Na predictions: {non_na_correct}")
    logging.info(f"Non-Na Accuracy: {non_na_accuracy:.4f}")
    logging.info("")
    logging.info(f"Non-Na Micro Precision: {non_na_micro_precision:.4f}")
    logging.info(f"Non-Na Micro Recall: {non_na_micro_recall:.4f}")
    logging.info(f"Non-Na Micro F1: {non_na_micro_f1:.4f}")
    logging.info("")
    logging.info(f"Non-Na Macro Precision: {non_na_macro_precision:.4f}")
    logging.info(f"Non-Na Macro Recall: {non_na_macro_recall:.4f}")
    logging.info(f"Non-Na Macro F1: {non_na_macro_f1:.4f}")

    # Per-relation performance
    logging.info("\n" + "=" * 100)
    logging.info("PER-RELATION PERFORMANCE")
    logging.info("=" * 100)

    # Create header
    logging.info(f"{'Relation':<20} {'TP':>4} {'FP':>4} {'FN':>4} {'Gold':>5} {'Pred':>5} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    logging.info("-" * 100)

    # Sort relations by frequency (gold instances)
    sorted_relations = sorted(relation_stats.items(), key=lambda x: x[1]['total_gold'], reverse=True)

    for relation, stats in sorted_relations:
        if stats['total_gold'] == 0 and stats['total_pred'] == 0:
            continue

        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        total_gold = stats['total_gold']
        total_pred = stats['total_pred']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        logging.info(f"{relation:<20} {tp:>4} {fp:>4} {fn:>4} {total_gold:>5} {total_pred:>5} {precision:>6.3f} {recall:>6.3f} {f1:>6.3f}")

    # Detailed per-relation analysis
    logging.info("\n" + "=" * 100)
    logging.info("DETAILED PER-RELATION ANALYSIS")
    logging.info("=" * 100)

    for relation, stats in sorted_relations:
        if stats['total_gold'] == 0:
            continue

        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        total_gold = stats['total_gold']
        total_pred = stats['total_pred']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        logging.info(f"\n{relation}")
        logging.info(f"  TP: {tp}, FP: {fp}, FN: {fn}")
        logging.info(f"  Gold instances: {total_gold}, Predicted instances: {total_pred}")
        logging.info(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

        # Show correct examples
        if stats['correct_examples']:
            logging.info(f"  ✓ Correct examples:")
            for i, ex in enumerate(stats['correct_examples'][:3]):
                text = ex['text'][:80] + '...' if len(ex['text']) > 80 else ex['text']
                logging.info(f"    {i+1}. {text}")
                logging.info(f"       {ex['head']['name']} → {ex['tail']['name']}")

        # Show incorrect examples
        if stats['incorrect_examples']:
            logging.info(f"  ✗ Error examples (predicted as):")
            for i, ex in enumerate(stats['incorrect_examples'][:3]):
                text = ex['text'][:80] + '...' if len(ex['text']) > 80 else ex['text']
                logging.info(f"    {i+1}. {text}")
                logging.info(f"       {ex['head']['name']} → {ex['tail']['name']}")
                logging.info(f"       Predicted: {ex['predicted_relation']}")

    # Confusion analysis
    logging.info("\n" + "=" * 80)
    logging.info("MOST COMMON CONFUSIONS")
    logging.info("=" * 80)

    all_confusions = []
    for gold, pred_dict in confusion.items():
        for pred_rel, count in pred_dict.items():
            if gold != pred_rel:
                all_confusions.append((gold, pred_rel, count))

    all_confusions.sort(key=lambda x: x[2], reverse=True)

    for i, (gold, pred_rel, count) in enumerate(all_confusions[:20]):
        logging.info(f"{i+1}. {gold} → {pred_rel}: {count} times")

    # Analyze entity types
    logging.info("\n" + "=" * 80)
    logging.info("ENTITY TYPE ANALYSIS")
    logging.info("=" * 80)

    entity_pair_performance = defaultdict(lambda: {'correct': 0, 'total': 0})

    for pred in preds_with_gold:
        head_type = pred['head']['type']
        tail_type = pred['tail']['type']
        pair = f"{head_type} → {tail_type}"
        entity_pair_performance[pair]['total'] += 1
        if pred['correct']:
            entity_pair_performance[pair]['correct'] += 1

    sorted_pairs = sorted(entity_pair_performance.items(), key=lambda x: x[1]['total'], reverse=True)

    logging.info("Performance by entity type pairs:")
    for pair, stats in sorted_pairs[:20]:
        accuracy = stats['correct'] / stats['total']
        logging.info(f"  {pair}: {accuracy:.3f} ({stats['correct']}/{stats['total']})")

    # Analyze by text length
    logging.info("\n" + "=" * 80)
    logging.info("PERFORMANCE BY TEXT LENGTH")
    logging.info("=" * 80)

    length_buckets = {
        'Short (< 50 chars)': [],
        'Medium (50-100 chars)': [],
        'Long (> 100 chars)': []
    }

    for pred in preds_with_gold:
        text_len = len(pred['text'])
        if text_len < 50:
            length_buckets['Short (< 50 chars)'].append(pred['correct'])
        elif text_len < 100:
            length_buckets['Medium (50-100 chars)'].append(pred['correct'])
        else:
            length_buckets['Long (> 100 chars)'].append(pred['correct'])

    for bucket_name, results in length_buckets.items():
        if results:
            accuracy = sum(results) / len(results)
            logging.info(f"  {bucket_name}: {accuracy:.3f} ({sum(results)}/{len(results)})")

    # Relations that the model handles perfectly
    logging.info("\n" + "=" * 80)
    logging.info("PERFECTLY PREDICTED RELATIONS (100% accuracy)")
    logging.info("=" * 80)

    perfect_relations = []
    for relation, stats in sorted_relations:
        if stats['total_gold'] > 0 and stats['fn'] == 0 and stats['fp'] == 0:
            perfect_relations.append((relation, stats['total_gold']))

    if perfect_relations:
        for rel, count in perfect_relations:
            logging.info(f"  {rel}: {count} examples")
    else:
        logging.info("  None")

    # Most difficult relations
    logging.info("\n" + "=" * 80)
    logging.info("MOST DIFFICULT RELATIONS (lowest F1)")
    logging.info("=" * 80)

    relation_f1 = []
    for relation, stats in relation_stats.items():
        if stats['total_gold'] > 5:  # Only consider relations with enough examples
            precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
            recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            relation_f1.append((relation, f1, stats['total_gold']))

    relation_f1.sort(key=lambda x: x[1])

    for i, (rel, f1, count) in enumerate(relation_f1[:10]):
        logging.info(f"{i+1}. {rel}: F1={f1:.3f} (n={count})")

    # CSV table for Google Sheets (NON-NA RELATIONS ONLY)
    logging.info("\n" + "=" * 100)
    logging.info("CSV TABLE FOR NON-NA RELATIONS (copy everything below)")
    logging.info("=" * 100)

    # Header
    logging.info("Relation,TP,FP,FN,Gold,Pred,Precision,Recall,F1")

    # Sort non-Na relations by frequency (gold instances) for consistent ordering
    non_na_sorted_relations = sorted(non_na_stats.items(), key=lambda x: x[1]['total_gold'], reverse=True)

    # Per-relation rows (NON-NA ONLY)
    for relation, stats in non_na_sorted_relations:
        if stats['total_gold'] == 0:
            continue

        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        total_gold = stats['total_gold']
        total_pred = stats['total_pred']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        logging.info(f"{relation},{tp},{fp},{fn},{total_gold},{total_pred},{precision:.4f},{recall:.4f},{f1:.4f}")

    # Non-Na totals and averages
    logging.info(f"TOTAL,{non_na_tp},{non_na_fp},{non_na_fn},{non_na_total},-,-,-,-")
    logging.info(f"MICRO_AVG,-,-,-,-,-,{non_na_micro_precision:.4f},{non_na_micro_recall:.4f},{non_na_micro_f1:.4f}")
    logging.info(f"MACRO_AVG,-,-,-,-,-,{non_na_macro_precision:.4f},{non_na_macro_recall:.4f},{non_na_macro_f1:.4f}")

    # Also save to CSV file
    csv_filename = args.predictions_file.replace('.json', '_non_na_metrics.csv')
    with open(csv_filename, 'w', encoding='utf-8') as csv_file:
        csv_file.write("Relation,TP,FP,FN,Gold,Pred,Precision,Recall,F1\n")
        for relation, stats in non_na_sorted_relations:
            if stats['total_gold'] == 0:
                continue
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']
            total_gold = stats['total_gold']
            total_pred = stats['total_pred']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            csv_file.write(f"{relation},{tp},{fp},{fn},{total_gold},{total_pred},{precision:.4f},{recall:.4f},{f1:.4f}\n")
        csv_file.write(f"TOTAL,{non_na_tp},{non_na_fp},{non_na_fn},{non_na_total},-,-,-,-\n")
        csv_file.write(f"MICRO_AVG,-,-,-,-,-,{non_na_micro_precision:.4f},{non_na_micro_recall:.4f},{non_na_micro_f1:.4f}\n")
        csv_file.write(f"MACRO_AVG,-,-,-,-,-,{non_na_macro_precision:.4f},{non_na_macro_recall:.4f},{non_na_macro_f1:.4f}\n")

    logging.info(f"\n✓ CSV saved to: {csv_filename}")

if __name__ == "__main__":
    main()
