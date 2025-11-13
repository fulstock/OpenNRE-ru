#!/usr/bin/env python3
"""
Analyze the training vs test performance gap.
"""
import json
from collections import Counter

# Load predictions
print("=" * 100)
print("PERFORMANCE GAP ANALYSIS")
print("=" * 100)

# Load test data and predictions
test_data = [json.loads(l) for l in open('test.txt')]
predictions = [json.loads(l) for l in open('predictions.json')]

# Overall stats
total_test = len(test_data)
na_count_gold = sum(1 for d in test_data if d['relation'] == 'Na')
non_na_count_gold = total_test - na_count_gold

na_count_pred = sum(1 for p in predictions if p['predicted_relation'] == 'Na')
non_na_count_pred = len(predictions) - na_count_pred

print(f"\nTest Set Composition:")
print(f"  Total examples: {total_test}")
print(f"  Gold Na: {na_count_gold} ({na_count_gold/total_test*100:.1f}%)")
print(f"  Gold non-Na: {non_na_count_gold} ({non_na_count_gold/total_test*100:.1f}%)")

print(f"\nModel Predictions:")
print(f"  Predicted Na: {na_count_pred} ({na_count_pred/total_test*100:.1f}%)")
print(f"  Predicted non-Na: {non_na_count_pred} ({non_na_count_pred/total_test*100:.1f}%)")

# Performance on Na vs non-Na
na_predictions = [p for p in predictions if p['gold_relation'] == 'Na']
non_na_predictions = [p for p in predictions if p['gold_relation'] != 'Na']

na_correct = sum(1 for p in na_predictions if p['correct'])
non_na_correct = sum(1 for p in non_na_predictions if p['correct'])

print(f"\nPerformance Breakdown:")
print(f"  ⭐ RELATION EXTRACTION ACCURACY (non-Na): {non_na_correct}/{len(non_na_predictions)} = {non_na_correct/len(non_na_predictions)*100:.1f}%")
print(f"  Accuracy on Na examples: {na_correct}/{len(na_predictions)} = {na_correct/len(na_predictions)*100:.1f}%")
print(f"  Overall accuracy (including Na): {(na_correct+non_na_correct)/len(predictions)*100:.1f}%")

# Analyze where non-Na examples are being misclassified
print(f"\n" + "=" * 100)
print("NON-NA MISCLASSIFICATION ANALYSIS")
print("=" * 100)

# Count how many true relations are predicted as Na
non_na_as_na = sum(1 for p in non_na_predictions if p['predicted_relation'] == 'Na')
non_na_as_wrong_rel = sum(1 for p in non_na_predictions if p['predicted_relation'] != 'Na' and not p['correct'])

print(f"\nWhen model gets non-Na examples wrong:")
print(f"  Predicted as Na (false negative): {non_na_as_na}/{len(non_na_predictions)} ({non_na_as_na/len(non_na_predictions)*100:.1f}%)")
print(f"  Predicted as wrong relation: {non_na_as_wrong_rel}/{len(non_na_predictions)} ({non_na_as_wrong_rel/len(non_na_predictions)*100:.1f}%)")
print(f"  Predicted correctly: {non_na_correct}/{len(non_na_predictions)} ({non_na_correct/len(non_na_predictions)*100:.1f}%)")

# Analyze false positives (Na predicted as non-Na)
na_as_non_na = sum(1 for p in na_predictions if p['predicted_relation'] != 'Na')
print(f"\nFalse positives (Na predicted as relation): {na_as_non_na}/{len(na_predictions)} ({na_as_non_na/len(na_predictions)*100:.1f}%)")

# Most commonly confused relations
print(f"\n" + "=" * 100)
print("MOST COMMON CONFUSION PATTERNS")
print("=" * 100)

confusions = Counter()
for p in non_na_predictions:
    if not p['correct']:
        confusions[(p['gold_relation'], p['predicted_relation'])] += 1

print("\nTop 20 confusion pairs (gold → predicted):")
for (gold, pred), count in confusions.most_common(20):
    print(f"  {gold:20s} → {pred:20s}: {count:3d} times")

# Relation frequency analysis
print(f"\n" + "=" * 100)
print("RELATION FREQUENCY vs ACCURACY")
print("=" * 100)

relation_stats = {}
for p in non_na_predictions:
    rel = p['gold_relation']
    if rel not in relation_stats:
        relation_stats[rel] = {'total': 0, 'correct': 0}
    relation_stats[rel]['total'] += 1
    if p['correct']:
        relation_stats[rel]['correct'] += 1

# Sort by frequency
sorted_rels = sorted(relation_stats.items(), key=lambda x: x[1]['total'], reverse=True)

print(f"\n{'Relation':<25s} {'Count':>7s} {'Accuracy':>10s}")
print("-" * 45)
for rel, stats in sorted_rels[:30]:
    acc = stats['correct'] / stats['total'] * 100
    print(f"{rel:<25s} {stats['total']:>7d} {acc:>9.1f}%")
