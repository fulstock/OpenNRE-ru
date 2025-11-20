"""
Per-Class Threshold Optimization for Relation Extraction

After training with NCRL, we can further optimize per-class decision thresholds
on the validation set to maximize macro F1 score.

Strategy:
1. Train model with NCRL loss
2. On validation set, for each class, find optimal threshold that maximizes F1
3. Use optimized thresholds at test time

This is complementary to NCRL:
- NCRL provides adaptive threshold (f0) during training
- Per-class thresholds fine-tune decision boundaries per relation
"""

import torch
import numpy as np
import logging
from tqdm import tqdm


class PerClassThresholdOptimizer:
    """
    Optimizes per-class decision thresholds to maximize F1 scores.

    Usage:
        1. Train model with NCRL
        2. optimizer = PerClassThresholdOptimizer(num_classes)
        3. optimizer.optimize(model, val_loader)
        4. predictions = optimizer.predict(logits)
    """

    def __init__(self, num_classes, search_steps=100):
        """
        Args:
            num_classes: Total number of classes (including Na at index 0)
            search_steps: Number of threshold values to try (default: 100)
        """
        self.num_classes = num_classes
        self.search_steps = search_steps

        # Initialize thresholds to 0.5 (will be optimized)
        self.thresholds = np.ones(num_classes) * 0.5

        logging.info(f"Per-Class Threshold Optimizer initialized:")
        logging.info(f"  Num classes: {num_classes}")
        logging.info(f"  Search steps: {search_steps}")

    def optimize(self, model, val_loader):
        """
        Optimize per-class thresholds on validation set.

        Args:
            model: Trained model (should be in eval mode)
            val_loader: Validation data loader

        Returns:
            optimized_thresholds: (num_classes,) array of optimal thresholds
        """
        logging.info("Optimizing per-class thresholds on validation set...")

        model.eval()

        # Step 1: Collect all logits and labels from validation set
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for data in tqdm(val_loader, desc="Collecting predictions"):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass

                label = data[0]
                args = data[1:]

                logits = model(*args)

                all_logits.append(logits.cpu())
                all_labels.append(label.cpu())

        # Concatenate all batches
        all_logits = torch.cat(all_logits, dim=0)  # (N, num_classes)
        all_labels = torch.cat(all_labels, dim=0)  # (N,)

        N = all_logits.size(0)
        logging.info(f"Collected {N} validation samples")

        # Step 2: Convert logits to probabilities
        all_probs = torch.softmax(all_logits, dim=1).numpy()  # (N, num_classes)
        all_labels = all_labels.numpy()  # (N,)

        # Step 3: For each class, find optimal threshold
        logging.info("Searching for optimal thresholds per class...")

        for class_id in tqdm(range(self.num_classes), desc="Optimizing thresholds"):
            # Get probabilities for this class
            class_probs = all_probs[:, class_id]  # (N,)

            # Binary labels: 1 if true class is class_id, 0 otherwise
            binary_labels = (all_labels == class_id).astype(int)  # (N,)

            # Skip if no positive examples for this class
            if binary_labels.sum() == 0:
                logging.warning(f"Class {class_id}: No positive examples in validation set. Keeping default threshold.")
                continue

            # Try different thresholds and find the one that maximizes F1
            best_threshold = 0.5
            best_f1 = 0.0

            # Search space: min_prob to max_prob
            min_prob = class_probs.min()
            max_prob = class_probs.max()
            threshold_candidates = np.linspace(min_prob, max_prob, self.search_steps)

            for threshold in threshold_candidates:
                # Predict: 1 if prob > threshold, 0 otherwise
                preds = (class_probs > threshold).astype(int)

                # Calculate F1 score
                tp = ((preds == 1) & (binary_labels == 1)).sum()
                fp = ((preds == 1) & (binary_labels == 0)).sum()
                fn = ((preds == 0) & (binary_labels == 1)).sum()

                if tp == 0:
                    f1 = 0.0
                else:
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

            self.thresholds[class_id] = best_threshold

            # Log statistics
            num_positive = binary_labels.sum()
            logging.info(f"Class {class_id}: threshold={best_threshold:.4f}, F1={best_f1:.4f}, positives={num_positive}")

        logging.info("Threshold optimization complete!")
        logging.info(f"Threshold range: [{self.thresholds.min():.4f}, {self.thresholds.max():.4f}]")
        logging.info(f"Mean threshold: {self.thresholds.mean():.4f}")

        return self.thresholds

    def predict(self, logits):
        """
        Make predictions using optimized per-class thresholds.

        Strategy:
        1. Convert logits to probabilities
        2. For each class, check if prob > threshold
        3. If multiple classes pass threshold, take highest probability
        4. If no class passes threshold, predict Na (class 0)

        Args:
            logits: (batch_size, num_classes) - model outputs

        Returns:
            predictions: (batch_size,) - predicted class indices
        """
        # Convert to probabilities
        probs = torch.softmax(logits, dim=1)  # (batch_size, num_classes)

        batch_size = probs.size(0)
        device = probs.device

        # Convert thresholds to tensor
        thresholds = torch.tensor(self.thresholds, device=device).unsqueeze(0)  # (1, num_classes)

        # Check which classes pass their thresholds
        above_threshold = (probs > thresholds).float()  # (batch_size, num_classes)

        # Mask probabilities: keep only those above threshold
        masked_probs = probs * above_threshold  # (batch_size, num_classes)

        # For each sample, predict class with highest masked probability
        max_probs, predictions = masked_probs.max(dim=1)  # (batch_size,)

        # If no class passes threshold (max_prob == 0), predict Na (class 0)
        predictions = torch.where(max_probs > 0, predictions, torch.zeros_like(predictions))

        return predictions

    def save_thresholds(self, path):
        """Save optimized thresholds to file."""
        np.save(path, self.thresholds)
        logging.info(f"Saved thresholds to {path}")

    def load_thresholds(self, path):
        """Load optimized thresholds from file."""
        self.thresholds = np.load(path)
        logging.info(f"Loaded thresholds from {path}")
        logging.info(f"Threshold range: [{self.thresholds.min():.4f}, {self.thresholds.max():.4f}]")


def optimize_thresholds_simple(model, val_loader, num_classes, search_steps=100):
    """
    Convenience function for one-shot threshold optimization.

    Args:
        model: Trained model
        val_loader: Validation data loader
        num_classes: Total number of classes
        search_steps: Number of threshold values to try

    Returns:
        thresholds: (num_classes,) array of optimal thresholds
    """
    optimizer = PerClassThresholdOptimizer(num_classes, search_steps)
    thresholds = optimizer.optimize(model, val_loader)
    return thresholds, optimizer
