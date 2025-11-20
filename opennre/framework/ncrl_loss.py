"""
None Class Ranking Loss (NCRL) for Imbalanced Relation Extraction

Based on: Zhou & Lee (2022) - "None Class Ranking Loss for Document-Level Relation Extraction"

Paper: https://arxiv.org/abs/2205.00476
GitHub: https://github.com/yangzhou12/NCRL

CORRECT IMPLEMENTATION following the paper's formulation (Equations 2, 5, 6, 7).

Key Innovation:
- Treats the "Na" (none) class as an adaptive, learnable threshold
- Maximizes margin between none class score (f0) and each relation score (fi)
- Provides instance-dependent thresholding without manual tuning
- Bayes consistent with respect to None Class Ranking Error (NCRE)

Mathematical Formulation:
1. Positive margin: m+_i = fi - f0 (should be > 0 for positive relations)
2. Negative margin: m-_i = f0 - fi (should be > 0 for negative relations)
3. Loss: L_NA = -Σ[yi·log σ(m+_i) + (1-yi)·log σ(m-_i)]
4. Margin regularization: Ensures f0 > avg(fi) for Na instances
5. Margin shifting: Attenuates hard negatives (γ parameter)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class NCRLLoss(nn.Module):
    """
    None Class Ranking Loss (NCRL)

    Maximizes the margin between the none class (Na) score and each
    pre-defined relation class score. Positive relations should have
    scores > f0 (none class score), negative relations should have
    scores < f0.

    Key Equation (Equation 2 from paper):
        L_NA(y, f) = -Σ[yi·log σ(fi-f0) + (1-yi)·log σ(f0-fi)]

    Where:
    - f0: Score of the none class (Na)
    - fi: Score of the i-th pre-defined relation class
    - yi: Binary label (1 if relation i exists, 0 otherwise)
    - σ(x): Sigmoid function 1/(1 + exp(-x))

    With margin regularization (Equation 5) and margin shifting (Equation 6).
    """

    def __init__(self, num_classes, gamma=0.01, use_margin_reg=True):
        """
        Args:
            num_classes: Total number of classes (including Na at index 0)
            gamma: Margin shifting parameter (0 < γ < 1), typically 0.01 or 0.05
                   Higher gamma = more tolerance for hard negatives
                   Paper recommends: 0.01 for DocRED, 0.05 for DialogRE
            use_margin_reg: Whether to use margin regularization (L_NA0)
                           Recommended: True (provides better Na class handling)
        """
        super().__init__()
        assert num_classes >= 2, "Need at least Na class (0) and one relation class"
        self.num_classes = num_classes
        self.gamma = gamma
        self.use_margin_reg = use_margin_reg

        logging.info(f"NCRL Loss initialized (CORRECT PAPER FORMULATION):")
        logging.info(f"  Num classes: {num_classes}")
        logging.info(f"  Gamma (margin shift): {gamma}")
        logging.info(f"  Margin regularization: {use_margin_reg}")
        logging.info(f"  Paper: Zhou & Lee (2022) - Equations 2, 5, 6, 7")

    def forward(self, logits, labels):
        """
        Compute NCRL loss following paper's Equations 2, 5, 6, 7.

        Args:
            logits: (batch_size, num_classes) - raw model outputs
                    Index 0 = Na class, indices 1...K = relation classes
            labels: (batch_size,) - ground truth class indices
                    0 = Na, 1...K = relation classes

        Returns:
            loss: scalar tensor
        """
        batch_size = logits.size(0)
        device = logits.device

        # Extract f0 (none class score) and fi (relation class scores)
        f0 = logits[:, 0]  # (batch_size,) - Na class scores
        f_relations = logits[:, 1:]  # (batch_size, K) - Relation class scores
        K = f_relations.size(1)  # Number of pre-defined relations

        # Create binary label mask: yi = 1 for positive class, 0 otherwise
        # Shape: (batch_size, K)
        y_mask = torch.zeros(batch_size, K, device=device)
        for b in range(batch_size):
            if labels[b] > 0:  # Not Na
                y_mask[b, labels[b] - 1] = 1  # -1 because relations start at index 1

        # ============================================================
        # Equation 2: NCRL Loss for Relation Classes
        # ============================================================
        # Compute margins:
        # Positive margin: m+_i = fi - f0 (should be > 0 for positive relations)
        # Negative margin: m-_i = f0 - fi (should be > 0 for negative relations)
        m_pos = f_relations - f0.unsqueeze(1)  # (batch_size, K)
        m_neg = f0.unsqueeze(1) - f_relations  # (batch_size, K)

        # Apply sigmoid to margins to get probabilities
        # P(yi=1|x) ∝ σ(fi - f0)  [probability relation i exists]
        # P(yi=0|x) ∝ σ(f0 - fi)  [probability relation i doesn't exist]
        p_pos = torch.sigmoid(m_pos)  # (batch_size, K)
        p_neg = torch.sigmoid(m_neg)  # (batch_size, K)

        # ============================================================
        # Equation 6: Margin Shifting
        # ============================================================
        # p-_i = min(σ(m-_i) + γ, 1)
        # This attenuates hard negatives (suspected mislabeled) and ignores easy negatives
        # Intuition: If σ(f0 - fi) is already high (easy negative), adding γ brings it to 1.0
        #           If σ(f0 - fi) is low (hard negative / possible error), γ helps reduce penalty
        p_neg_shifted = torch.clamp(p_neg + self.gamma, max=1.0)  # (batch_size, K)

        # ============================================================
        # Equation 7: NCRL Loss (simplified for relation classes)
        # ============================================================
        # L = -Σ[yi·log σ(fi-f0) + (1-yi)·log p-_i]
        #
        # Interpretation:
        # - For positive relations (yi=1): Maximize σ(fi-f0) → minimize -log σ(fi-f0)
        # - For negative relations (yi=0): Maximize σ(f0-fi) → minimize -log σ(f0-fi)
        loss_pos = -torch.log(p_pos + 1e-8) * y_mask  # (batch_size, K)
        loss_neg = -torch.log(p_neg_shifted + 1e-8) * (1 - y_mask)  # (batch_size, K)

        # Sum over all K relations per sample, then average over batch
        loss_relations = (loss_pos + loss_neg).sum(dim=1).mean()  # More explicit averaging

        # ============================================================
        # Equation 5: Margin Regularization for Na Class
        # ============================================================
        if self.use_margin_reg:
            # For Na instances (y0=1): maximize f0 - avg(fi)
            # For non-Na instances (y0=0): maximize avg(fi) - f0
            #
            # Why average instead of max?
            # Paper: "Directly maximizing f0 - max(fi) can lead to unstable results...
            #         We maximize f0 - (1/K)Σfi, which is an upper-bound of f0 - max(fi)."

            y0_mask = (labels == 0).float()  # (batch_size,) - 1 if Na, 0 otherwise

            # Average relation scores
            f_avg = f_relations.mean(dim=1)  # (batch_size,)

            # Compute margins for Na class
            # For Na instances: m0+ = f0 - avg(fi) should be > 0
            # For non-Na instances: m0- = avg(fi) - f0 should be > 0
            m0_pos = f0 - f_avg  # (batch_size,)
            m0_neg = f_avg - f0  # (batch_size,)

            # Apply sigmoid
            p0_pos = torch.sigmoid(m0_pos)  # P(y0=1|x) for Na instances
            p0_neg = torch.sigmoid(m0_neg)  # P(y0=0|x) for non-Na instances

            # Apply margin shift to negative part
            p0_neg_shifted = torch.clamp(p0_neg + self.gamma, max=1.0)

            # L_NA0 (Equation 5)
            # For Na instances: maximize σ(f0 - avg(fi))
            # For non-Na instances: maximize σ(avg(fi) - f0)
            loss_na_pos = -torch.log(p0_pos + 1e-8) * y0_mask  # (batch_size,)
            loss_na_neg = -torch.log(p0_neg_shifted + 1e-8) * (1 - y0_mask)  # (batch_size,)

            # Average over batch (consistent with loss_relations)
            loss_na_reg = (loss_na_pos + loss_na_neg).mean()

            # Combine losses
            total_loss = loss_relations + loss_na_reg
        else:
            # Use only relation class loss
            total_loss = loss_relations

        return total_loss

    def get_inference_predictions(self, logits):
        """
        Get predictions using NCRL inference strategy.

        Paper's inference rule:
        - Predict relations where fi > f0 (none class score)
        - If no relation scores > f0, predict Na
        - If multiple relations > f0, predict the one with highest score

        Args:
            logits: (batch_size, num_classes) - model outputs

        Returns:
            pred: (batch_size,) - predicted class indices
        """
        batch_size = logits.size(0)
        device = logits.device

        # Extract f0 and relation scores
        f0 = logits[:, 0].unsqueeze(1)  # (batch_size, 1)
        f_relations = logits[:, 1:]  # (batch_size, K)

        # Find relations where fi > f0
        above_threshold = (f_relations > f0).float()  # (batch_size, K)

        # For each sample, get the max relation score among those above threshold
        f_relations_masked = f_relations * above_threshold
        max_scores, max_indices = f_relations_masked.max(dim=1)  # (batch_size,)

        # If any relation is above threshold, predict it; otherwise predict Na (0)
        has_relation = (above_threshold.sum(dim=1) > 0)  # (batch_size,)
        pred = torch.where(has_relation,
                          max_indices + 1,  # +1 because Na is at index 0
                          torch.zeros_like(max_indices))

        return pred
