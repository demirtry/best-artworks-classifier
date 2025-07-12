from typing import Optional

import torch
import torch.nn as nn


class SoftF1Loss(nn.Module):
    """
    Soft F1 Loss implementation for multi-class classification tasks.
    This loss function approximates the F1 score using soft probabilities
    instead of hard predictions, making it differentiable and suitable for training.
    :param epsilon: Small constant to avoid division by zero
    :param class_weights: Optional tensor of weights for each class to handle class imbalance
    """
    def __init__(
            self,
            epsilon: float = 1e-6,
            class_weights: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.class_weights = class_weights

    def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Soft F1 Loss between inputs and targets.
        :param inputs: Raw logits from the model (shape: [batch_size, num_classes])
        :param targets: Ground truth class indices (shape: [batch_size])
        :return: Scalar loss value
        """
        targets_onehot = torch.zeros_like(inputs)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1)

        probs = torch.softmax(inputs, dim=1)

        tp = (targets_onehot * probs).sum(dim=0)
        fp = ((1 - targets_onehot) * probs).sum(dim=0)
        fn = (targets_onehot * (1 - probs)).sum(dim=0)

        if self.class_weights is not None:
            device = inputs.device
            weights = self.class_weights.to(device)
            tp = tp * weights
            fp = fp * weights
            fn = fn * weights

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        soft_f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)

        return 1 - soft_f1.mean()
