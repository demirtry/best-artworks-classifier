import torch
import torch.nn.functional as F
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1, weights=None):
        """
        Args:
            num_classes (int): количество классов
            smoothing (float): степень сглаживания (default: 0.1)
            weights (Tensor): веса классов, shape [num_classes], опционально
        """
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.register_buffer('weight', weights)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C] — логиты (без softmax)
            targets: [B] — истинные метки
        """
        log_probs = F.log_softmax(inputs, dim=-1)

        with torch.no_grad():
            targets = targets.view(-1, 1)
            true_dist = torch.zeros_like(inputs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets, 1 - self.smoothing)

        if self.weight is not None:
            weights = self.weight[None, :]
            weighted_true_dist = true_dist * weights
        else:
            weighted_true_dist = true_dist

        loss = -(weighted_true_dist * log_probs).sum(dim=-1).mean()

        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        B, D = embeddings.shape
        dist = torch.cdist(embeddings, embeddings)

        labels_eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels_ne = 1 - labels_eq

        hardest_positive = (dist * labels_eq).max(dim=1).values
        hardest_negative = (dist * labels_ne + 2 * dist.max() * labels_eq).min(dim=1).values

        loss = torch.relu(hardest_positive - hardest_negative + self.margin)
        return loss.mean()
