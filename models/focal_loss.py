import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None, reduction='mean'):
        """
        Args:
          gamma (float): The focusing parameter. Higher values (e.g., 2.0)
                         give more weight to hard-to-classify examples.
          pos_weight (torch.Tensor, optional): A weight for positive examples.
                                             Shape (num_classes,).
          reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, labels):
        # Calculate the standard BCE loss (without reduction)
        if self.pos_weight is not None:
            pw = self.pos_weight.to(logits.device)
            log_sig_x = F.logsigmoid(logits)
            log_one_minus_sig_x = F.logsigmoid(-logits) # log(1 - sigmoid(x))
            weighted_bce_loss = - (pw * labels * log_sig_x + (1 - labels) * log_one_minus_sig_x)
        else:
            weighted_bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        
        # Calculate p_t (probability of the true class)
        probs = torch.sigmoid(logits)
        p_t = torch.where(labels > 0, probs, 1 - probs)

        # Calculate focal modulation factor
        focal_term = (1 - p_t) ** self.gamma

        # Apply focal loss modulation
        loss = focal_term * weighted_bce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
