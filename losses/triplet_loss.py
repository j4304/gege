import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Element-Wise Triplet Loss (SQUARED EUCLIDEAN).

    Strictly respects the (Anchor, Positive, Negative) pairings provided
    by the DataLoader's offline hard negative miner. 
    Does NOT cross-compare across the batch, completely preventing class collisions.
    """
    def __init__(self, margin=0.25, mode='euclidean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mode = mode.lower()
        self.last_fraction_active = 0.0

    def forward(self, anchor, positive, negative):
        # 1. Distance between Anchor and Positive (Element-wise SQUARED)
        if self.mode == 'euclidean':
            dist_pos = torch.sum(torch.pow(anchor - positive, 2), dim=1) 
        else:
            dist_pos = 1.0 - F.cosine_similarity(anchor, positive)

        # 2. Distance between Anchor and its Explicit Negative (Element-wise SQUARED)
        if self.mode == 'euclidean':
            dist_neg = torch.sum(torch.pow(anchor - negative, 2), dim=1)
        else:
            dist_neg = 1.0 - F.cosine_similarity(anchor, negative)

        # 3. Compute standard Hinge Loss strictly on the provided pairs
        losses = F.relu(dist_pos - dist_neg + self.margin)

        # --- Active triplet filtering ---
        active_mask = losses > 0
        active_triplets = losses[active_mask]

        self.last_fraction_active = active_mask.float().mean().item()

        if active_triplets.numel() > 0:
            return active_triplets.mean()
        else:
            return losses.mean()