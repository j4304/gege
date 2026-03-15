import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Online Batch-Hard Triplet Loss.

    Instead of relying on the dataset to randomly guess a hard negative,
    this loss function computes the distance between every Anchor in the batch
    and every Negative in the batch. For each Anchor, it mathematically selects
    the absolute closest Negative (the Hardest Negative) to compute the loss.
    """
    def __init__(self, margin=0.25, mode='euclidean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mode = mode.lower()
        self.last_fraction_active = 0.0

    def forward(self, anchor, positive, negative):
        B = anchor.size(0)

        # 1. True Euclidean Distance between Anchor and Positive
        if self.mode == 'euclidean':
            # Calculate squared distance first
            dist_pos_sq = torch.sum(torch.pow(anchor - positive, 2), dim=1) 
            # Clamp before square root to avoid NaN gradients at distance 0
            dist_pos = torch.sqrt(torch.clamp(dist_pos_sq, min=1e-16)) # Shape: [B]
        else:
            dist_pos = 1.0 - F.cosine_similarity(anchor, positive)

        # 2. Pairwise True Euclidean Distance between ALL Anchors and Negatives
        if self.mode == 'euclidean':
            dot_product = torch.mm(anchor, negative.t())             
            anchor_norm = torch.sum(anchor ** 2, dim=1, keepdim=True)  
            negative_norm = torch.sum(negative ** 2, dim=1).unsqueeze(0) 
            
            dist_matrix_sq = torch.clamp(anchor_norm + negative_norm - 2.0 * dot_product, min=1e-16)
            # Apply square root to the entire matrix
            dist_matrix = torch.sqrt(dist_matrix_sq)
        else:
            anchor_normed = F.normalize(anchor, p=2, dim=1)
            negative_normed = F.normalize(negative, p=2, dim=1)
            cosine_sim_matrix = torch.mm(anchor_normed, negative_normed.t())
            dist_matrix = 1.0 - cosine_sim_matrix

        # 3. HARD NEGATIVE MINING
        hardest_dist_neg, _ = torch.min(dist_matrix, dim=1) # Shape: [B]

        # 4. Compute standard Hinge Loss
        losses = F.relu(dist_pos - hardest_dist_neg + self.margin)

        active_mask = losses > 0
        active_triplets = losses[active_mask]

        self.last_fraction_active = active_mask.float().mean().item()

        if active_triplets.numel() > 0:
            return active_triplets.mean()
        else:
            return losses.mean()