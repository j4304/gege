import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet Loss with active triplet filtering (semi-hard negative mining).

    Computes the standard hinge-based triplet loss:
        L = max(d(a, p) - d(a, n) + margin, 0)

    where d(·,·) is either Squared Euclidean Distance or Cosine distance.
    Squared Euclidean is preferred here as it aligns smoothly with L2-normalized 
    embeddings and avoids square-root derivative instabilities near zero.

    Active Triplet Filtering:
        Only triplets that violate the margin constraint (loss > 0) contribute
        to the gradient. Satisfied triplets — where the negative is already
        further from the anchor than the positive by at least `margin` — produce
        zero loss and are excluded from the mean calculation. 

        If ALL triplets in a batch are satisfied (loss = 0 for all), the mean
        of the zero tensor is returned to keep the computation graph connected.

    Args:
        margin (float): Margin for the triplet hinge loss. Default 0.2
                        (optimized for L2-normalized embeddings).
        mode (str): Distance metric. One of 'euclidean' or 'cosine'. Default 'euclidean'.
    """

    SUPPORTED_MODES = ('euclidean', 'cosine')

    def __init__(self, margin=0.2, mode='euclidean'):
        super(TripletLoss, self).__init__()

        if mode.lower() not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported distance mode '{mode}'. "
                f"Must be one of: {self.SUPPORTED_MODES}"
            )

        self.margin = margin
        self.mode = mode.lower()

        # Diagnostic: fraction of active (non-zero loss) triplets in last batch.
        self.last_fraction_active = 0.0

    def forward(self, anchor, positive, negative):
        """
        Computes the triplet loss over a batch of embedding triplets.

        Args:
            anchor   (Tensor): Anchor embeddings.   Shape: [B, D]
            positive (Tensor): Positive embeddings. Shape: [B, D]
            negative (Tensor): Negative embeddings. Shape: [B, D]

        Returns:
            Tensor: Scalar loss value (mean over active triplets).
        """
        # --- Distance computation ---
        if self.mode == 'euclidean':
            # Squared Euclidean Distance: ||a - b||^2
            dist_pos = torch.sum(torch.pow(anchor - positive, 2), dim=1)
            dist_neg = torch.sum(torch.pow(anchor - negative, 2), dim=1)
        else:
            # Cosine distance = 1 - cosine_similarity
            dist_pos = 1.0 - F.cosine_similarity(anchor, positive)
            dist_neg = 1.0 - F.cosine_similarity(anchor, negative)

        # --- Hinge loss ---
        losses = F.relu(dist_pos - dist_neg + self.margin)

        # --- Active triplet filtering ---
        active_mask = losses > 0
        active_triplets = losses[active_mask]

        # Update diagnostic fraction
        self.last_fraction_active = active_mask.float().mean().item()

        if active_triplets.numel() > 0:
            # Mean over active (violated) triplets only
            return active_triplets.mean()
        else:
            # All triplets satisfied — return zero mean to keep graph connected
            return losses.mean()