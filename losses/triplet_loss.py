import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet Loss with active triplet filtering (semi-hard negative mining).

    Computes the standard hinge-based triplet loss:
        L = max(d(a, p) - d(a, n) + margin, 0)

    where d(·,·) is either Euclidean (L2) or cosine distance.

    Active Triplet Filtering:
        Only triplets that violate the margin constraint (loss > 0) contribute
        to the gradient. Satisfied triplets — where the negative is already
        further from the anchor than the positive by at least `margin` — produce
        zero loss and are excluded from the mean calculation. This prevents
        the gradient from being diluted by easy triplets that the model has
        already learned to handle, accelerating convergence on hard cases.

        If ALL triplets in a batch are satisfied (loss = 0 for all), the mean
        of the zero tensor is returned to keep the computation graph connected
        and avoid a NaN gradient.

    Diagnostic:
        After each forward pass, `self.last_fraction_active` is updated with
        the fraction of triplets in the batch that were active (loss > 0).
        The training loop can log this to monitor mining effectiveness:
            - High fraction early, decreasing over epochs → mining is working
            - Persistently high fraction → model not converging
            - Fraction drops to near zero → consider harder mining strategy

    Args:
        margin (float): Margin for the triplet hinge loss. Default 1.0.
        mode (str): Distance metric. One of 'euclidean' or 'cosine'. Default 'euclidean'.

    Raises:
        ValueError: If `mode` is not 'euclidean' or 'cosine'.
    """

    SUPPORTED_MODES = ('euclidean', 'cosine')

    def __init__(self, margin=1.0, mode='euclidean'):
        super(TripletLoss, self).__init__()

        if mode.lower() not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported distance mode '{mode}'. "
                f"Must be one of: {self.SUPPORTED_MODES}"
            )

        self.margin = margin
        self.mode = mode.lower()

        # Diagnostic: fraction of active (non-zero loss) triplets in last batch.
        # Updated every forward pass. Accessible from the training loop for logging.
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
            # L2 distance (p=2 norm)
            dist_pos = F.pairwise_distance(anchor, positive, p=2)
            dist_neg = F.pairwise_distance(anchor, negative, p=2)
        else:
            # Cosine distance = 1 - cosine_similarity
            dist_pos = 1.0 - F.cosine_similarity(anchor, positive)
            dist_neg = 1.0 - F.cosine_similarity(anchor, negative)

        # --- Hinge loss ---
        losses = F.relu(dist_pos - dist_neg + self.margin)

        # --- Active triplet filtering ---
        # Filter triplets where the margin constraint is violated (loss > 0).
        # Threshold is 0 — F.relu() produces exact zeros for satisfied triplets,
        # so losses > 0 cleanly separates active from satisfied triplets.
        # Previously used 1e-16 (machine epsilon) which was too tight and
        # failed to filter near-zero losses from nearly-satisfied triplets.
        active_mask = losses > 0
        active_triplets = losses[active_mask]

        # Update diagnostic fraction
        self.last_fraction_active = active_mask.float().mean().item()

        if active_triplets.numel() > 0:
            # Mean over active (violated) triplets only
            return active_triplets.mean()
        else:
            # All triplets satisfied — return zero mean to keep graph connected
            # and avoid NaN gradients. Optimizer step will have no effect.
            return losses.mean()