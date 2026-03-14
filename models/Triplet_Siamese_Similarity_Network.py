import torch
import torch.nn as nn
from models.feature_extractor import DenseNetFeatureExtractor

class tDCBAM(nn.Module):
    """
    A basic Triplet Siamese Similarity Network structure.

    This class wraps the feature extractor and provides a convenient forward
    method that takes anchor, positive, and negative inputs simultaneously,
    primarily used during the pre-training phase with standard Triplet Loss.

    In the meta-learning phase, the feature_extractor is used directly,
    and the metric generation is handled separately by the MetricGenerator.
    """
    def __init__(self, backbone_name='densenet121', output_dim=1024, pretrained=True, baseline=False):
        super(tDCBAM, self).__init__()
        # Instantiate the shared feature extractor
        self.feature_extractor = DenseNetFeatureExtractor(
            backbone_name=backbone_name,
            output_dim=output_dim,
            pretrained=pretrained,
            baseline=baseline
        )

    def forward(self, anchor, positive, negative):
        """
        Processes a triplet of images through the shared feature extractor.
        """
        # Apply the *same* feature extractor to all three inputs (Siamese architecture)
        anchor_feat = self.feature_extractor(anchor)
        positive_feat = self.feature_extractor(positive)
        negative_feat = self.feature_extractor(negative)

        return anchor_feat, positive_feat, negative_feat

    def get_backbone_params(self):
        """Passthrough method to get backbone parameters for the optimizer."""
        return self.feature_extractor.get_backbone_params()

    def get_head_params(self):
        """Passthrough method to get head parameters (CBAM + FC) for the optimizer."""
        return self.feature_extractor.get_head_params()