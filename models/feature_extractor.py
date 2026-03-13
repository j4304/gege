import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# =============================================================================
# CBAM ATTENTION MODULES
# =============================================================================

class ChannelAttention(nn.Module):
    """
    CBAM Channel Attention Module.

    Computes channel-wise attention weights by aggregating spatial information
    via both average pooling and max pooling, then passing through a shared MLP.
    The two outputs are summed and passed through a sigmoid to produce
    per-channel attention weights in [0, 1].

    Reference:
        Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.

    Args:
        channels (int): Number of input channels.
        ratio (int): Channel reduction ratio for the MLP bottleneck. Default 8.
    """
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // ratio, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    CBAM Spatial Attention Module.

    Computes spatial attention weights by aggregating channel information
    via average pooling and max pooling along the channel dimension,
    concatenating the results, and passing through a convolutional layer.

    Reference:
        Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.

    Args:
        kernel_size (int): Convolution kernel size. Default 7 (as in paper).
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_concat))


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).

    Sequentially applies channel attention followed by spatial attention.
    Each attention map is applied multiplicatively as a feature recalibration.

    Args:
        channels (int): Number of input channels.
        ratio (int): Channel reduction ratio for ChannelAttention MLP. Default 8.
        kernel_size (int): Kernel size for SpatialAttention conv. Default 7.
    """
    def __init__(self, channels, ratio=8, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# =============================================================================
# DENSENET-121 FEATURE EXTRACTOR
# =============================================================================

# DenseNet-121 channel counts at each CBAM insertion point.
#
# CBAM is placed AFTER each Dense Block, BEFORE each Transition layer.
# This is the architecturally faithful placement per the CBAM paper (Woo et al.,
# ECCV 2018), which inserts attention after convolutional feature extraction —
# i.e. after the Dense Block has enriched features via concatenated skip
# connections, but before the Transition's 1×1 conv + avg-pool compresses them.
# Recalibrating before compression allows CBAM to suppress irrelevant channels
# and spatial regions at full resolution, before information is discarded by the
# transition's bottleneck.
#
# For Stage 4 (no transition): CBAM is placed after block4, before norm5.
#
# Channel counts AFTER each Dense Block (before transition compression):
#   After Dense Block 1: 256   (64 stem + 6 layers × 32 growth)
#   After Dense Block 2: 512   (128 trans1 out + 12 layers × 32 growth)
#   After Dense Block 3: 1024  (256 trans2 out + 24 layers × 32 growth)
#   After Dense Block 4: 1024  (512 trans3 out + 16 layers × 32 growth)
#
# These are fixed architectural constants — do NOT infer at runtime.
_DENSENET121_CBAM_CHANNELS = {
    'cbam1': 256,   # after block1, before trans1
    'cbam2': 512,   # after block2, before trans2
    'cbam3': 1024,  # after block3, before trans3
    'cbam4': 1024,  # after block4, before norm5
}


class DenseNetFeatureExtractor(nn.Module):
    """
    DenseNet-121 feature extractor with optional CBAM attention integration.

    Two modes are supported via the `baseline` flag:

    Baseline mode (baseline=True):
        Standard DenseNet-121 backbone with a Regularized Dense Block head.
        No CBAM modules. Produces a feature vector of size `output_dim`.
        Backbone freeze/unfreeze is handled EXTERNALLY by the training loop —
        NOT in __init__.

    Proposed mode (baseline=False):
        DenseNet-121 backbone with 4 CBAM blocks placed AFTER each Dense Block
        and BEFORE each Transition layer. This is the architecturally faithful
        placement per the CBAM paper — attention recalibrates the fully-enriched
        dense features at full channel resolution before the transition compresses
        them. For Stage 4 (no transition), CBAM is placed before norm5.

    Forward pass (proposed):
        Input → Stem
               → Block1 → CBAM1 → Trans1
               → Block2 → CBAM2 → Trans2
               → Block3 → CBAM3 → Trans3
               → Block4 → CBAM4 → Norm5 → ReLU
               → GlobalAvgPool → Flatten → RegularizedDenseBlock → Output

    Args:
        backbone_name (str): Backbone identifier. Only 'densenet121' supported.
        output_dim (int): Output embedding dimension. Default 1024.
        pretrained (bool): Load ImageNet pretrained weights. Default True.
        baseline (bool): Use baseline mode (no CBAM). Default False.
    """

    def __init__(self, backbone_name='densenet121', output_dim=1024,
                 pretrained=True, baseline=False):
        super().__init__()
        self.baseline = baseline

        if backbone_name != 'densenet121':
            raise ValueError(
                f"Unsupported backbone_name '{backbone_name}'. "
                "Only 'densenet121' is currently supported."
            )

        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        original_model = models.densenet121(weights=weights)
        features = original_model.features

        if self.baseline:
            # ── Baseline Mode ──────────────────────────────────────────────
            # Full DenseNet-121 backbone, no CBAM.
            # Backbone freeze/unfreeze handled by training loop, not here.
            self.backbone = features
            self.post_norm_relu = nn.ReLU(inplace=False)
            self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
            self.regularized_dense_block = nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.Dropout(p=0.5),
                nn.Linear(1024, output_dim)
            )

        else:
            # ── Proposed Mode ──────────────────────────────────────────────
            # CBAM after each Dense Block, before each Transition.
            # This is the architecturally faithful placement: attention acts
            # on fully-enriched dense features before transition compression.

            # Stem: Conv7×7 + BN + ReLU + MaxPool
            self.initial_layers = nn.Sequential(*list(features.children())[:4])

            # Stage 1: Block1 → CBAM1(256ch) → Trans1
            self.block1 = features.denseblock1
            self.cbam1  = CBAMBlock(channels=_DENSENET121_CBAM_CHANNELS['cbam1'])  # 256
            self.trans1 = features.transition1

            # Stage 2: Block2 → CBAM2(512ch) → Trans2
            self.block2 = features.denseblock2
            self.cbam2  = CBAMBlock(channels=_DENSENET121_CBAM_CHANNELS['cbam2'])  # 512
            self.trans2 = features.transition2

            # Stage 3: Block3 → CBAM3(1024ch) → Trans3
            self.block3 = features.denseblock3
            self.cbam3  = CBAMBlock(channels=_DENSENET121_CBAM_CHANNELS['cbam3'])  # 1024
            self.trans3 = features.transition3

            # Stage 4: Block4 → CBAM4(1024ch) → Norm5
            # No transition after block4; CBAM placed before final BN.
            self.block4 = features.denseblock4
            self.cbam4  = CBAMBlock(channels=_DENSENET121_CBAM_CHANNELS['cbam4'])  # 1024

            self.norm5   = features.norm5
            self.post_norm_relu = nn.ReLU(inplace=False)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

            # Regularized Dense Block: BN → Dropout → Linear projection
            self.regularized_dense_block = nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.Dropout(p=0.5),
                nn.Linear(1024, output_dim)
            )

    def forward(self, x):
        """
        Forward pass through the feature extractor.

        Args:
            x (torch.Tensor): Input tensor of shape [B, 3, H, W].

        Returns:
            torch.Tensor: Feature embedding of shape [B, output_dim].
        """
        if self.baseline:
            x = self.backbone(x)
            x = self.post_norm_relu(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.regularized_dense_block(x)
            return x
        else:
            # Stem
            x = self.initial_layers(x)

            # Block1 → CBAM1 → Trans1
            x = self.block1(x)
            x = self.cbam1(x)         # attention at 256ch, before trans1 compresses
            x = self.trans1(x)

            # Block2 → CBAM2 → Trans2
            x = self.block2(x)
            x = self.cbam2(x)         # attention at 512ch, before trans2 compresses
            x = self.trans2(x)

            # Block3 → CBAM3 → Trans3
            x = self.block3(x)
            x = self.cbam3(x)         # attention at 1024ch, before trans3 compresses
            x = self.trans3(x)

            # Block4 → CBAM4 → Norm5 → ReLU
            x = self.block4(x)
            x = self.cbam4(x)         # attention at 1024ch, before final BN
            x = self.norm5(x)
            x = self.post_norm_relu(x)

            # GlobalAvgPool → Flatten → RegularizedDenseBlock
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.regularized_dense_block(x)
            x = F.normalize(x, p=2, dim=1)
            return x

    def get_backbone_params(self):
        """
        Returns backbone parameter groups for use in optimizer construction.

        Baseline:  all parameters in self.backbone
        Proposed:  initial_layers + all dense blocks + transitions + norm5
                   (excludes CBAM modules — those are head params)
        """
        if self.baseline:
            return list(self.backbone.parameters())
        else:
            backbone_modules = [
                self.initial_layers,
                self.block1, self.trans1,
                self.block2, self.trans2,
                self.block3, self.trans3,
                self.block4, self.norm5,
            ]
            params = []
            for module in backbone_modules:
                params.extend(module.parameters())
            return params

    def get_head_params(self):
        """
        Returns non-backbone (head) parameter groups for optimizer construction.

        Baseline:  regularized_dense_block
        Proposed:  cbam1-4 + regularized_dense_block
        """
        if self.baseline:
            return list(self.regularized_dense_block.parameters())
        else:
            head_modules = [
                self.cbam1, self.cbam2, self.cbam3, self.cbam4,
                self.regularized_dense_block,
            ]
            params = []
            for module in head_modules:
                params.extend(module.parameters())
            return params