import torch
import torch.nn as nn
import torch.nn.functional as F

class MetricGenerator(nn.Module):
    """
    Implements a Deep Relation Network for Learnable Metric Similarity in One-Shot Learning.

    Research Context:
    Instead of concatenating features (which is asymmetrical and forces the network 
    to learn how to subtract), this updated module computes the element-wise 
    Absolute Difference between the Support and Query embeddings. This explicitly 
    highlights structural stroke discrepancies, allowing the MLP to focus immediately 
    on scoring the severity of those differences.
    """

    # CHANGED: embedding_dim defaults to 1024 (size of absolute difference)
    def __init__(self, embedding_dim=1024, hidden_dim=256, dropout=0.5):
        """
        Initializes the Relation Network architecture.

        Args:
            embedding_dim (int): The size of the compared feature vector. 
                                 For DenseNet121, absolute difference yields 1024 dim.
            hidden_dim (int): The size of the hidden interaction layer. Default: 256.
            dropout (float): The dropout probability for regularization during training.
        """
        super(MetricGenerator, self).__init__()
        
        # Deep Relation Module (MLP)
        self.relation_module = nn.Sequential(

            # 1. Feature Interaction Layer
            nn.Linear(embedding_dim, hidden_dim),
            
            # 2. Normalization Strategy
            nn.LayerNorm(hidden_dim),
            
            # 3. Non-Linear Activation
            nn.ReLU(),
            
            # 4. Regularization
            nn.Dropout(dropout),
            
            # 5. Scalar Scoring Layer
            nn.Linear(hidden_dim, 1)
        )

    # CHANGED: Now takes support and query features separately
    def forward(self, support_features, query_features):
        """
        Performs the forward pass to compute the similarity score (relation) between feature pairs.

        Args:
            support_features (Tensor): Feature vectors of the support images. [Batch_Size, 1024]
            query_features (Tensor): Feature vectors of the query images. [Batch_Size, 1024]

        Returns:
            similarity_logits (Tensor): The raw similarity scores (logits). Shape: [Batch_Size, 1]
        """
        # Compute the element-wise absolute difference
        diff_features = torch.abs(support_features - query_features)
        
        # Compute the non-linear relation score based on the differences
        similarity_logits = self.relation_module(diff_features)
        
        return similarity_logits