import torch
import torch.nn as nn
from .transformer import vit_conv_small


class ViTIC(nn.Module):
    """
    ViT-IC model implementation using MoCo3 ViT.
    """
    def __init__(self, feature_dim, num_classes, projector_dim, img_size):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.projector_dim = projector_dim
        self.vit = vit_conv_small(img_size=img_size)
        self.instance_projector = nn.Sequential(
            nn.Linear(self.vit.embed_dim, self.projector_dim),
            nn.BatchNorm1d(self.projector_dim),
            nn.ReLU(),
            nn.Linear(self.projector_dim, self.projector_dim),
            nn.BatchNorm1d(self.projector_dim),
            nn.ReLU(),
            nn.Linear(self.projector_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.vit.embed_dim, self.projector_dim),
            nn.BatchNorm1d(self.projector_dim),
            nn.ReLU(),
            nn.Linear(self.projector_dim, self.projector_dim),
            nn.BatchNorm1d(self.projector_dim),
            nn.ReLU(),
            nn.Linear(self.projector_dim, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        x_i = self.vit(x_i)
        x_j = self.vit(x_j)
        z_i = self.instance_projector(x_i)
        z_j = self.instance_projector(x_j)
        c_i = self.cluster_projector(x_i)
        c_j = self.cluster_projector(x_j)
        return z_i, z_j, c_i, c_j

    def evaluate(self, x):
        x = self.vit(x)
        z = self.instance_projector(x)
        c = self.cluster_projector(x)
        c = torch.argmax(c, dim=1)
        return x, z, c
