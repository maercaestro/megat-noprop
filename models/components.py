# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, List, Dict

# ----------------------------------------------------------------------------
# Sinusoidal embedding for scalar t ∈ [0,1]
# ----------------------------------------------------------------------------
def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal positional embeddings for time."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1)
    )
    args = t * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

# ----------------------------------------------------------------------------
# ResNet backbone selector (improved feature extraction)
# ----------------------------------------------------------------------------
class ResNetBackbone(nn.Module):
    """Powerful feature extractor using ResNet architectures."""
    
    def __init__(self, name: str, embed_dim: int, input_channels: int = 3):
        super().__init__()
        resnets = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet152': models.resnet152,
        }
        assert name in resnets, f"Unsupported backbone '{name}'"
        
        # Create ResNet and modify first conv for different input channels
        resnet = resnets[name](weights=None)
        
        # Modify first conv layer if needed
        if input_channels != 3:
            resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove final fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        feat_dim = resnet.fc.in_features
        self.proj = nn.Linear(feat_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        out = self.features(x).view(B, -1)
        return self.proj(out)

# ----------------------------------------------------------------------------
# Simple CNN backbone for smaller datasets
# ----------------------------------------------------------------------------
class SimpleCNNBackbone(nn.Module):
    """Lightweight CNN backbone for datasets like MNIST."""
    
    def __init__(self, embed_dim: int, input_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        self.proj = nn.Linear(128 * 4 * 4, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(x))

# ----------------------------------------------------------------------------
# Label embedding encoder with residual connection
# ----------------------------------------------------------------------------
class LabelEncoder(nn.Module):
    """
    Encodes a label-embedding vector z_t (shape [B, embed_dim]) via a small FC net with skip connection.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.hidden_dim = embed_dim
        self.fc1 = nn.Linear(embed_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, embed_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.fc2(self.relu(self.fc1(z)))
        return out + z  # Residual connection

# ----------------------------------------------------------------------------
# Time embedding encoder with sinusoidal features
# ----------------------------------------------------------------------------
class TimeEncoder(nn.Module):
    """
    Encodes a timestamp t (shape [B,1]) into embedding (shape [B, embed_dim]).
    """
    def __init__(self, time_emb_dim: int, embed_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(time_emb_dim, embed_dim),
            nn.ReLU()
        )
        self.time_emb_dim = time_emb_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B,1] -> sinusoidal [B, time_emb_dim]
        te = sinusoidal_embedding(t, self.time_emb_dim)
        return self.fc(te)

# ----------------------------------------------------------------------------
# Fusion head to combine image, label, and time features
# ----------------------------------------------------------------------------
class FuseHead(nn.Module):
    """Combines image, label embedding, and time features for final prediction."""
    
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.BatchNorm1d(embed_dim), 
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim), 
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes)
        )
    
    @property
    def out_features(self):
        return self.net[-1].out_features
        
    def forward(self, fx: torch.Tensor, fz: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
        x = torch.cat([fx, fz, ft], dim=1)
        return self.net(x)

# ----------------------------------------------------------------------------
# Learnable noise schedule module
# ----------------------------------------------------------------------------
class NoiseSchedule(nn.Module):
    """Learnable noise schedule using MLPs."""
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.gamma_tilde = nn.Sequential(
            nn.Linear(1, hidden_dim), 
            nn.Softplus(),
            nn.Linear(hidden_dim, 1), 
            nn.Softplus()
        )
        self.gamma0 = nn.Parameter(torch.tensor(-7.0))
        self.gamma1 = nn.Parameter(torch.tensor(7.0))

    def _gamma_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Normalized gamma function."""
        g0 = self.gamma_tilde(torch.zeros_like(t))
        g1 = self.gamma_tilde(torch.ones_like(t))
        return ((self.gamma_tilde(t) - g0) / (g1 - g0 + 1e-8)).clamp(0, 1)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Compute alpha_bar(t) from learnable schedule."""
        γt = self.gamma0 + (self.gamma1 - self.gamma0) * (1 - self._gamma_bar(t))
        return torch.sigmoid(-γt / 2).clamp(0.01, 0.99)
