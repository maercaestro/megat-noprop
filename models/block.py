import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoiseBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, input_type='image'):
        super().__init__()
        self.input_type = input_type
        
        # For image inputs
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # Assuming MNIST with 1 channel
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 128)  # Calculate proper dimensions
        )
        
        # For flattened inputs
        self.fc_block = nn.Sequential(
            nn.Linear(784, 256),  # For MNIST: 28×28=784
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self.fc_z1 = nn.Linear(embedding_dim, 256)
        self.bn_z1 = nn.BatchNorm1d(256)

        self.fc_z2 = nn.Linear(256, 256)
        self.bn_z2 = nn.BatchNorm1d(256)

        self.fc_z3 = nn.Linear(256, 256)
        self.bn_z3 = nn.BatchNorm1d(256)

        self.fc_f1 = nn.Linear(128 + 256, 256)  # Fixed dimensions
        self.bn_f1 = nn.BatchNorm1d(256)
        self.fc_f2 = nn.Linear(256, 128)
        self.bn_f2 = nn.BatchNorm1d(128)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor, z_prev: torch.Tensor, W_embed: torch.Tensor) -> tuple:
        # Detect input type automatically
        if len(x.shape) == 4:  # Image input: [B, C, H, W]
            x_feat = self.conv_block(x)
        else:  # Flattened input: [B, D]
            x_feat = self.fc_block(x)

        h1 = F.relu(self.bn_z1(self.fc_z1(z_prev)))
        h2 = F.relu(self.bn_z2(self.fc_z2(h1)))
        h3 = self.bn_z3(self.fc_z3(h2))

        z_feat = h3 + h1 # residual connection
        h_f = torch.cat([x_feat, z_feat], dim=1)
        h_f = F.relu(self.bn_f1(self.fc_f1(h_f)))
        h_f = F.relu(self.bn_f2(self.fc_f2(h_f)))
        logits = self.fc_out(h_f)
        p = F.softmax(logits, dim=1)
        z_next = p @ W_embed

        return z_next, logits