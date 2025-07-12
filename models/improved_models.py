# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List, Dict
import random
import time

from .components import (
    ResNetBackbone, SimpleCNNBackbone, LabelEncoder, 
    TimeEncoder, FuseHead, NoiseSchedule
)

# ----------------------------------------------------------------------------
# Improved NoProp Continuous-Time Model
# ----------------------------------------------------------------------------
class NoPropCTImproved(nn.Module):
    """
    Improved continuous-time NoProp implementation with:
    - ResNet/CNN backbone options
    - Learnable noise schedule
    - Prototype initialization
    - Modular design
    """
    
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        time_emb_dim: int,
        embed_dim: int,
        input_channels: int = 3,
        use_resnet: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Feature backbone
        if use_resnet:
            self.backbone = ResNetBackbone(backbone, embed_dim, input_channels)
        else:
            self.backbone = SimpleCNNBackbone(embed_dim, input_channels)
            
        # Encoders
        self.label_enc = LabelEncoder(embed_dim)
        self.time_enc = TimeEncoder(time_emb_dim, embed_dim)
        
        # Fusion and output
        self.fuse = FuseHead(embed_dim, num_classes)
        
        # Learnable noise schedule
        self.noise_schedule = NoiseSchedule(hidden_dim=64)
        
        # Class embeddings (will be initialized with prototypes)
        self.W_embed = nn.Parameter(torch.zeros(num_classes, embed_dim))
        
        # Initialize with small random values initially
        nn.init.normal_(self.W_embed, std=0.1)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Get alpha_bar from learnable noise schedule."""
        return self.noise_schedule.alpha_bar(t)

    def forward_u(self, x: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the denoising network."""
        fx = self.backbone(x)
        fz = self.label_enc(z_t)
        ft = self.time_enc(t)
        return self.fuse(fx, fz, ft)

# ----------------------------------------------------------------------------
# Prototype initialization (KEY IMPROVEMENT)
# ----------------------------------------------------------------------------
def initialize_with_prototypes(
    model: NoPropCTImproved,
    dataset: Dataset,
    num_classes: int,
    device: torch.device,
    samples_per_class: int = 10,
    batch_size: int = 512,
    num_workers: int = 0,  # Use 0 to avoid multiprocessing issues
) -> Tuple[torch.Tensor, List[int]]:
    """
    Initialize W_embed with class prototypes from actual data.
    This is a MAJOR improvement over random initialization.
    """
    print("Initializing prototypes from training data...")
    model.eval()

    # 1) Embed entire dataset in batches
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    feats_list, labels_list = [], []
    
    with torch.no_grad():
        for imgs, labels in loader:
            feats = model.backbone(imgs.to(device))
            feats_list.append(feats.cpu())
            labels_list.append(labels)
            
    all_feats = torch.cat(feats_list, dim=0)   # [N, D] on CPU
    all_labels = torch.cat(labels_list, dim=0)  # [N]

    D = all_feats.size(1)
    W_proto = torch.zeros(num_classes, D, device=device)
    proto_idxs = []

    # 2) For each class, randomly pick up to samples_per_class embeddings, then find medoid
    for c in range(num_classes):
        # Get indices of class-c samples
        idxs_c = (all_labels == c).nonzero(as_tuple=True)[0].tolist()
        
        if len(idxs_c) == 0:
            print(f"Warning: No samples found for class {c}")
            continue
            
        # Randomly choose up to samples_per_class
        chosen = random.sample(idxs_c, min(samples_per_class, len(idxs_c)))
        embs = all_feats[chosen].to(device)     # [≤samples_per_class, D]
        idxs = torch.tensor(chosen)

        # Compute pairwise distances and find medoid
        if len(embs) > 1:
            dmat = torch.cdist(embs, embs)          # [k, k]
            dmed = dmat.median(dim=1).values        # [k]
            best = torch.argmin(dmed).item()
        else:
            best = 0

        W_proto[c] = embs[best]
        proto_idxs.append(idxs[best].item())

    print(f"Initialized {num_classes} prototypes")
    return W_proto, proto_idxs

# ----------------------------------------------------------------------------
# Training step with improved loss
# ----------------------------------------------------------------------------
def train_step(
    model: NoPropCTImproved, 
    x: torch.Tensor, 
    y: torch.Tensor, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device, 
    η: float = 1.0
) -> Dict[str, float]:
    """
    Single training step with improved loss computation.
    """
    model.train()
    B = x.size(0)
    
    # Get true class embeddings
    u_y = model.W_embed[y]  # (B, embed_dim)
    
    # Sample random time and compute noise schedule
    t = torch.rand(B, 1, device=device, requires_grad=True)
    αb = model.alpha_bar(t)
    snr = αb / (1 - αb + 1e-8)
    
    # Compute SNR derivative for weighting
    snr_p = torch.autograd.grad(snr.sum(), t, create_graph=True)[0]
    
    # Add noise to true embeddings
    eps = torch.randn_like(u_y)
    zt = αb * u_y + (1 - αb).sqrt() * eps
    
    # Get model predictions
    logits = model.forward_u(x, zt, t)
    p = F.softmax(logits, dim=1)
    pred_e = p @ model.W_embed
    
    # Score matching loss (weighted by SNR derivative)
    mse = F.mse_loss(pred_e, u_y, reduction='none').sum(dim=1, keepdim=True)
    loss_sdm = 0.5 * η * (snr_p * mse).mean()
    
    # KL divergence (embedding regularization)
    loss_kl = 0.5 * (u_y.pow(2).sum(dim=1)).mean()
    
    # Cross-entropy at t=1 (final classification)
    t1 = torch.ones_like(t)
    αb1 = model.alpha_bar(t1)
    z1 = αb1 * u_y + (1 - αb1).sqrt() * torch.randn_like(u_y)
    loss_ce = F.cross_entropy(model.forward_u(x, z1, t1), y)
    
    # Total loss
    loss = loss_ce + loss_kl + loss_sdm
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return {
        'total_loss': loss.item(),
        'ce_loss': loss_ce.item(),
        'kl_loss': loss_kl.item(),
        'sdm_loss': loss_sdm.item()
    }

# ----------------------------------------------------------------------------
# Inference with ODE solvers
# ----------------------------------------------------------------------------
@torch.no_grad()
def run_euler_inference(
    model: NoPropCTImproved, 
    x: torch.Tensor, 
    T_steps: int = 1000
) -> torch.Tensor:
    """Euler method for inference."""
    model.eval()
    B = x.size(0)
    embed_dim = model.W_embed.size(1)
    dt = 1.0 / T_steps
    
    # Start from noise
    z = torch.randn(B, embed_dim, device=x.device)
    
    for i in range(T_steps):
        t = torch.full((B, 1), i / T_steps, device=x.device)
        αb = model.alpha_bar(t)
        
        # Get model prediction
        logits = model.forward_u(x, z, t)
        p = F.softmax(logits, dim=1)
        pred_e = p @ model.W_embed
        
        # Euler update
        z = z + dt * (pred_e - z) / (1 - αb + 1e-8)
    
    # Final classification
    final_logits = model.forward_u(x, z, torch.ones_like(t))
    return final_logits.argmax(dim=1)

@torch.no_grad()
def run_heun_inference(
    model: NoPropCTImproved, 
    x: torch.Tensor, 
    T_steps: int = 40
) -> torch.Tensor:
    """Heun's method (2nd order) for inference - more accurate."""
    model.eval()
    B = x.size(0)
    embed_dim = model.W_embed.size(1)
    dt = 1.0 / T_steps
    
    # Start from noise
    z = torch.randn(B, embed_dim, device=x.device)
    
    for i in range(T_steps):
        t_n = torch.full((B, 1), i / T_steps, device=x.device)
        t_np1 = torch.full((B, 1), (i + 1) / T_steps, device=x.device)
        
        # Current state
        αn = model.alpha_bar(t_n)
        p_n = F.softmax(model.forward_u(x, z, t_n), dim=1)
        pred_n = p_n @ model.W_embed
        f_n = (pred_n - z) / (1 - αn + 1e-8)
        
        # Predictor step
        z_mid = z + dt * f_n
        
        # Corrector step
        αm = model.alpha_bar(t_np1)
        p_mid = F.softmax(model.forward_u(x, z_mid, t_np1), dim=1)
        pred_mid = p_mid @ model.W_embed
        f_mid = (pred_mid - z_mid) / (1 - αm + 1e-8)
        
        # Heun update (average of slopes)
        z = z + 0.5 * dt * (f_n + f_mid)
    
    # Final classification
    final_logits = model.forward_u(x, z, torch.ones_like(t_n))
    return final_logits.argmax(dim=1)

# ----------------------------------------------------------------------------
# Evaluation function
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(
    model: NoPropCTImproved,
    test_loader: DataLoader,
    device: torch.device,
    T_steps: int = 40,
    use_heun: bool = True
) -> float:
    """Evaluate model accuracy on test set."""
    model.eval()
    correct = 0
    total = 0
    
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        
        if use_heun:
            preds = run_heun_inference(model, x, T_steps)
        else:
            preds = run_euler_inference(model, x, T_steps)
            
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    return correct / total
