import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import DenoiseBlock

class NoPropDT(nn.Module):
    """
    Discrete-time NoProp implementation.
    Uses fixed timesteps and cosine noise schedule.
    """
    def __init__(self, num_classes: int, embedding_dim: int, T: int = 10, eta: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.T = T
        self.eta = eta

        # Stacked denoise blocks
        self.blocks = nn.ModuleList([
            DenoiseBlock(embedding_dim, num_classes) for _ in range(T)
        ])

        # Learnable class embeddings
        self.W_embed = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.1)

        # Final classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # Precompute cosine noise schedule
        t = torch.arange(1, T+1, dtype=torch.float32)
        alpha_t = torch.cos(t / T * (torch.pi / 2)).pow(2)
        alpha_bar = torch.cumprod(alpha_t, dim=0)

        # SNR deltas for ELBO weighting
        snr = alpha_bar / (1 - alpha_bar)
        snr_prev = torch.cat([torch.tensor([0.], dtype=snr.dtype), snr[:-1]], dim=0)
        snr_delta = snr - snr_prev

        # Register buffers so they move with .to(device)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('snr_delta', snr_delta)

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """Simple inference through all blocks sequentially."""
        B = x.size(0)
        z = torch.randn(B, self.embedding_dim, device=x.device)
        
        for t in range(self.T):
            z, _ = self.blocks[t](x, z, self.W_embed)
        
        return self.classifier(z)


class NoPropCT(nn.Module):
    """
    Continuous-time NoProp implementation.
    Uses learnable noise schedule and continuous time sampling.
    """
    def __init__(self, num_classes: int, embedding_dim: int, eta: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.eta = eta

        # Embeddings & classifier
        self.W_embed = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.1)
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64)
        )

        # Image & latent encoders  
        self.image_embed = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(), nn.Linear(64, 128), nn.ReLU()
        )
        self.latent_embed = nn.Sequential(
            nn.Linear(embedding_dim, 128), nn.ReLU()
        )

        # Denoising network u_theta
        self.u_theta = nn.Sequential(
            nn.Linear(128 + 128 + 64, 256), nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

        # Learnable gamma schedule
        self.gamma_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.Softplus(), nn.Linear(64, 1), nn.Softplus()
        )
        self.gamma_0 = nn.Parameter(torch.tensor(1.0))
        self.gamma_1 = nn.Parameter(torch.tensor(5.0))

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Learnable noise schedule."""
        gamma_hat = self.gamma_mlp(t)
        gamma_max = self.gamma_mlp(torch.ones_like(t))
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * (1 - gamma_hat / (gamma_max + 1e-8))
        return torch.sigmoid(-gamma).clamp(0.01, 0.99)

    def snr_prime(self, t: torch.Tensor) -> torch.Tensor:
        """SNR derivative for weighting."""
        t_ = t.clone().detach().requires_grad_(True)
        alpha = self.alpha_bar(t_)
        snr = alpha / (1 - alpha + 1e-8)
        grad = torch.autograd.grad(snr.sum(), t_, create_graph=True)[0]
        return grad

    def forward_denoise(self, x: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through denoising network."""
        x_feat = self.image_embed(x)
        z_feat = self.latent_embed(z_t)
        t_feat = self.time_embed(t)
        fused = torch.cat([x_feat, z_feat, t_feat], dim=1)
        return self.u_theta(fused)

    def inference(self, x: torch.Tensor, steps: int = 1000) -> torch.Tensor:
        """Inference with Euler integration."""
        B = x.size(0)
        z = torch.randn(B, self.embedding_dim, device=x.device)
        dt = 1.0 / steps
        
        for i in range(steps):
            t = torch.full((B, 1), i / steps, device=x.device)
            alpha_bar_t = self.alpha_bar(t)
            z_pred = self.forward_denoise(x, z, t)
            # Simple Euler step
            z = z + dt * (z_pred - z) / (1 - alpha_bar_t + 1e-8)
        
        return self.classifier(z)


class NoPropFM(nn.Module):
    """
    Flow Matching NoProp implementation.
    Uses vector fields and linear interpolation paths.
    """
    def __init__(self, num_classes: int, embedding_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Embeddings & classifier
        self.W_embed = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.1)
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # Time, image, latent encoders
        self.time_embed = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64))
        self.image_embed = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(), nn.Linear(64, 128), nn.ReLU()
        )
        self.latent_embed = nn.Sequential(nn.Linear(embedding_dim, 128), nn.ReLU())

        # Vector field network v_theta
        self.vector_field = nn.Sequential(
            nn.Linear(128 + 128 + 64, 256), nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward_vector_field(self, x: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict vector field at (x, z_t, t)."""
        x_f = self.image_embed(x)
        z_f = self.latent_embed(z_t)
        t_f = self.time_embed(t)
        fused = torch.cat([x_f, z_f, t_f], dim=1)
        return self.vector_field(fused)

    def extrapolate_z1(self, z_t: torch.Tensor, v_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Extrapolate to z1 using predicted vector field."""
        # For flow matching: z1 = z_t + (1-t) * v_pred
        return z_t + (1 - t) * v_pred

    def inference(self, x: torch.Tensor, steps: int = 1000) -> torch.Tensor:
        """Flow matching inference with Euler integration."""
        B = x.size(0)
        z = torch.randn(B, self.embedding_dim, device=x.device)
        dt = 1.0 / steps
        
        for i in range(steps):
            t = torch.full((B, 1), i / steps, device=x.device)
            v = self.forward_vector_field(x, z, t)
            z = z + dt * v
        
        return self.classifier(z)
