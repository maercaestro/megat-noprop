import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import DenoiseBlock

class NoPropModel(nn.Module):
    """
    Discrete-time NoProp implementation matching Qinyu Li et al. (arXiv:2503.24322v1).
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        B = x.size(0)
        eps = 1e-6

        # True class embeddings
        u_y = F.embedding(y, self.W_embed)

        # Sample z_T ~ q(z_T | y)
        noise_T = torch.randn_like(u_y)
        z_T = torch.sqrt(self.alpha_bar[-1]) * u_y + torch.sqrt(1 - self.alpha_bar[-1] + eps) * noise_T

        # Cross-entropy on final logits
        logits = self.classifier(z_T)
        L_ce = F.cross_entropy(logits, y)

        # KL divergence between q(z0|y)=N(u_y,I) and p(z0)=N(0,I)
        L_kl = 0.5 * torch.mean(torch.sum(u_y.pow(2), dim=1))

        # Random timestep for block-wise loss
        t = torch.randint(1, self.T + 1, (1,)).item()

        # Sample z_t ~ q(z_t | y)
        noise_t = torch.randn_like(u_y)
        alpha_t = self.alpha_bar[t-1]
        z_t = torch.sqrt(alpha_t) * u_y + torch.sqrt(1 - alpha_t + eps) * noise_t

        # Reverse sample z_{t-1} ~ q(z_{t-1} | z_t, y)
        alpha_prev = self.alpha_bar[t-2] if t > 1 else torch.tensor(1.0, device=x.device)
        mu = (torch.sqrt(alpha_prev) * (z_t - torch.sqrt(1 - alpha_t + eps) * noise_t)
              / torch.sqrt(alpha_t + eps))
        noise_tm1 = torch.randn_like(u_y)
        z_tm1 = mu + torch.sqrt(1 - alpha_prev + eps) * noise_tm1

        # Block prediction and L2
        z_pred, _ = self.blocks[t-1](x, z_tm1, self.W_embed)
        L_l2 = F.mse_loss(z_pred, z_t) * self.snr_delta[t-1]

        # Total loss
        L_total = L_ce + L_kl + self.eta * L_l2

        return {
            'loss': L_total,
            'cross_entropy': L_ce,
            'kl_divergence': L_kl,
            'l2_loss': L_l2,
            'logits': logits
        }

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        z = torch.randn(B, self.embedding_dim, device=x.device)
        for t in range(self.T):
            z, _ = self.blocks[t](x, z, self.W_embed)
        return torch.argmax(self.classifier(z), dim=1)


class NoPropModelCT(nn.Module):
    """
    Continuous-time NoProp implementation with learnable noise schedule.
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
            nn.Linear(256, num_classes), nn.Softmax(dim=1)
        )

        # Learnable gamma schedule
        self.gamma_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.Softplus(), nn.Linear(64, 1), nn.Softplus()
        )
        self.gamma_0 = nn.Parameter(torch.tensor(1.0))
        self.gamma_1 = nn.Parameter(torch.tensor(5.0))

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        gamma_hat = self.gamma_mlp(t)
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * (1 - gamma_hat / gamma_hat.max())
        return torch.sigmoid(-gamma)

    def snr_prime(self, t: torch.Tensor) -> torch.Tensor:
        t_ = t.clone().detach().requires_grad_(True)
        alpha = self.alpha_bar(t_)
        grad = torch.autograd.grad(alpha.sum(), t_, create_graph=True)[0]
        return grad / (1 - alpha)**2

    def forward_denoise(self, x: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_feat = self.image_embed(x)
        z_feat = self.latent_embed(z_t)
        t_feat = self.time_embed(t)
        fused = torch.cat([x_feat, z_feat, t_feat], dim=1)
        weights = self.u_theta(fused)
        return weights @ self.W_embed

    def inference(self, x: torch.Tensor, steps: int = 1000) -> torch.Tensor:
        B = x.size(0)
        z = torch.randn(B, self.embedding_dim, device=x.device)
        for i in range(steps):
            t = torch.full((B,1), i/steps, device=x.device)
            z = self.forward_denoise(x, z, t)
        return torch.argmax(self.classifier(z), dim=1)


class NoPropModelFM(nn.Module):
    """
    Flow-matching NoProp implementation.
    """
    def __init__(self, num_classes: int, embedding_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Embeddings & classifier
        self.W_embed = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.1)
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # Time, image, latent encoders
        self.time_embed = nn.Sequential(nn.Linear(1,64), nn.ReLU(), nn.Linear(64,64))
        self.image_embed = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(), nn.Linear(64,128), nn.ReLU()
        )
        self.latent_embed = nn.Sequential(nn.Linear(embedding_dim,128), nn.ReLU())

        # Vector field network v_theta
        self.vector_field = nn.Sequential(
            nn.Linear(128 + 128 + 64, 256), nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward_vector_field(self, x: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_f = self.image_embed(x)
        z_f = self.latent_embed(z_t)
        t_f = self.time_embed(t)
        fused = torch.cat([x_f, z_f, t_f], dim=1)
        return self.vector_field(fused)

    def extrapolate_z(self, z_t: torch.Tensor, v_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return z_t + (1 - t) * v_t

    def inference(self, x: torch.Tensor, steps: int = 1000) -> torch.Tensor:
        B = x.size(0)
        z = torch.randn(B, self.embedding_dim, device=x.device)
        for i in range(steps):
            t = torch.full((B,1), i/steps, device=x.device)
            v = self.forward_vector_field(x, z, t)
            z = self.extrapolate_z(z, v, t)
        return torch.argmax(self.classifier(z), dim=1)
