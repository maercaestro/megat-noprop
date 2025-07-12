import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import torch.nn.functional as F

# -----------------------------
# 1) Generate Lorenz data
# -----------------------------
def lorenz_fn(t, state):
    sigma, rho, beta = 10.0, 28.0, 8/3
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return torch.stack([dx, dy, dz])

# Time series parameters
t_points = 2000
t_max = 20.0
time = torch.linspace(0, t_max, t_points)
init_state = torch.tensor([1.0, 1.0, 1.0])

# Integrate Lorenz to obtain trajectory
true_states = odeint(lorenz_fn, init_state, time)

# Prepare next-step pairs
X_data = true_states[:-1]  # (t_points-1, 3)
Y_data = true_states[1:]   # (t_points-1, 3)

# Train-test split
split_idx = 1600
X_train, X_test = X_data[:split_idx],  X_data[split_idx:]
Y_train, Y_test = Y_data[:split_idx],  Y_data[split_idx:]

# -----------------------------
# 2) Define CT-LNN model
# -----------------------------
class CT_LNN(nn.Module):
    def __init__(self, dim=3, hidden=64):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64)
        )
        # Latent embedding
        self.latent_embed = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU()
        )
        # Vector field network
        self.vector_field = nn.Sequential(
            nn.Linear(hidden + 64, 128), nn.ReLU(),
            nn.Linear(128, dim)
        )
        # Learnable gamma schedule parameters
        self.gamma_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.Softplus(), nn.Linear(64, 1), nn.Softplus()
        )
        self.gamma_0 = nn.Parameter(torch.tensor(1.0))
        self.gamma_1 = nn.Parameter(torch.tensor(5.0))

        # Add dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def alpha_bar(self, t):
        # Replace the complex gamma approach with a simple cosine schedule
        return torch.cos((t * torch.pi) / 2) ** 2  # 1.0 at t=0, 0.0 at t=1

    def snr_prime(self, t):
        t_ = t.clone().detach().requires_grad_(True)
        alpha = self.alpha_bar(t_)
        grad = torch.autograd.grad(alpha.sum(), t_, create_graph=True)[0]
        return grad / (1 - alpha) ** 2

    def drift(self, z_t, t):
        z_feat = self.latent_embed(z_t)
        t_feat = self.time_embed(t)
        fused = torch.cat([z_feat, t_feat], dim=1)
        fused = self.dropout(fused)  # Apply dropout
        return self.vector_field(fused)

# -----------------------------
# 3) Continuous-time NoProp training
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CT_LNN(dim=3, hidden=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
batch_size = 64
epochs = 100

# Move data to device
X_train_d = X_train.to(device)
Y_train_d = Y_train.to(device)

loss_history = []
# Add early stopping logic
best_loss = float('inf')
patience = 10
patience_counter = 0

print("Checking alpha schedule and SNR before training...")
t_test = torch.linspace(0, 1, 100).view(-1, 1).to(device)
alpha_values = model.alpha_bar(t_test)
snr_values = model.snr_prime(t_test)
print(f"Alpha range: {alpha_values.min().item():.6f} to {alpha_values.max().item():.6f}")
print(f"SNR' range: {snr_values.min().item():.6f} to {snr_values.max().item():.6f}")

# Plot alpha schedule
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(t_test.cpu().numpy(), alpha_values.cpu().detach().numpy())
plt.title('Alpha Schedule')
plt.xlabel('Time t')
plt.ylabel('Alpha')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_test.cpu().numpy(), snr_values.cpu().detach().numpy())
plt.title('SNR Prime')
plt.xlabel('Time t')
plt.ylabel('SNR\'')
plt.grid(True)
plt.tight_layout()
plt.show()

for epoch in range(1, epochs+1):
    model.train()
    perm = torch.randperm(X_train_d.size(0))
    total_loss = 0.0
    for i in range(0, X_train_d.size(0), batch_size):
        idx = perm[i:i+batch_size]
        u_y = Y_train_d[idx]
        B = u_y.size(0)

        # Sample t ~ Uniform(0,1)
        t = torch.rand(B, 1, device=device)
        # Construct noisy state z_t
        alpha = model.alpha_bar(t)
        noise = torch.randn_like(u_y)
        z_t = torch.sqrt(alpha) * u_y + torch.sqrt(1 - alpha) * noise

        # Predict drift
        v_pred = model.drift(z_t, t)
        # True drift
        score = (u_y - z_t) / torch.sqrt(1 - alpha)
        v_true = -0.5 * torch.sqrt(1 - alpha) * score

        # Loss weighted by snr'
        # snr_p = model.snr_prime(t)
        # mse = F.mse_loss(v_pred, v_true, reduction='none').mean(dim=1)
        # loss = (snr_p.squeeze() * mse).mean()

        # Fixed version - take absolute value and ensure positive weighting
        snr_p = torch.abs(model.snr_prime(t))  # Take abs value to ensure positive 
        mse = F.mse_loss(v_pred, v_true, reduction='none').mean(dim=1)
        loss = (snr_p.squeeze() * mse).mean()

        # Alternative fix - use simple MSE without weighting if SNR is unstable
        # loss = F.mse_loss(v_pred, v_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * B

    avg_loss = total_loss / X_train_d.size(0)
    loss_history.append(avg_loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, CT NoProp Loss: {avg_loss:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        # Save model
        torch.save(model.state_dict(), 'best_lnn_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Plot training loss
plt.figure()
plt.plot(range(1, len(loss_history)+1), loss_history)
plt.title('CT NoProp Loss (Lorenz)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# -----------------------------
# 4) Evaluation: simulate learned dynamics
# -----------------------------
model.eval()
with torch.no_grad():
    def learned_drift(t, state):
        state = state.view(1,3)
        # Add gradient clipping to prevent explosion
        with torch.no_grad():  # Make sure we don't track gradients during inference
            drift = model.drift(state.to(device), t.view(1,1).to(device))
            # Clip extremely large values
            drift = torch.clamp(drift, min=-50.0, max=50.0)
        return drift.view(3)

    # Use fixed step solver instead of adaptive
    pred_states = odeint(
        learned_drift, 
        init_state.to(device), 
        time,
        method='rk4',           # Use 4th order Runge-Kutta (fixed step)
        options={'step_size': 0.01}  # Small fixed step size
    ).cpu()

# Plot true vs predicted trajectories (x component)
plt.figure()
plt.plot(time.numpy(), true_states[:,0].numpy(), label='True x(t)')
plt.plot(time.numpy(), pred_states[:,0].numpy(), '--', label='Pred x(t)')
plt.title('Lorenz Attractor: True vs Predicted (x dimension)')
plt.xlabel('Time')
plt.ylabel('x')
plt.legend()
plt.grid(True)
plt.show()

# After evaluation
# 3D trajectory plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(true_states[:,0].numpy(), true_states[:,1].numpy(), true_states[:,2].numpy(), 
       label='True trajectory')
ax.plot(pred_states[:,0].numpy(), pred_states[:,1].numpy(), pred_states[:,2].numpy(), 
       '--', label='Predicted trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('Lorenz Attractor: 3D Trajectory')
plt.savefig('lorenz_trajectory.png', dpi=300)
plt.show()

# Add phase space plot
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(true_states[:,0].numpy(), true_states[:,1].numpy(), 'b')
plt.plot(pred_states[:,0].numpy(), pred_states[:,1].numpy(), 'r--')
plt.title('X vs Y')
plt.xlabel('X'); plt.ylabel('Y')

plt.subplot(1, 3, 2)
plt.plot(true_states[:,0].numpy(), true_states[:,2].numpy(), 'b')
plt.plot(pred_states[:,0].numpy(), pred_states[:,2].numpy(), 'r--')
plt.title('X vs Z')
plt.xlabel('X'); plt.ylabel('Z')

plt.subplot(1, 3, 3)
plt.plot(true_states[:,1].numpy(), true_states[:,2].numpy(), 'b')
plt.plot(pred_states[:,1].numpy(), pred_states[:,2].numpy(), 'r--')
plt.title('Y vs Z')
plt.xlabel('Y'); plt.ylabel('Z')

plt.tight_layout()
plt.savefig('lorenz_phase_space.png', dpi=300)
plt.show()

# After training loop
model.eval()
X_test_d = X_test.to(device)
Y_test_d = Y_test.to(device)
with torch.no_grad():
    test_loss = 0.0
    for i in range(0, X_test_d.size(0), batch_size):
        # Similar to training code but on test data
        u_y = Y_test_d[i:i+batch_size]
        t = torch.rand(u_y.size(0), 1, device=device)
        alpha = model.alpha_bar(t)
        noise = torch.randn_like(u_y)
        z_t = torch.sqrt(alpha) * u_y + torch.sqrt(1 - alpha) * noise
        v_pred = model.drift(z_t, t)
        score = (u_y - z_t) / torch.sqrt(1 - alpha)
        v_true = -0.5 * torch.sqrt(1 - alpha) * score
        test_loss += F.mse_loss(v_pred, v_true).item() * u_y.size(0)
    
    print(f"Test loss: {test_loss/X_test_d.size(0):.6f}")

# Debug code to check model outputs
with torch.no_grad():
    test_state = init_state.to(device).view(1,3)
    test_t = torch.tensor([[0.0]]).to(device)
    drift = model.drift(test_state, test_t)
    print(f"Initial drift magnitude: {drift.abs().mean().item():.4f}")
    print(f"Initial drift values: {drift.cpu().numpy()}")
    
    # Check drift across state space
    print("\nDrift at various points:")
    for x in [-10, 0, 10]:
        for y in [-10, 0, 10]:
            for z in [-10, 0, 10]:
                if x == 0 and y == 0 and z == 0:
                    continue
                test_pt = torch.tensor([[x, y, z]]).to(device)
                d = model.drift(test_pt, test_t)
                mag = d.abs().mean().item()
                if mag > 50:
                    print(f"Large drift at ({x},{y},{z}): {mag:.4f}")