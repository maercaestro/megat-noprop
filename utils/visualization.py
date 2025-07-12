import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from models.models import NoPropModelCT, NoPropModelFM, NoPropModel
from models.block import DenoiseBlock

def visualize_denoising(model, test_loader, device):
    """Visualize the denoising process for a batch of images."""
    model.eval()
    # Get a batch of test images
    x, y = next(iter(test_loader))
    x, y = x.to(device), y.to(device)
    
    # Number of images to visualize
    n_samples = min(8, x.size(0))
    
    # For discrete model (showing intermediate steps)
    if hasattr(model, 'T'):  # Discrete time model
        steps = model.T
        fig, axes = plt.subplots(n_samples, steps+1, figsize=(2*(steps+1), 2*n_samples))
        
        # Start with random noise
        z = torch.randn(n_samples, model.embedding_dim, device=device)
        
        # Display input images
        for i in range(n_samples):
            axes[i, 0].imshow(x[i].cpu().squeeze(), cmap='gray')
            axes[i, 0].set_title(f"Input ({y[i].item()})")
            axes[i, 0].axis('off')
        
        # For each denoising step
        for t in range(steps):
            # Denoise one step
            z_new, logits = model.blocks[t](x[:n_samples], z, model.W_embed)
            
            # Get current class predictions
            if t == steps-1:
                preds = model.classifier(z_new).argmax(dim=1)
            else: 
                preds = logits.argmax(dim=1)
            
            # Visualize latent embedding (projected to 2D)
            for i in range(n_samples):
                # Use logits to create a grayscale "image" representation
                # We normalize the logits to 0-1 for visualization
                probs = logits[i].detach().cpu().softmax(dim=0)
                
                # Create a bar chart instead of trying to reshape
                axes[i, t+1].bar(range(10), probs)
                axes[i, t+1].set_ylim(0, 1)
                axes[i, t+1].set_title(f"Step {t+1}: {probs.argmax().item()}")
                axes[i, t+1].set_xticks(range(10))
            
            # Update z for next step
            z = z_new
        
        plt.tight_layout()
        plt.savefig('denoising_process.png')
        plt.show()


def visualize_embeddings(model, test_loader, device):
    """Visualize class embeddings in 2D using t-SNE."""
    from sklearn.manifold import TSNE
    
    model.eval()
    all_embeddings = []
    all_labels = []
    
    # Extract embeddings for test samples
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            if isinstance(model, NoPropModelCT):
                # For CT model
                z = torch.randn(x.size(0), model.embedding_dim, device=device)
                for i in range(100):  # Use fewer steps for visualization
                    t = torch.full((x.size(0), 1), i/100, device=device)
                    z = model.forward_denoise(x, z, t)
            else:
                # For discrete model
                z = torch.randn(x.size(0), model.embedding_dim, device=device)
                for t in range(model.T):
                    z, _ = model.blocks[t](x, z, model.W_embed)
            
            all_embeddings.append(z.cpu())
            all_labels.append(y.cpu())
    
    # Concatenate all embeddings and labels
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings[:2000])  # Limit to 2000 for speed
    
    # Plot
    plt.figure(figsize=(10, 8))
    for i in range(10):  # MNIST has 10 classes
        mask = labels[:2000] == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=f"Class {i}")
    
    plt.title("t-SNE of Learned Embeddings")
    plt.colorbar()
    plt.legend()
    plt.savefig('embedding_tsne.png')
    plt.show()

def create_denoising_animation(model, test_loader, device):
    """Create a GIF showing the denoising process."""
    import matplotlib.animation as animation
    
    model.eval()
    x, y = next(iter(test_loader))
    x, y = x.to(device), y.to(device)
    
    # Select a single image
    img = x[0].unsqueeze(0)
    label = y[0].item()
    
    # Initialize figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img.cpu().squeeze(), cmap='gray')
    ax1.set_title(f"Input (Class {label})")
    ax1.axis('off')
    
    # For animation frames
    frames = []
    
    # Initialize with noise
    if isinstance(model, NoPropModelCT):
        # For CT model
        n_steps = 100  # Use more steps for smoother animation
        z = torch.randn(1, model.embedding_dim, device=device)
        
        for i in range(n_steps):
            t = torch.full((1, 1), i/n_steps, device=device)
            z = model.forward_denoise(img, z, t)
            
            # Get logits
            logits = model.classifier(z)
            probs = F.softmax(logits, dim=1).squeeze().cpu().detach()
            
            # Create bar plot
            frame = ax2.bar(range(10), probs)
            ax2.set_ylim(0, 1)
            ax2.set_title(f"Step {i+1}/{n_steps}: Pred {logits.argmax().item()}")
            frames.append(frame)
    else:
        # For discrete model
        z = torch.randn(1, model.embedding_dim, device=device)
        
        for t in range(model.T):
            z, logits = model.blocks[t](img, z, model.W_embed)
            
            # Get probabilities
            probs = F.softmax(logits, dim=1).squeeze().cpu().detach()
            
            # Create bar plot
            frame = ax2.bar(range(10), probs)
            ax2.set_ylim(0, 1)
            ax2.set_title(f"Step {t+1}/{model.T}: Pred {logits.argmax().item()}")
            frames.append(frame)
    
    # Create animation
    ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True)
    ani.save('denoising_animation.gif', writer='pillow')
    plt.show()

def block_accuracy_analysis(model, test_loader, device):
    """Analyze and visualize accuracy using only k blocks."""
    if not hasattr(model, 'T'):  # Only for discrete model
        print("This visualization only works for discrete time model")
        return
        
    accuracies = []
    
    for k in range(1, model.T + 1):
        correct, total = 0, 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                # Start with random noise
                z = torch.randn(x.size(0), model.embedding_dim, device=device)
                
                # Run through only k blocks
                for t in range(k):
                    z, _ = model.blocks[t](x, z, model.W_embed)
                
                # Get predictions
                logits = model.classifier(z)
                preds = logits.argmax(dim=1)
                
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        accuracies.append(100 * correct / total)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, model.T + 1), accuracies, marker='o')
    plt.xlabel("Number of Blocks Used")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs. Number of Blocks")
    plt.grid(True)
    plt.savefig('block_accuracy.png')
    plt.show()


def class_embedding_analysis(model):
    """Analyze and visualize class embedding properties."""
    # Get embedding matrix
    W = model.W_embed.detach().cpu()
    
    # Compute pairwise distances
    distances = torch.cdist(W, W, p=2)
    
    # Compute norms
    norms = torch.norm(W, dim=1)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot pairwise distances
    im = ax1.imshow(distances, cmap='viridis')
    ax1.set_title("Pairwise Distances Between Class Embeddings")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Class")
    plt.colorbar(im, ax=ax1)
    
    # Plot norms
    ax2.bar(range(10), norms)
    ax2.set_title("Class Embedding Norms")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("L2 Norm")
    
    plt.tight_layout()
    plt.savefig('class_embeddings.png')
    plt.show()