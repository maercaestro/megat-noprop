import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from models.models import NoPropModel  # your updated discrete‐time model

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_noprop(model, train_loader, val_loader, epochs, lr, weight_decay):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {'train_acc': [], 'val_acc': []}
    
    # Print initial model state
    print(f"Initial embedding norm: {torch.norm(model.W_embed).item():.6f}")
    
    # Track the time spent in each section
    times = {"train": 0, "eval": 0}
    
    # Add these variables for tracking best model
    best_val_acc = 0.0
    os.makedirs('checkpoints/mnist', exist_ok=True)
    
    for epoch in range(1, epochs+1):
        epoch_start = time.time()
        model.train()
        
        # Metrics for this epoch
        epoch_loss_l2 = 0
        epoch_loss_ce = 0
        epoch_loss_kl = 0
        
        # ---- Loop over timesteps instead of full unrolled chain ----
        for t in range(model.T):
            print(f"Epoch {epoch}/{epochs}, Block {t+1}/{model.T}")
            
            # Add progress bar for batches
            pbar = tqdm(train_loader, desc=f"Training block {t+1}")
            batch_count = 0
            
            for x, y in pbar:
                batch_start = time.time()
                batch_count += 1
                x, y = x.to(device), y.to(device)
                
                # Shape check
                if batch_count == 1:
                    print(f"Input shape: {x.shape}, Target shape: {y.shape}")
                
                # Prepare true embeddings
                uy = F.embedding(y, model.W_embed)
                
                # Sample z_t
                alpha_bar_t = model.alpha_bar[t]
                noise = torch.randn_like(uy)
                z_t = torch.sqrt(alpha_bar_t) * uy + torch.sqrt(1 - alpha_bar_t) * noise
                
                # Reshape input for proper processing if needed
                if len(x.shape) == 3:  # Missing channel dimension
                    x = x.unsqueeze(1)  # Add channel dimension
                
                # Shape check for first batch
                if batch_count == 1:
                    print(f"z_t shape: {z_t.shape}")
                
                # Predict next latent via block t
                z_pred, logits = model.blocks[t](x, z_t, model.W_embed)
                
                # L2 loss weighted
                loss_l2 = F.mse_loss(z_pred, uy)  # Note: against clean uy for better training
                loss = 0.5 * model.eta * model.snr_delta[t] * loss_l2
                epoch_loss_l2 += loss_l2.item()
                
                # At final step add CE + KL
                if t == model.T - 1:
                    logits = model.classifier(z_pred)
                    loss_ce = F.cross_entropy(logits, y)
                    loss_kl = 0.01 * 0.5 * (uy.pow(2).sum(dim=1)).mean()  # Reduced KL weight
                    loss = loss + loss_ce + loss_kl
                    epoch_loss_ce += loss_ce.item()
                    epoch_loss_kl += loss_kl.item()
                    
                    # Calculate accuracy for this batch
                    preds = torch.argmax(logits, dim=1)
                    acc = (preds == y).float().mean().item() * 100
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{acc:.2f}%"
                    })
                else:
                    pbar.set_postfix({'loss_l2': f"{loss_l2.item():.4f}"})
                
                # Every 100 batches, check embedding norms
                if batch_count % 100 == 0:
                    embed_norm = torch.norm(model.W_embed).item()
                    print(f"  Batch {batch_count}: Embedding norm = {embed_norm:.6f}")
                
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                
                # Add batch processing time
                batch_time = time.time() - batch_start
                if batch_count % 100 == 0:
                    print(f"  Batch time: {batch_time:.4f}s")
            
            # Print block summary
            avg_loss_l2 = epoch_loss_l2 / (batch_count * (t + 1))
            print(f"Block {t+1} complete. Avg L2 loss: {avg_loss_l2:.6f}")
        
        # ——— Evaluate accuracies ———
        times["train"] += time.time() - epoch_start
        eval_start = time.time()
        
        def eval_loader(loader, name="evaluation"):
            correct = 0
            total = 0
            model.eval()
            print(f"Running {name}...")
            
            with torch.no_grad():
                for batch_idx, (x, y) in enumerate(tqdm(loader, desc=name)):
                    x, y = x.to(device), y.to(device)
                    
                    # Add channel dimension if needed
                    if len(x.shape) == 3:
                        x = x.unsqueeze(1)
                    
                    # Debug first batch inference
                    start_time = time.time()
                    if batch_idx == 0:
                        print(f"Running inference on first batch...")
                        # Print shape information
                        print(f"  Input shape: {x.shape}")
                        
                        # Get starting embedding norm
                        emb_norm = torch.norm(model.W_embed).item()
                        print(f"  Embedding norm: {emb_norm:.6f}")
                        
                        # Get predictions with timing information
                        z = torch.randn(x.size(0), model.embedding_dim, device=device)
                        for t_idx in range(model.T):
                            block_start = time.time()
                            z_pred, _ = model.blocks[t_idx](x, z, model.W_embed)
                            z = z_pred
                            print(f"  Block {t_idx+1}/{model.T}: {time.time() - block_start:.4f}s")
                        
                        logits = model.classifier(z)
                        preds = torch.argmax(logits, dim=1)
                        
                        # Display first few examples
                        for i in range(min(5, x.size(0))):
                            print(f"  Sample {i}: Pred={preds[i].item()}, True={y[i].item()}")
                    
                    # Normal inference for accuracy calculation
                    preds = model.inference(x)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
                    
                    if batch_idx == 0:
                        print(f"Inference time: {time.time() - start_time:.4f}s")
                    
            return correct / total
        
        train_acc = eval_loader(train_loader, "train evaluation")
        val_acc = eval_loader(val_loader, "validation")
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        times["eval"] += time.time() - eval_start
        
        # Print epoch summary with more stats
        print(f"Epoch {epoch:2d}/{epochs:2d}  "
              f"Train Acc: {100*train_acc:5.2f}%  "
              f"Val Acc: {100*val_acc:5.2f}%  "
              f"Train time: {times['train']/epoch:.1f}s/epoch  "
              f"Eval time: {times['eval']/epoch:.1f}s/epoch")
        
        # Print embedding norms for each class to see if they're separating
        print("Class embedding norms:")
        for i in range(10):  # MNIST has 10 classes
            norm = torch.norm(model.W_embed[i]).item()
            print(f"  Class {i}: {norm:.6f}")
        
        # After computing validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best model! Saving checkpoint with accuracy: {100*val_acc:.2f}%")
            
            # Save the model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'val_acc': val_acc,
                'embedding_dim': model.embedding_dim,
                'T': model.T,
                'eta': model.eta
            }, 'checkpoints/mnist/model_best.pth')
    
    # Also save final model regardless of performance
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'val_acc': history['val_acc'][-1],
        'embedding_dim': model.embedding_dim,
        'T': model.T,
        'eta': model.eta
    }, 'checkpoints/mnist/model_final.pth')
    
    print(f"Saved final model to checkpoints/mnist/model_final.pth")
    
    # ——— Plot learning curves ———
    plt.figure(figsize=(6,4))
    plt.plot(range(1, epochs+1), history['train_acc'], label='Train')
    plt.plot(range(1, epochs+1), history['val_acc'],   label='Validation')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('NoProp MNIST Training')
    plt.legend(); plt.grid(True)
    plt.savefig('noprop_mnist_accuracy.png')
    plt.show()

    print(f"\nFinal test accuracy: {100*history['val_acc'][-1]:.2f}%")

if __name__ == "__main__":
    # Hyperparams
    T = 10
    eta = 0.1
    embedding_dim = 512
    batch_size = 128
    lr = 1e-3
    epochs = 20
    weight_decay = 1e-3

    # Data
    transform = transforms.ToTensor()
    train_ds = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    test_ds  = torchvision.datasets.MNIST(root='./data', train=False,
                                          download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # Model
    model = NoPropModel(num_classes=10,
                        embedding_dim=embedding_dim,
                        T=T, eta=eta).to(device)

    # Train & evaluate
    train_noprop(model, train_loader, test_loader,
                 epochs, lr, weight_decay)

    # ——— Show a few predictions ———
    model.eval()
    classes = [str(i) for i in range(10)]
    plt.figure(figsize=(5,5))
    shown = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model.inference(x)
            for i in range(x.size(0)):
                if shown >= 16: break
                plt.subplot(4,4,shown+1)
                img = x[i].cpu().squeeze()
                plt.imshow(img, cmap='gray')
                plt.title(f"P:{classes[preds[i]]} T:{classes[y[i]]}", fontsize=8)
                plt.axis('off')
                shown += 1
            if shown >= 16: break
    plt.tight_layout()
    plt.show()

    # Import visualization functions
    from utils.visualization import (
        visualize_denoising,
        visualize_embeddings,
        create_denoising_animation,
        block_accuracy_analysis,
        class_embedding_analysis
    )

    print("\n===== Running Visualizations =====")

    # Basic class embedding analysis (quick)
    print("1. Analyzing class embeddings...")
    class_embedding_analysis(model)

    # Block-wise accuracy analysis (medium speed)
    print("2. Analyzing block-wise accuracy...")
    block_accuracy_analysis(model, test_loader, device)

    # Denoising process visualization (quick)
    print("3. Visualizing denoising process...")
    visualize_denoising(model, test_loader, device)

    # Try more intensive visualizations with error handling
    try:
        print("4. Creating t-SNE embedding visualization (this may take a while)...")
        visualize_embeddings(model, test_loader, device)
    except Exception as e:
        print(f"t-SNE visualization failed: {e}")

    # Animation is the slowest, so make it optional
    create_animation = input("Create denoising animation? (y/n): ").lower().startswith('y')
    if create_animation:
        try:
            print("5. Creating denoising animation...")
            create_denoising_animation(model, test_loader, device)
            print("Animation saved to denoising_animation.gif")
        except Exception as e:
            print(f"Animation creation failed: {e}")