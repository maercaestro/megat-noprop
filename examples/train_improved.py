# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Improved NoProp training script with all optimizations:
- Prototype initialization
- Learnable noise schedule  
- ResNet/CNN backbone options
- ODE solvers for inference
- Comprehensive evaluation
"""

import argparse
import math
import time
import gc
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Add parent directory to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.improved_models import (
    NoPropCTImproved, initialize_with_prototypes,
    train_step, run_heun_inference, run_euler_inference, evaluate_model
)

# Picklable transform classes to replace lambda functions
class RepeatChannels:
    """Transform to repeat single channel to 3 channels for ResNet compatibility."""
    def __call__(self, x):
        return x.repeat(3, 1, 1)

class ToThreeChannel:
    """Convert single channel to 3 channels."""
    def __init__(self, channels=3):
        self.channels = channels
    
    def __call__(self, x):
        if x.size(0) == 1 and self.channels == 3:
            return x.repeat(3, 1, 1)
        return x

def get_dataset(dataset_name: str, data_root: str):
    """Get dataset with appropriate transforms."""
    
    if dataset_name == 'mnist':
        # MNIST with 3-channel conversion for ResNet compatibility
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            ToThreeChannel(3),  # Replace lambda with picklable class
            transforms.Normalize((0.1307,) * 3, (0.3081,) * 3),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            ToThreeChannel(3),  # Replace lambda with picklable class
            transforms.Normalize((0.1307,) * 3, (0.3081,) * 3),
        ])
        
        ds_train = torchvision.datasets.MNIST(
            data_root, train=True, download=True, transform=transform_train)
        ds_test = torchvision.datasets.MNIST(
            data_root, train=False, download=True, transform=transform_test)
        num_classes = 10
        input_channels = 3
        
    elif dataset_name == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        ds_train = torchvision.datasets.CIFAR10(
            data_root, train=True, download=True, transform=transform_train)
        ds_test = torchvision.datasets.CIFAR10(
            data_root, train=False, download=True, transform=transform_test)
        num_classes = 10
        input_channels = 3
        
    elif dataset_name == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        ds_train = torchvision.datasets.CIFAR100(
            data_root, train=True, download=True, transform=transform_train)
        ds_test = torchvision.datasets.CIFAR100(
            data_root, train=False, download=True, transform=transform_test)
        num_classes = 100
        input_channels = 3
        
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'")
    
    return ds_train, ds_test, num_classes, input_channels

def train_and_eval(
    backbone: str, 
    time_emb_dim: int, 
    embed_dim: int, 
    dataset: str, 
    data_root: str, 
    epochs: int,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    eta: float = 1.0,
    use_resnet: bool = True
):
    """Main training and evaluation loop."""
    
    print(f"🚀 Starting improved NoProp training")
    print(f"Dataset: {dataset.upper()}")
    print(f"Backbone: {backbone}")
    print(f"Use ResNet: {use_resnet}")
    print(f"Embed dim: {embed_dim}")
    print(f"Epochs: {epochs}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Get dataset
    ds_train, ds_test, num_classes, input_channels = get_dataset(dataset, data_root)
    
    # Create data loaders
    tr_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, 
                          num_workers=0, drop_last=True)  # Set to 0 to avoid multiprocessing
    te_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, 
                          num_workers=0)  # Set to 0 to avoid multiprocessing
    
    print(f"Train samples: {len(ds_train)}, Test samples: {len(ds_test)}")
    print(f"Classes: {num_classes}, Input channels: {input_channels}")
    
    # Build model
    print("🏗️  Building model...")
    model = NoPropCTImproved(
        backbone=backbone,
        num_classes=num_classes,
        time_emb_dim=time_emb_dim,
        embed_dim=embed_dim,
        input_channels=input_channels,
        use_resnet=use_resnet
    ).to(device)
    
    # Initialize W_embed from prototypes (KEY IMPROVEMENT!)
    print("🎯 Initializing prototypes...")
    W_proto, proto_idxs = initialize_with_prototypes(
        model, ds_train, num_classes, device, samples_per_class=10
    )
    
    with torch.no_grad():
        model.W_embed.copy_(W_proto)
    print(f"✅ Prototype initialization complete")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    print("🏃 Starting training...")
    best_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        
        # Training phase
        model.train()
        total_losses = {'total_loss': 0, 'ce_loss': 0, 'kl_loss': 0, 'sdm_loss': 0}
        num_batches = 0
        
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            losses = train_step(model, x, y, optimizer, device, η=eta)
            
            for key, value in losses.items():
                total_losses[key] += value
            num_batches += 1
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        train_time = time.time() - t0
        
        print(f"Epoch {epoch:03d} | "
              f"Loss {avg_losses['total_loss']:.4f} "
              f"(CE: {avg_losses['ce_loss']:.4f}, "
              f"KL: {avg_losses['kl_loss']:.4f}, "
              f"SDM: {avg_losses['sdm_loss']:.4f}) | "
              f"Train {train_time:.1f}s", end='')
        
        # Evaluation every 5 epochs
        if epoch % 5 == 0:
            eval_t0 = time.time()
            accuracy = evaluate_model(model, te_loader, device, T_steps=40, use_heun=True)
            eval_time = time.time() - eval_t0
            
            print(f" | Acc {100 * accuracy:.2f}% | Eval {eval_time:.1f}s")
            
            # Save best model
            if accuracy > best_acc:
                best_acc = accuracy
                os.makedirs('checkpoints', exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'accuracy': accuracy,
                    'args': {
                        'backbone': backbone,
                        'dataset': dataset,
                        'embed_dim': embed_dim,
                        'time_emb_dim': time_emb_dim,
                        'num_classes': num_classes,
                        'input_channels': input_channels,
                        'use_resnet': use_resnet
                    }
                }, f'checkpoints/best_model_{dataset}_{backbone}.pth')
                print(f"💾 Saved new best model (acc: {100*accuracy:.2f}%)")
        else:
            print()
    
    # Final comprehensive evaluation
    print("\n🔬 Final evaluation with multiple T values:")
    print("Heun's method:")
    for T in [10, 20, 40, 80, 100]:
        ti = time.time()
        accuracy = evaluate_model(model, te_loader, device, T_steps=T, use_heun=True)
        eval_time = time.time() - ti
        print(f"  T={T:3d}: {100*accuracy:.2f}% | {eval_time:.1f}s")
    
    print("Euler method:")
    for T in [10, 20, 40, 80, 100]:
        ti = time.time()
        accuracy = evaluate_model(model, te_loader, device, T_steps=T, use_heun=False)
        eval_time = time.time() - ti
        print(f"  T={T:3d}: {100*accuracy:.2f}% | {eval_time:.1f}s")
    
    print(f"\n✅ Training complete! Best accuracy: {100*best_acc:.2f}%")
    
    # Cleanup
    del model, optimizer, ds_train, ds_test, tr_loader, te_loader
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Improved NoProp Training')
    parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'cifar100'], 
                       required=True, help='Dataset to use')
    parser.add_argument('--data-root', default='./data', help='Data directory')
    parser.add_argument('--backbone', choices=['resnet18', 'resnet34', 'resnet50', 'resnet152'], 
                       default='resnet18', help='Backbone architecture')
    parser.add_argument('--time-emb-dim', type=int, default=64, help='Time embedding dimension')
    parser.add_argument('--embed-dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--eta', type=float, default=1.0, help='SDM loss weight')
    parser.add_argument('--no-resnet', action='store_true', help='Use simple CNN instead of ResNet')
    
    args = parser.parse_args()
    
    print("🎯 Arguments:", args)
    
    train_and_eval(
        backbone=args.backbone,
        time_emb_dim=args.time_emb_dim,
        embed_dim=args.embed_dim,
        dataset=args.dataset,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eta=args.eta,
        use_resnet=not args.no_resnet
    )
