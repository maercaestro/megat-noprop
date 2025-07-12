# -*- coding: utf-8 -*-
"""
Quick comparison between your original implementation and the improved version.
Run both on MNIST to see the difference in accuracy.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.models import NoPropModel  # Your original model
from models.improved_models import (
    NoPropCTImproved, initialize_with_prototypes, 
    train_step, evaluate_model
)

# Picklable transform class
class ToThreeChannel:
    """Convert single channel to 3 channels."""
    def __call__(self, x):
        return x.repeat(3, 1, 1)

def get_mnist_data(batch_size=256):
    """Get MNIST data for comparison."""
    # For original model (1 channel)
    transform_orig = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # For improved model (3 channels for ResNet)
    transform_improved = transforms.Compose([
        transforms.ToTensor(),
        ToThreeChannel(),  # Use picklable class instead of lambda
        transforms.Normalize((0.1307,) * 3, (0.3081,) * 3)
    ])
    
    # Original model data
    train_orig = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform_orig)
    test_orig = torchvision.datasets.MNIST('./data', train=False, transform=transform_orig)
    
    # Improved model data  
    train_improved = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform_improved)
    test_improved = torchvision.datasets.MNIST('./data', train=False, transform=transform_improved)
    
    # Create loaders
    train_loader_orig = DataLoader(train_orig, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader_orig = DataLoader(test_orig, batch_size=batch_size, shuffle=False, num_workers=0)
    
    train_loader_improved = DataLoader(train_improved, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader_improved = DataLoader(test_improved, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return (train_loader_orig, test_loader_orig, train_loader_improved, test_loader_improved)

def train_original_model(train_loader, test_loader, device, epochs=20):
    """Train your original NoProp model."""
    print("🔧 Training Original NoProp Model...")
    
    model = NoPropModel(
        num_classes=10,
        embedding_dim=128,
        T=5,  # Fewer timesteps for faster training
        eta=1.0
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Add channel dimension if needed
            if len(x.shape) == 3:
                x = x.unsqueeze(1)
            
            optimizer.zero_grad()
            output = model(x, y)
            loss = output['loss']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Evaluate every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    if len(x.shape) == 3:
                        x = x.unsqueeze(1)
                    
                    preds = model.inference(x)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            
            accuracy = correct / total
            best_acc = max(best_acc, accuracy)
            print(f"  Epoch {epoch}: Loss {avg_loss:.4f}, Acc {100*accuracy:.2f}%")
    
    print(f"✅ Original model best accuracy: {100*best_acc:.2f}%")
    return best_acc

def train_improved_model(train_loader, test_loader, device, epochs=20):
    """Train the improved NoProp model."""
    print("🚀 Training Improved NoProp Model...")
    
    model = NoPropCTImproved(
        backbone='resnet18',
        num_classes=10,
        time_emb_dim=64,
        embed_dim=256,
        input_channels=3,
        use_resnet=True
    ).to(device)
    
    # Initialize with prototypes
    print("  Initializing prototypes...")
    dataset = torchvision.datasets.MNIST('./data', train=True, 
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            ToThreeChannel(),  # Use picklable class
                                            transforms.Normalize((0.1307,) * 3, (0.3081,) * 3)
                                        ]))
    
    W_proto, _ = initialize_with_prototypes(model, dataset, 10, device)
    with torch.no_grad():
        model.W_embed.copy_(W_proto)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        total_losses = {'total_loss': 0, 'ce_loss': 0, 'kl_loss': 0, 'sdm_loss': 0}
        num_batches = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            losses = train_step(model, x, y, optimizer, device, η=1.0)
            
            for key, value in losses.items():
                total_losses[key] += value
            num_batches += 1
        
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        # Evaluate every 5 epochs
        if epoch % 5 == 0:
            accuracy = evaluate_model(model, test_loader, device, T_steps=20, use_heun=True)
            best_acc = max(best_acc, accuracy)
            print(f"  Epoch {epoch}: Loss {avg_losses['total_loss']:.4f}, Acc {100*accuracy:.2f}%")
    
    print(f"✅ Improved model best accuracy: {100*best_acc:.2f}%")
    return best_acc

def main():
    """Run comparison between original and improved models."""
    print("🔬 NoProp Model Comparison on MNIST")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data
    train_orig, test_orig, train_improved, test_improved = get_mnist_data()
    
    # Train both models
    print("\n1️⃣ Original Implementation:")
    start_time = time.time()
    original_acc = train_original_model(train_orig, test_orig, device, epochs=20)
    original_time = time.time() - start_time
    
    print("\n2️⃣ Improved Implementation:")
    start_time = time.time()
    improved_acc = train_improved_model(train_improved, test_improved, device, epochs=20)
    improved_time = time.time() - start_time
    
    # Results
    print("\n" + "=" * 50)
    print("📊 FINAL COMPARISON RESULTS")
    print("=" * 50)
    print(f"Original Model:")
    print(f"  Accuracy: {100*original_acc:.2f}%")
    print(f"  Time: {original_time:.1f}s")
    print()
    print(f"Improved Model:")
    print(f"  Accuracy: {100*improved_acc:.2f}%")
    print(f"  Time: {improved_time:.1f}s")
    print()
    print(f"🎯 Improvement:")
    print(f"  Accuracy: +{100*(improved_acc - original_acc):.2f}%")
    print(f"  Relative gain: {100*(improved_acc/original_acc - 1):.1f}%")
    
    if improved_acc > original_acc:
        print("✅ Improved model is better!")
    else:
        print("❌ Something went wrong...")

if __name__ == '__main__':
    main()
