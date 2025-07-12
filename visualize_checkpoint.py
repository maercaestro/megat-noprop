import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.models import NoPropModel
from utils.visualization import (
    visualize_denoising,
    visualize_embeddings,
    create_denoising_animation,
    block_accuracy_analysis,
    class_embedding_analysis
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set up test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load model configuration (adjust these values to match your saved model)
model = NoPropModel(
    num_classes=10,
    embedding_dim=128,
    T=10,
    eta=1.0
)

# Load model checkpoint
checkpoint = torch.load('checkpoints/mnist/model_best.pth')
model = NoPropModel(
    num_classes=10, 
    embedding_dim=checkpoint['embedding_dim'],
    T=checkpoint['T'],
    eta=checkpoint['eta']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

model.to(device)
model.eval()

print("Model loaded successfully!")
print(f"Running visualizations on model from checkpoint: {checkpoint}")

# Run visualizations
print("\n1. Analyzing class embeddings...")
class_embedding_analysis(model)

print("\n2. Analyzing block-wise accuracy...")
block_accuracy_analysis(model, test_loader, device)

print("\n3. Visualizing denoising process...")
visualize_denoising(model, test_loader, device)

try:
    print("\n4. Creating t-SNE embedding visualization (this may take a while)...")
    visualize_embeddings(model, test_loader, device)
except Exception as e:
    print(f"t-SNE visualization failed: {e}")

# Animation is the slowest, so make it optional
create_animation = input("\nCreate denoising animation? (y/n): ").lower().startswith('y')
if create_animation:
    try:
        print("Creating denoising animation...")
        create_denoising_animation(model, test_loader, device)
        print("Animation saved to denoising_animation.gif")
    except Exception as e:
        print(f"Animation creation failed: {e}")

print("\nAll visualizations completed!")