#!/bin/bash
# Quick launch script for testing improved NoProp

echo "🚀 Testing Improved NoProp Implementation"
echo "========================================"

# Test on MNIST with simple CNN (faster)
echo "1️⃣ Testing with Simple CNN on MNIST (quick test):"
python examples/train_improved.py \
    --dataset mnist \
    --backbone resnet18 \
    --no-resnet \
    --epochs 10 \
    --batch-size 512 \
    --embed-dim 128

echo ""
echo "2️⃣ Testing with ResNet18 on MNIST (full power):"
python examples/train_improved.py \
    --dataset mnist \
    --backbone resnet18 \
    --epochs 15 \
    --batch-size 256 \
    --embed-dim 256

echo ""
echo "✅ Tests complete! Check the results above."
