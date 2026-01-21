#!/bin/bash
# =============================================================================
# Runpod Setup Script for LoRA Training
# Run this ONCE when you start your Runpod instance
# =============================================================================

set -e

echo "=========================================="
echo "  Setting up LoRA Training Environment"
echo "=========================================="

cd /workspace

# -----------------------------------------------------------------------------
# 1. Clone AI Toolkit
# -----------------------------------------------------------------------------
echo ""
echo "[1/5] Cloning AI Toolkit..."
if [ ! -d "ai-toolkit" ]; then
    git clone https://github.com/ostris/ai-toolkit.git
    cd ai-toolkit
    git submodule update --init --recursive
    cd /workspace
else
    echo "AI Toolkit already exists, pulling latest..."
    cd ai-toolkit && git pull && cd /workspace
fi

# -----------------------------------------------------------------------------
# 2. Install dependencies
# -----------------------------------------------------------------------------
echo ""
echo "[2/5] Installing dependencies..."
cd ai-toolkit
pip install -r requirements.txt
pip install peft accelerate bitsandbytes
cd /workspace

# -----------------------------------------------------------------------------
# 3. Login to Hugging Face (for model access)
# -----------------------------------------------------------------------------
echo ""
echo "[3/5] Hugging Face login..."
echo "You'll need a HF token with access to FLUX.2 Klein Base"
huggingface-cli login

# -----------------------------------------------------------------------------
# 4. Create directory structure
# -----------------------------------------------------------------------------
echo ""
echo "[4/5] Creating directories..."
mkdir -p /workspace/lora_training/configs
mkdir -p /workspace/lora_training/logs
mkdir -p /workspace/lora_outputs
mkdir -p /workspace/training_dataset

# -----------------------------------------------------------------------------
# 5. Instructions
# -----------------------------------------------------------------------------
echo ""
echo "[5/5] Setup complete!"
echo ""
echo "=========================================="
echo "  NEXT STEPS"
echo "=========================================="
echo ""
echo "1. Upload your training dataset to:"
echo "   /workspace/training_dataset/"
echo "   (should contain 'dataset/' folder with images + .txt captions)"
echo ""
echo "2. Upload the lora_training folder:"
echo "   /workspace/lora_training/"
echo "   (contains configs/ and scripts)"
echo ""
echo "3. Generate configs:"
echo "   cd /workspace/lora_training"
echo "   python generate_configs.py"
echo ""
echo "4. Run training:"
echo "   bash run_all_training.sh"
echo ""
echo "5. Download results from:"
echo "   /workspace/lora_outputs/"
echo ""
echo "=========================================="
