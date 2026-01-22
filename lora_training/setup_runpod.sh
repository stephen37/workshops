#!/bin/bash
# =============================================================================
# Runpod Setup Script for LoRA Training (FLUX.2 Klein Base)
# Run this ONCE when you start your Runpod instance
# =============================================================================

set -e

echo "=========================================="
echo "  Setting up LoRA Training Environment"
echo "  Model: FLUX.2 Klein Base 4B"
echo "=========================================="

cd /workspace

# -----------------------------------------------------------------------------
# 1. Clone AI Toolkit
# -----------------------------------------------------------------------------
echo ""
echo "[1/4] Cloning AI Toolkit..."
if [ ! -d "ai-toolkit" ]; then
    git clone https://github.com/ostris/ai-toolkit.git
    cd ai-toolkit
    git submodule update --init --recursive
    cd /workspace
else
    echo "AI Toolkit already exists, pulling latest..."
    cd ai-toolkit && git pull && git submodule update --init --recursive && cd /workspace
fi

# -----------------------------------------------------------------------------
# 2. Create venv and install dependencies
# -----------------------------------------------------------------------------
echo ""
echo "[2/4] Setting up Python virtual environment..."
cd /workspace/ai-toolkit

if [ ! -d "venv" ]; then
    python -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt
pip install peft accelerate bitsandbytes

cd /workspace

# -----------------------------------------------------------------------------
# 3. Login to Hugging Face (for model access)
# -----------------------------------------------------------------------------
echo ""
echo "[3/4] Hugging Face login..."
echo "You need a HF token with access to FLUX.2 Klein Base"
echo "Get one at: https://huggingface.co/settings/tokens"
echo ""
huggingface-cli login

# -----------------------------------------------------------------------------
# 4. Create output directories
# -----------------------------------------------------------------------------
echo ""
echo "[4/4] Creating directories..."
mkdir -p /workspace/lora_outputs
mkdir -p /workspace/lora_training/logs

# -----------------------------------------------------------------------------
# Done!
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: Always activate the venv before training:"
echo "  source /workspace/ai-toolkit/venv/bin/activate"
echo ""
echo "To start training all variants:"
echo "  cd /workspace/lora_training"
echo "  source /workspace/ai-toolkit/venv/bin/activate"
echo "  python train_all.py"
echo ""
echo "Or run a single config:"
echo "  python /workspace/ai-toolkit/run.py configs/filmlut_r8_s250.yaml"
echo ""
echo "=========================================="
