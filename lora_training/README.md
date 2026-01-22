# LoRA Training for FLUX.2 Klein Base

Train 12 LoRA variants with different hyperparameters to demonstrate the effect of rank and training duration.

## Training Matrix

| Rank | Steps | Learning Rate | Output Name |
|------|-------|---------------|-------------|
| 8 | 250, 500, 1000 | 3e-4 | filmlut_r8_s250, etc. |
| 16 | 250, 500, 1000 | 2e-4 | filmlut_r16_s250, etc. |
| 32 | 250, 500, 1000 | 1e-4 | filmlut_r32_s250, etc. |
| 64 | 250, 500, 1000 | 1e-4 | filmlut_r64_s250, etc. |

**Total: 12 LoRAs**

## Quick Start on Runpod

### 1. Start a Runpod Instance

- **Template:** PyTorch 2.x
- **GPU:** A100 80GB recommended (48GB minimum)
- **Disk:** 100GB+

### 2. Clone the Repo

```bash
cd /workspace
git clone -b together_workshop https://github.com/stephen37/workshops.git
```

### 3. Upload Your Dataset

Upload your training dataset to `/workspace/training_dataset/`:

```
/workspace/training_dataset/
└── dataset/
    ├── image_001.png
    ├── image_001.txt
    ├── image_002.png
    ├── image_002.txt
    └── ...
```

Each `.txt` file should contain the caption with your trigger word (e.g., "filmlut, cinematic portrait").

### 4. Run Setup

```bash
cd /workspace/workshops/lora_training
bash setup_runpod.sh
```

This will:
- Clone and set up AI Toolkit with venv
- Install all dependencies
- Prompt you to login to Hugging Face

### 5. Start Training

```bash
cd /workspace/workshops/lora_training
source /workspace/ai-toolkit/venv/bin/activate
python train_all.py
```

This runs all 12 variants sequentially, starting with the smallest (rank 8) for quick verification.

### 6. Download Results

When done, your LoRAs will be in `/workspace/lora_outputs/`:

```
/workspace/lora_outputs/
├── filmlut_r8_s250/
│   ├── filmlut_r8_s250.safetensors
│   └── samples/
├── filmlut_r8_s500/
└── ... (12 folders total)
```

## Running a Single Variant

To test with just one config:

```bash
source /workspace/ai-toolkit/venv/bin/activate
python /workspace/ai-toolkit/run.py configs/filmlut_r8_s250.yaml
```

## Using the Trained LoRAs

```python
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-base-4B",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# Load a specific LoRA
pipe.load_lora_weights("./lora_outputs/filmlut_r32_s500")

# Generate with trigger word
image = pipe(
    "filmlut, portrait of a woman in golden hour light",
    num_inference_steps=50,
    guidance_scale=4.0
).images[0]
```

## Customizing

Edit `generate_configs.py` to change:
- `RANKS` - Which ranks to test
- `STEPS_LIST` - Which step counts
- `LEARNING_RATES` - LR per rank
- `TRIGGER_WORD` - Your trigger word

Then regenerate:
```bash
python generate_configs.py
```

## Troubleshooting

**"Out of memory"**
- Use A100 80GB instead of 48GB GPU
- The configs already use `low_vram: true`

**"Model not found"**
- Accept the license on HuggingFace: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B
- Run `huggingface-cli login` with a token that has access

**Training seems stuck**
- First run caches latents to disk, which takes time
- Check logs in `/workspace/lora_training/logs/`
