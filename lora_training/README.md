# LoRA Training Variants for Workshop Demo

This folder contains everything needed to train 12 LoRA variants with different hyperparameters on Runpod.

## What Gets Trained

| Rank | Steps | Learning Rate | Output Name |
|------|-------|---------------|-------------|
| 8 | 250, 500, 1000 | 3e-4 | filmlut_r8_s250, etc. |
| 16 | 250, 500, 1000 | 2e-4 | filmlut_r16_s250, etc. |
| 32 | 250, 500, 1000 | 1e-4 | filmlut_r32_s250, etc. |
| 64 | 250, 500, 1000 | 1e-4 | filmlut_r64_s250, etc. |

**Total: 12 LoRAs** to demonstrate the effect of rank and training duration.

## Estimated Time & Cost

- **Per LoRA:** ~10-20 min on A100/H100
- **Total:** ~3-4 hours for all 12
- **Runpod cost:** ~$8-15 depending on GPU

## Quick Start on Runpod

### 1. Start a Runpod Instance

- **Template:** PyTorch 2.x
- **GPU:** A100 80GB or H100 (recommended) / A40 48GB (minimum)
- **Disk:** 100GB+

### 2. Upload Files

Upload these to `/workspace/`:

```
/workspace/
├── training_dataset/        # Your dataset (from local)
│   ├── dataset/
│   │   ├── dataset_1.png
│   │   ├── dataset_1.txt
│   │   └── ...
│   └── control/
│       └── ...
└── lora_training/           # This folder
    ├── configs/
    ├── generate_configs.py
    ├── run_all_training.sh
    └── setup_runpod.sh
```

### 3. Run Setup

```bash
cd /workspace/lora_training
bash setup_runpod.sh
```

This will:
- Clone AI Toolkit
- Install dependencies
- Login to Hugging Face (have your token ready)

### 4. Start Training

```bash
cd /workspace/lora_training
bash run_all_training.sh
```

### 5. Download Results

When done, your LoRAs will be in:
```
/workspace/lora_outputs/
├── filmlut_r8_s250/
├── filmlut_r8_s500/
├── filmlut_r8_s1000/
├── filmlut_r16_s250/
└── ... (12 folders total)
```

Each folder contains:
- `*.safetensors` - The LoRA weights
- `samples/` - Generated samples during training

## Using the LoRAs in Your Workshop

```python
from diffusers import Flux2KleinPipeline

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-base-4B",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# Load a specific LoRA
pipe.load_lora_weights("./lora_outputs/filmlut_r32_s500")

# Generate
image = pipe(
    "filmlut, portrait of a woman in golden hour light",
    num_inference_steps=50,
    guidance_scale=4.0
).images[0]
```

## Customizing

Edit `generate_configs.py` to change:
- `RANKS` - Which ranks to test
- `STEPS_LIST` - Which step counts to test
- `LEARNING_RATES` - LR per rank
- `TRIGGER_WORD` - Your trigger word

Then regenerate:
```bash
python generate_configs.py
```

## Troubleshooting

**"Out of memory"**
- Use a larger GPU (A100 80GB recommended)
- Or add `"low_vram": true` to model config

**"Model not found"**
- Make sure you've accepted the license on HF
- Run `huggingface-cli login` with a token that has access

**Training seems stuck**
- Check logs in `/workspace/lora_training/logs/`
- First run caches latents, which takes time
