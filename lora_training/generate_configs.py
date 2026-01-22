#!/usr/bin/env python3
"""
Generate AI Toolkit config files for multiple LoRA training variants.
This creates configs for different rank/steps combinations to demonstrate
how hyperparameters affect LoRA training.
"""

import yaml
from pathlib import Path
from itertools import product

# =============================================================================
# CONFIGURATION - Edit these for your setup
# =============================================================================

# Base paths (will be updated for Runpod)
DATASET_PATH = "/workspace/training_dataset"  # Where you'll upload the dataset on Runpod
OUTPUT_BASE = "/workspace/lora_outputs"
CONFIG_DIR = Path("configs")

# Model
MODEL_NAME = "black-forest-labs/FLUX.2-klein-base-4B"

# Trigger word
TRIGGER_WORD = "filmlut"

# Experiment variants
RANKS = [8, 16, 32, 64]
STEPS_LIST = [250, 500, 1000]
LEARNING_RATES = {
    # Adjust LR based on rank (higher rank can use slightly lower LR)
    8: 3e-4,
    16: 2e-4,
    32: 1e-4,
    64: 1e-4,
}

# Fixed settings
ALPHA_RATIO = 1.0  # alpha = rank * ratio
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 1
RESOLUTION = 1024
SAVE_EVERY_N_STEPS = 100

# =============================================================================
# CONFIG TEMPLATE
# =============================================================================

def create_config(rank: int, steps: int, lr: float, run_name: str) -> dict:
    """Create an AI Toolkit config dictionary."""

    alpha = int(rank * ALPHA_RATIO)

    config = {
        "job": "extension",
        "config": {
            # Naming
            "name": run_name,
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": OUTPUT_BASE,
                    "device": "cuda:0",
                    "trigger_word": TRIGGER_WORD,
                    "network": {
                        "type": "lora",
                        "linear": rank,
                        "linear_alpha": alpha,
                    },
                    "save": {
                        "dtype": "float16",
                        "save_every": SAVE_EVERY_N_STEPS,
                        "max_step_saves_to_keep": 3,
                    },
                    "datasets": [
                        {
                            "folder_path": f"{DATASET_PATH}/dataset",
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0.05,
                            "shuffle_tokens": False,
                            "cache_latents_to_disk": True,
                            "resolution": [RESOLUTION, RESOLUTION],
                        }
                    ],
                    "train": {
                        "batch_size": BATCH_SIZE,
                        "steps": steps,
                        "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw8bit",
                        "lr": lr,
                        "ema_config": {
                            "use_ema": True,
                            "ema_decay": 0.99,
                        },
                        "dtype": "bf16",
                    },
                    "model": {
                        "name_or_path": MODEL_NAME,
                        "arch": "flux2_klein_4b",
                        "low_vram": True,
                    },
                    "sample": {
                        "sampler": "flowmatch",
                        "sample_every": 100,
                        "width": RESOLUTION,
                        "height": RESOLUTION,
                        "prompts": [
                            f"{TRIGGER_WORD}, cinematic photo of a woman",
                            f"{TRIGGER_WORD}, landscape photograph of mountains",
                            f"{TRIGGER_WORD}, portrait of a man in natural light",
                            "photo of a sunset over the ocean",  # No trigger - test for bleed
                        ],
                        "neg": "",
                        "seed": 42,
                        "walk_seed": True,
                        "guidance_scale": 4.0,
                        "sample_steps": 50,
                    },
                }
            ],
        },
        # Metadata for tracking
        "meta": {
            "rank": rank,
            "alpha": alpha,
            "steps": steps,
            "lr": lr,
            "trigger": TRIGGER_WORD,
        }
    }

    return config


def generate_all_configs():
    """Generate config files for all experiment variants."""

    CONFIG_DIR.mkdir(exist_ok=True)

    configs_generated = []

    for rank, steps in product(RANKS, STEPS_LIST):
        lr = LEARNING_RATES[rank]
        run_name = f"filmlut_r{rank}_s{steps}"

        config = create_config(rank, steps, lr, run_name)

        # Save config
        config_path = CONFIG_DIR / f"{run_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        configs_generated.append({
            "name": run_name,
            "path": str(config_path),
            "rank": rank,
            "steps": steps,
            "lr": lr,
        })

        print(f"Generated: {config_path}")

    # Generate summary
    print(f"\n{'='*60}")
    print(f"Generated {len(configs_generated)} configs:")
    print(f"{'='*60}")
    print(f"{'Name':<25} {'Rank':<6} {'Steps':<6} {'LR':<10}")
    print(f"{'-'*25} {'-'*6} {'-'*6} {'-'*10}")
    for c in configs_generated:
        print(f"{c['name']:<25} {c['rank']:<6} {c['steps']:<6} {c['lr']:<10.0e}")

    return configs_generated


if __name__ == "__main__":
    generate_all_configs()
