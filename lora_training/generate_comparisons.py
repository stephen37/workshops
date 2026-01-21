#!/usr/bin/env python3
"""
Generate comparison images for workshop slides.
Run this on Runpod after training completes.

Usage:
    cd /workspace/workshops/lora_training
    source /workspace/ai-toolkit/venv/bin/activate
    python generate_comparisons.py
"""

import torch
from diffusers import Flux2KleinPipeline
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Paths
LORA_DIR = Path("/workspace/lora_outputs")
OUTPUT_DIR = Path("/workspace/comparison_assets")
OUTPUT_DIR.mkdir(exist_ok=True)

# Device and dtype
device = "cuda"
dtype = torch.bfloat16

# Load pipeline
print("Loading FLUX.2 Klein Base...")
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-base-4B",
    torch_dtype=dtype
)
pipe.enable_model_cpu_offload()
print("Pipeline ready!")

# Fixed generation settings
def get_generator():
    return torch.Generator(device=device).manual_seed(42)

GEN_KWARGS = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 4.0,
}

def generate_with_lora(prompt, lora_name=None):
    """Generate an image, optionally with a LoRA."""
    if lora_name:
        lora_path = LORA_DIR / lora_name
        weight_file = f"{lora_name}.safetensors"
        if lora_path.exists() and (lora_path / weight_file).exists():
            pipe.load_lora_weights(str(lora_path), weight_name=weight_file)

    image = pipe(
        prompt=prompt,
        generator=get_generator(),
        **GEN_KWARGS
    ).images[0]

    if lora_name:
        try:
            pipe.unload_lora_weights()
        except:
            pass

    return image


def generate_rank_comparison():
    """Compare different ranks at 500 steps."""
    print("\n=== Generating Rank Comparison ===")
    prompt = "filmlut, cinematic portrait of a woman in golden hour light"

    ranks = [8, 16, 32, 64]
    print("  Generating baseline (no LoRA)...")
    images = {"baseline": generate_with_lora(prompt, None)}

    for rank in ranks:
        lora_name = f"filmlut_r{rank}_s500"
        print(f"  Generating rank {rank}...")
        images[f"r{rank}"] = generate_with_lora(prompt, lora_name)

    # Plot
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(images["baseline"])
    axes[0].set_title("No LoRA\n(baseline)", fontsize=11)
    axes[0].axis('off')

    for i, rank in enumerate(ranks):
        axes[i+1].imshow(images[f"r{rank}"])
        axes[i+1].set_title(f"Rank {rank}\n(500 steps)", fontsize=11)
        axes[i+1].axis('off')

    plt.suptitle("Effect of Rank on Style Transfer", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rank_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'rank_comparison.png'}")


def generate_steps_comparison():
    """Compare different step counts at rank 32."""
    print("\n=== Generating Steps Comparison ===")
    prompt = "filmlut, cinematic portrait of a woman in golden hour light"

    steps_list = [250, 500, 1000]
    print("  Generating baseline (no LoRA)...")
    images = {"baseline": generate_with_lora(prompt, None)}

    for steps in steps_list:
        lora_name = f"filmlut_r32_s{steps}"
        print(f"  Generating {steps} steps...")
        images[f"s{steps}"] = generate_with_lora(prompt, lora_name)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(images["baseline"])
    axes[0].set_title("No LoRA\n(baseline)", fontsize=11)
    axes[0].axis('off')

    for i, steps in enumerate(steps_list):
        axes[i+1].imshow(images[f"s{steps}"])
        axes[i+1].set_title(f"{steps} steps\n(rank 32)", fontsize=11)
        axes[i+1].axis('off')

    plt.suptitle("Effect of Training Steps", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "steps_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'steps_comparison.png'}")


def generate_overfitting_test():
    """Test if LoRA still follows prompts."""
    print("\n=== Generating Overfitting Test ===")

    test_prompts = [
        ("with_trigger", "filmlut, landscape photograph of mountains at sunset"),
        ("style_change", "filmlut, watercolor painting of a cat"),
        ("no_trigger", "portrait of a man in natural light"),
    ]

    lora_name = "filmlut_r32_s1000"
    images = {}

    for key, prompt in test_prompts:
        print(f"  Testing: {key}")
        images[key] = generate_with_lora(prompt, lora_name)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["With Trigger\n(should show style)",
              "Style Change\n(should mix both)",
              "No Trigger\n(should NOT show style)"]

    for i, (key, _) in enumerate(test_prompts):
        axes[i].imshow(images[key])
        axes[i].set_title(titles[i], fontsize=11)
        axes[i].axis('off')

    plt.suptitle("Overfitting Test: Does LoRA Still Follow Prompts?", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "overfitting_test.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'overfitting_test.png'}")


def generate_full_grid():
    """Generate full 4x3 grid of all variants."""
    print("\n=== Generating Full Grid ===")
    prompt = "filmlut, cinematic portrait of a woman in golden hour light"

    ranks = [8, 16, 32, 64]
    steps_list = [250, 500, 1000]

    images = {}
    for rank in ranks:
        for steps in steps_list:
            lora_name = f"filmlut_r{rank}_s{steps}"
            print(f"  Generating r{rank}_s{steps}...")
            images[(rank, steps)] = generate_with_lora(prompt, lora_name)

    # Plot grid
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))

    for i, rank in enumerate(ranks):
        for j, steps in enumerate(steps_list):
            axes[i, j].imshow(images[(rank, steps)])
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(f"{steps} steps", fontsize=12, fontweight='bold')

    # Row labels
    for i, rank in enumerate(ranks):
        axes[i, 0].annotate(f"Rank {rank}", xy=(-0.15, 0.5), xycoords='axes fraction',
                            fontsize=12, fontweight='bold', ha='right', va='center')

    plt.suptitle("LoRA Grid: Rank × Steps", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "full_grid.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'full_grid.png'}")


def main():
    print("="*50)
    print("Generating Workshop Comparison Assets")
    print("="*50)

    generate_rank_comparison()
    generate_steps_comparison()
    generate_overfitting_test()
    generate_full_grid()

    print("\n" + "="*50)
    print("Done! Assets saved to:", OUTPUT_DIR)
    print("="*50)
    print("\nDownload with:")
    print(f"  scp -r runpod:{OUTPUT_DIR} ./assets/")


if __name__ == "__main__":
    main()
