#!/usr/bin/env python3
"""
Generate comparison images using AI Toolkit's training samples.
No inference needed - just assembles the samples that were already generated.

Usage:
    cd /workspace/workshops/lora_training
    python generate_comparisons.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import glob

# Paths
LORA_DIR = Path("/workspace/lora_outputs")
OUTPUT_DIR = Path("/workspace/comparison_assets")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_final_sample(lora_name, prompt_idx=0):
    """Get the last sample image from a training run."""
    sample_dir = LORA_DIR / lora_name / "samples"
    if not sample_dir.exists():
        print(f"    Warning: No samples found for {lora_name}")
        return None

    # Get all sample images (jpg or png)
    samples = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
    if not samples:
        print(f"    Warning: No image files in {sample_dir}")
        return None

    # Parse step number from filename like "1768998807483__000000500_0.jpg"
    # Format: {timestamp}__{step}_{prompt_idx}.jpg
    def get_step(path):
        try:
            parts = path.stem.split('__')
            if len(parts) >= 2:
                step_part = parts[1].split('_')[0]
                return int(step_part)
        except:
            pass
        return 0

    # Filter to just one prompt index and sort by step
    filtered = [s for s in samples if f"_{prompt_idx}.jpg" in s.name or f"_{prompt_idx}.png" in s.name]
    if not filtered:
        filtered = samples  # fallback to all

    filtered.sort(key=get_step)

    # Return the last (highest step) sample
    return Image.open(filtered[-1])


def generate_rank_comparison():
    """Compare different ranks at 500 steps using training samples."""
    print("\n=== Generating Rank Comparison ===")

    ranks = [8, 16, 32, 64]
    images = {}

    for rank in ranks:
        lora_name = f"filmlut_r{rank}_s500"
        print(f"  Loading samples for rank {rank}...")
        images[f"r{rank}"] = get_final_sample(lora_name)

    # Filter out None values
    valid_ranks = [r for r in ranks if images.get(f"r{r}") is not None]

    if not valid_ranks:
        print("  No valid samples found!")
        return

    # Plot
    fig, axes = plt.subplots(1, len(valid_ranks), figsize=(5*len(valid_ranks), 5))
    if len(valid_ranks) == 1:
        axes = [axes]

    for i, rank in enumerate(valid_ranks):
        axes[i].imshow(images[f"r{rank}"])
        axes[i].set_title(f"Rank {rank}\n(500 steps)", fontsize=12)
        axes[i].axis('off')

    plt.suptitle("Effect of Rank on Style Transfer", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rank_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'rank_comparison.png'}")


def generate_steps_comparison():
    """Compare different step counts at rank 32."""
    print("\n=== Generating Steps Comparison ===")

    steps_list = [250, 500, 1000]
    images = {}

    for steps in steps_list:
        lora_name = f"filmlut_r32_s{steps}"
        print(f"  Loading samples for {steps} steps...")
        images[f"s{steps}"] = get_final_sample(lora_name)

    valid_steps = [s for s in steps_list if images.get(f"s{s}") is not None]

    if not valid_steps:
        print("  No valid samples found!")
        return

    fig, axes = plt.subplots(1, len(valid_steps), figsize=(5*len(valid_steps), 5))
    if len(valid_steps) == 1:
        axes = [axes]

    for i, steps in enumerate(valid_steps):
        axes[i].imshow(images[f"s{steps}"])
        axes[i].set_title(f"{steps} steps\n(rank 32)", fontsize=12)
        axes[i].axis('off')

    plt.suptitle("Effect of Training Steps", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "steps_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'steps_comparison.png'}")


def generate_full_grid(prompt_idx=0):
    """Generate full grid of all variants for a specific prompt index."""
    print(f"\n=== Generating Full Grid (prompt {prompt_idx}) ===")

    ranks = [8, 16, 32, 64]
    steps_list = [250, 500, 1000]

    images = {}
    for rank in ranks:
        for steps in steps_list:
            lora_name = f"filmlut_r{rank}_s{steps}"
            print(f"  Loading r{rank}_s{steps}...")
            images[(rank, steps)] = get_final_sample(lora_name, prompt_idx=prompt_idx)

    # Check what we have
    valid_count = sum(1 for img in images.values() if img is not None)
    print(f"  Found {valid_count}/12 valid samples")

    if valid_count == 0:
        print("  No valid samples found!")
        return

    # Plot grid
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))

    for i, rank in enumerate(ranks):
        for j, steps in enumerate(steps_list):
            img = images.get((rank, steps))
            if img:
                axes[i, j].imshow(img)
            else:
                axes[i, j].text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=20)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(f"{steps} steps", fontsize=12, fontweight='bold')

    # Row labels
    for i, rank in enumerate(ranks):
        axes[i, 0].annotate(f"Rank {rank}", xy=(-0.15, 0.5), xycoords='axes fraction',
                            fontsize=12, fontweight='bold', ha='right', va='center')

    plt.suptitle(f"LoRA Grid: Rank × Steps (Prompt {prompt_idx})", fontsize=16, fontweight='bold')
    plt.tight_layout()
    filename = f"full_grid_prompt{prompt_idx}.png"
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / filename}")


def generate_training_progression():
    """Show how training progresses over steps for one config."""
    print("\n=== Generating Training Progression ===")

    # Pick r32_s1000 as it has the most samples
    lora_name = "filmlut_r32_s1000"
    sample_dir = LORA_DIR / lora_name / "samples"

    if not sample_dir.exists():
        print(f"  No samples found for {lora_name}")
        return

    samples = sorted(sample_dir.glob("*.png"))
    if len(samples) < 3:
        print("  Not enough samples for progression")
        return

    # Pick evenly spaced samples
    n_samples = min(5, len(samples))
    indices = [int(i * (len(samples)-1) / (n_samples-1)) for i in range(n_samples)]
    selected = [samples[i] for i in indices]

    fig, axes = plt.subplots(1, n_samples, figsize=(4*n_samples, 4))

    for i, sample_path in enumerate(selected):
        img = Image.open(sample_path)
        axes[i].imshow(img)
        # Extract step from filename
        step = sample_path.stem.split('_')[-1] if '_' in sample_path.stem else str(i)
        axes[i].set_title(f"Step {step}", fontsize=11)
        axes[i].axis('off')

    plt.suptitle("Training Progression (Rank 32)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_progression.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'training_progression.png'}")


def main():
    print("="*50)
    print("Generating Workshop Comparison Assets")
    print("Using training samples (no inference needed)")
    print("="*50)

    # List what we have
    print("\nAvailable LoRA outputs:")
    for d in sorted(LORA_DIR.iterdir()):
        if d.is_dir() and (d / "samples").exists():
            samples = list((d / "samples").glob("*.jpg")) + list((d / "samples").glob("*.png"))
            print(f"  {d.name}: {len(samples)} samples")

    generate_rank_comparison()
    generate_steps_comparison()

    # Generate grids for all prompt indices
    for prompt_idx in range(4):  # 0, 1, 2, 3
        generate_full_grid(prompt_idx=prompt_idx)

    generate_training_progression()

    print("\n" + "="*50)
    print("Done! Assets saved to:", OUTPUT_DIR)
    print("="*50)
    print("\nFiles generated:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
