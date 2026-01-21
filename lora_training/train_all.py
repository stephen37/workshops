#!/usr/bin/env python3
"""
Train all LoRA variants for workshop demo.
Run this from the lora_training directory on Runpod.

Usage:
    cd /workspace/lora_training
    source /workspace/ai-toolkit/venv/bin/activate
    python train_all.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Config directory
CONFIG_DIR = Path(__file__).parent / "configs"

# Training order: start with smallest (fastest) to verify everything works
TRAINING_ORDER = [
    # Rank 8 (smallest, fastest)
    "filmlut_r8_s250",
    "filmlut_r8_s500",
    "filmlut_r8_s1000",
    # Rank 16
    "filmlut_r16_s250",
    "filmlut_r16_s500",
    "filmlut_r16_s1000",
    # Rank 32
    "filmlut_r32_s250",
    "filmlut_r32_s500",
    "filmlut_r32_s1000",
    # Rank 64 (largest)
    "filmlut_r64_s250",
    "filmlut_r64_s500",
    "filmlut_r64_s1000",
]


def run_training(config_name: str) -> bool:
    """Run training for a single config. Returns True if successful."""
    config_path = CONFIG_DIR / f"{config_name}.yaml"

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        return False

    print(f"\n{'='*60}")
    print(f"Starting: {config_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Run AI Toolkit training
    cmd = [
        sys.executable,
        "/workspace/ai-toolkit/run.py",
        str(config_path)
    ]

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\nCompleted: {config_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nFAILED: {config_name} (exit code {e.returncode})")
        return False
    except KeyboardInterrupt:
        print(f"\nInterrupted during: {config_name}")
        raise


def main():
    print("="*60)
    print("LoRA Training - All Variants")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total configs: {len(TRAINING_ORDER)}")
    print("="*60)

    results = []

    for i, config_name in enumerate(TRAINING_ORDER, 1):
        print(f"\n[{i}/{len(TRAINING_ORDER)}] {config_name}")
        success = run_training(config_name)
        results.append((config_name, success))

        if not success:
            print(f"\nTraining failed for {config_name}.")
            response = input("Continue with remaining configs? [y/N]: ").strip().lower()
            if response != 'y':
                break

    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    successful = [r[0] for r in results if r[1]]
    failed = [r[0] for r in results if not r[1]]

    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    for name in successful:
        print(f"  - {name}")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for name in failed:
            print(f"  - {name}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutputs saved to: /workspace/lora_outputs/")


if __name__ == "__main__":
    main()
