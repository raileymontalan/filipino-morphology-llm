#!/usr/bin/env python3
"""
Wrapper script to evaluate NeMo checkpoints using existing evaluation pipeline.

This script:
1. Converts NeMo/Megatron checkpoint to HuggingFace format (if needed)
2. Registers it in configs/models.yaml
3. Calls the existing run_evaluation.py script

Note: For batch evaluation of multiple checkpoints, use:
      bash scripts/convert_and_evaluate_checkpoint.sh

Usage:
    # Convert and evaluate a specific checkpoint
    python scripts/evaluate_checkpoint.py \
        --checkpoint /workspace/checkpoints/.../step=1000.ckpt \
        --base-model cerebras/Cerebras-GPT-1.3B
    
    # Convert and evaluate all checkpoints in a directory
    python scripts/evaluate_checkpoint.py \
        --checkpoint-dir /workspace/checkpoints/cerebras-Cerebras-GPT-1.3B/vanilla/ \
        --base-model cerebras/Cerebras-GPT-1.3B \
        --latest-only
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import glob
import re
import yaml


def find_checkpoints(checkpoint_dir, latest_only=False):
    """Find all checkpoint files in a directory."""
    ckpt_pattern = os.path.join(checkpoint_dir, "*.ckpt")
    checkpoints = glob.glob(ckpt_pattern)
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return []
    
    # Sort by step number
    def extract_step(ckpt_path):
        match = re.search(r'step=(\d+)', os.path.basename(ckpt_path))
        return int(match.group(1)) if match else 0
    
    checkpoints.sort(key=extract_step)
    
    if latest_only:
        return [checkpoints[-1]]
    
    return checkpoints


def convert_to_hf(checkpoint_path, base_model, output_dir):
    """Convert NeMo checkpoint to HuggingFace format."""
    print(f"\nConverting checkpoint to HuggingFace format...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Base model: {base_model}")
    print(f"  Output: {output_dir}")
    
    cmd = [
        "python", "scripts/convert_megatron_to_hf.py",
        "--megatron-checkpoint", checkpoint_path,
        "--base-model", base_model,
        "--output", output_dir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ Conversion failed:")
        print(result.stderr)
        return False
    
    print(f"✓ Converted successfully")
    return True


def add_to_model_config(model_name, model_path, config_path="configs/models.yaml"):
    """Add converted model to models.yaml config."""
    
    # Load existing config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
    
    if 'models' not in config:
        config['models'] = {}
    
    # Check if already exists
    if model_name in config['models']:
        print(f"  Model '{model_name}' already in config")
        return
    
    # Add new model
    config['models'][model_name] = {
        'path': model_path,
        'type': 'pt'
    }
    
    # Save config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"  ✓ Added '{model_name}' to {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate NeMo checkpoints (converts to HF first)")
    
    # Checkpoint source
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="Path to a specific NeMo .ckpt file")
    group.add_argument("--checkpoint-dir", type=str, help="Directory containing NeMo checkpoints")
    
    # Required for conversion
    parser.add_argument("--base-model", type=str, required=True,
                       help="Base HuggingFace model (e.g., cerebras/Cerebras-GPT-1.3B)")
    
    # Options
    parser.add_argument("--latest-only", action="store_true",
                       help="Only evaluate latest checkpoint (use with --checkpoint-dir)")
    parser.add_argument("--model-name-prefix", type=str, default="checkpoint",
                       help="Prefix for model names in config (default: checkpoint)")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                       default=["pacute-affixation-mcq", "pacute-composition-mcq", 
                               "hierarchical-mcq", "langgame-mcq"],
                       help="Benchmarks to run")
    parser.add_argument("--max-samples", type=int, help="Max samples per benchmark")
    parser.add_argument("--output-dir", type=str, default="results/benchmark_evaluation",
                       help="Output directory for results")
    parser.add_argument("--skip-conversion", action="store_true",
                       help="Skip conversion (assumes HF model already exists)")
    
    args = parser.parse_args()
    
    # Find checkpoints
    if args.checkpoint:
        checkpoints = [args.checkpoint]
    else:
        checkpoints = find_checkpoints(args.checkpoint_dir, args.latest_only)
    
    if not checkpoints:
        print("No checkpoints found")
        return 1
    
    print(f"Found {len(checkpoints)} checkpoint(s) to evaluate")
    
    converted_models = []
    
    # Convert each checkpoint
    for ckpt_path in checkpoints:
        ckpt_name = os.path.basename(ckpt_path)
        step_match = re.search(r'step=(\d+)', ckpt_name)
        step = step_match.group(1) if step_match else "unknown"
        
        model_name = f"{args.model_name_prefix}-step{step}"
        hf_output_dir = f"checkpoints/hf/{model_name}"
        
        print(f"\n{'='*80}")
        print(f"Processing: {ckpt_name}")
        print(f"{'='*80}")
        
        # Convert if needed
        if not args.skip_conversion:
            if os.path.exists(hf_output_dir) and os.path.exists(f"{hf_output_dir}/config.json"):
                print(f"⚠ HF model already exists: {hf_output_dir}")
                print(f"  Skipping conversion...")
            else:
                if not convert_to_hf(ckpt_path, args.base_model, hf_output_dir):
                    print(f"✗ Failed to convert {ckpt_name}, skipping...")
                    continue
        
        # Add to config
        add_to_model_config(model_name, hf_output_dir)
        converted_models.append(model_name)
    
    if not converted_models:
        print("\n✗ No checkpoints converted successfully")
        return 1
    
    # Run evaluation using existing pipeline
    print(f"\n{'='*80}")
    print("Running Evaluation")
    print(f"{'='*80}")
    print(f"Models: {', '.join(converted_models)}")
    print(f"Benchmarks: {', '.join(args.benchmarks)}")
    
    eval_cmd = [
        "python", "scripts/run_evaluation.py",
        "--models"] + converted_models + [
        "--benchmarks"] + args.benchmarks + [
        "--output-dir", args.output_dir
    ]
    
    if args.max_samples:
        eval_cmd += ["--max-samples", str(args.max_samples)]
    
    print(f"\nCommand: {' '.join(eval_cmd)}\n")
    
    result = subprocess.run(eval_cmd)
    
    if result.returncode == 0:
        print("\n✓ Evaluation completed successfully")
        return 0
    else:
        print("\n✗ Evaluation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
