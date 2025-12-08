#!/usr/bin/env python3
"""
Continued Pretraining of Gemma 3 1B using NeMo Framework Container.

This script is designed to run INSIDE the NeMo Framework container.
It uses NeMo 2.0 API for a 100-step CPT run on SEA-PILE data.

Container setup (choose one):
    # Singularity/Apptainer:
    source .env
    bash setup_singularity.sh /scratch/$USER/container_cache
    
    # Enroot:
    source .env
    bash setup_enroot.sh

Run inside container (choose one):
    # Singularity/Apptainer:
    ./run_in_singularity.sh python scripts/run_cpt_gemma3_1b_container.py --max-steps 100
    
    # Enroot:
    ./run_in_enroot.sh python scripts/run_cpt_gemma3_1b_container.py --max-steps 100

Submit via SLURM (must run through container):
    # The SLURM job scripts use the container runtime internally
    qsub jobs/submit_cpt_singularity.sh  # Uses Singularity/Apptainer
    qsub jobs/submit_cpt_enroot.sh       # Uses Enroot
"""

import argparse
import os
import sys
from pathlib import Path

# Verify we're in the container environment
try:
    import nemo
    import nemo.collections.llm as llm
    from nemo import lightning as nl
    from nemo.collections.llm import PreTrainingDataModule
    from megatron.core.optimizer import OptimizerConfig
    print(f"✓ Running in NeMo Framework {nemo.__version__}")
except ImportError as e:
    print(f"✗ Error: {e}")
    print("\nThis script must run inside the NeMo Framework container.")
    print("\nFor Singularity/Apptainer:")
    print("  ./run_in_singularity.sh python scripts/run_cpt_gemma3_1b_container.py")
    print("\nFor Enroot:")
    print("  ./run_in_enroot.sh python scripts/run_cpt_gemma3_1b_container.py")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continued pretraining of Gemma 3 1B in NeMo container"
    )
    
    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        nargs="+",
        default=["/workspace/data/processed/seapile-v2"],
        help="Path prefix(es) for preprocessed Megatron binary files (without .bin/.idx extension). Can specify multiple paths for parallel chunks.",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=2048,
        help="Sequence length for training",
    )
    
    # Training arguments
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum number of training steps (default: 100 for quick test)",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=256,
        help="Global batch size across all GPUs",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=2,
        help="Micro batch size per GPU",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=8,
        help="Number of GPUs to use",
    )
    
    # Optimizer arguments
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        help="Minimum learning rate for scheduler",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
        help="Number of warmup steps (10% of total for 100-step run)",
    )
    
    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/logs/checkpoints/gemma3-1b-seapile-100steps",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="HuggingFace model ID to resume from (optional, trains from scratch if not provided)",
    )
    
    # Logging arguments
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="gemma3-seapile-cpt",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default="gemma3-1b-seapile-100steps",
        help="WandB run name",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/logs/wandb",
        help="Directory for logs",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=10,
        help="Log metrics every N steps",
    )
    
    # Validation arguments
    parser.add_argument(
        "--val-check-interval",
        type=int,
        default=50,
        help="Run validation every N steps",
    )
    
    return parser.parse_args()


def setup_wandb(args):
    """Set up Weights & Biases logging."""
    from pytorch_lightning.loggers import WandbLogger
    
    # Check for API key
    if not os.getenv("WANDB_API_KEY"):
        print("⚠  WARNING: WANDB_API_KEY not found in environment.")
        print("   Set it with: export WANDB_API_KEY='your-key-here'")
        print("   Disabling WandB logging")
        os.environ["WANDB_MODE"] = "disabled"
    
    return WandbLogger(
        project=args.wandb_project,
        name=args.wandb_name,
        save_dir=args.log_dir,
        log_model=True,
    )


def main():
    args = parse_args()
    
    # Print configuration
    print("\n" + "=" * 80)
    print("Continued Pretraining Configuration (100-Step Run)")
    print("=" * 80)
    for arg, value in vars(args).items():
        print(f"{arg:25s}: {value}")
    print("=" * 80 + "\n")
    
    # Verify preprocessed data exists (Megatron binary format)
    # The data path should be a prefix like "data/processed/seapile-v2"
    # which will have corresponding files: seapile-v2.bin and .idx
    # Support multiple paths for parallel preprocessing chunks
    data_prefixes = args.data_path if isinstance(args.data_path, list) else [args.data_path]
    
    print(f"✓ Verifying {len(data_prefixes)} data path(s)...")
    total_size_gb = 0
    verified_paths = []
    
    for data_prefix in data_prefixes:
        bin_path = Path(f"{data_prefix}.bin")
        idx_path = Path(f"{data_prefix}.idx")
        
        if not bin_path.exists() or not idx_path.exists():
            print(f"✗ Error: Preprocessed Megatron binary files not found for {data_prefix}")
            print(f"  Expected: {bin_path}")
            print(f"  Expected: {idx_path}")
            print()
            print("Please preprocess your data first:")
            print("  # Single file:")
            print("  python scripts/preprocess_data.py \\")
            print(f"    --input data/corpora/seapile-v2.jsonl \\")
            print(f"    --output-prefix {data_prefix} \\")
            print(f"    --tokenizer-model google/gemma-3-1b-pt")
            print()
            print("  # Or parallel chunks:")
            print("  qsub -J 1-N jobs/preprocess_data_parallel.pbs")
            sys.exit(1)
        
        size_gb = bin_path.stat().st_size / 1e9
        total_size_gb += size_gb
        verified_paths.append(data_prefix)
        print(f"  ✓ {data_prefix}: {size_gb:.2f} GB")
    
    print(f"✓ Total data size: {total_size_gb:.2f} GB across {len(verified_paths)} file(s)")
    
    # Set up WandB logger
    print("Setting up WandB logger...")
    wandb_logger = setup_wandb(args)
    
    # Configure the data module
    print(f"Configuring data module with {len(verified_paths)} data path(s)...")
    data = PreTrainingDataModule(
        paths=verified_paths,  # Use the preprocessed data prefix(es) (without .bin/.idx)
        seq_length=args.seq_length,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        num_workers=4,
    )
    
    # Set up optimizer
    print(f"Configuring optimizer (lr={args.lr})...")
    optimizer = nl.MegatronOptimizerModule(
        config=OptimizerConfig(
            lr=args.lr,
            optimizer="adam",
            use_distributed_optimizer=True,
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.95,
        ),
        lr_scheduler=nl.lr_scheduler.CosineAnnealingScheduler(
            warmup_steps=args.warmup_steps,
            constant_steps=0,
            min_lr=args.min_lr,
        ),
    )
    
    # Checkpoint configuration
    print(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    checkpoint_callback = nl.ModelCheckpoint(
        save_top_k=2,
        monitor="val_loss",
        save_last=True,
        every_n_train_steps=args.checkpoint_interval,
        dirpath=args.checkpoint_dir,
    )
    
    # Training configuration
    print(f"Configuring trainer with {args.devices} GPUs...")
    trainer = nl.Trainer(
        devices=args.devices,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=nl.MegatronStrategy(
            tensor_model_parallel_size=1,  # No model parallelism for 1B model
            pipeline_model_parallel_size=1,
            ddp="megatron",
        ),
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        limit_val_batches=10,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    
    # Configure model
    model = llm.Gemma3Model(
        config=llm.Gemma3Config1B(seq_length=args.seq_length)
    )
    
    # Train the model
    print("\n" + "=" * 80)
    print("Starting Continued Pretraining Run")
    print("=" * 80)
    if args.resume_from:
        print(f"Resume from: {args.resume_from}")
    else:
        print("Training from scratch (no checkpoint)")
    print(f"Training steps: {args.max_steps}")
    print(f"Batch size: {args.global_batch_size} (global), {args.micro_batch_size} (micro)")
    print(f"Sequence length: {args.seq_length}")
    print("=" * 80 + "\n")
    
    # Configure resume behavior
    resume_config = None
    if args.resume_from:
        resume_config = nl.AutoResume(
            resume_from_path=f"hf://{args.resume_from}",
            resume_if_exists=True,
        )
    
    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        optim=optimizer,
        resume=resume_config,
    )
    
    print("\n" + "=" * 80)
    print("✓ 100-Step Training Run Completed!")
    print("=" * 80)
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"Logs saved to: {args.log_dir}")
    print("\nTo continue training for more steps, run:")
    print(f"  # Singularity/Apptainer:")
    print(f"  ./run_in_singularity.sh python {__file__} \\")
    print(f"    --max-steps 1000 \\")
    print(f"    --checkpoint-dir {args.checkpoint_dir}")
    print(f"")
    print(f"  # Enroot:")
    print(f"  ./run_in_enroot.sh python {__file__} \\")
    print(f"    --max-steps 1000 \\")
    print(f"    --checkpoint-dir {args.checkpoint_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
