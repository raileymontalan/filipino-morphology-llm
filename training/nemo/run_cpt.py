#!/usr/bin/env python3
"""
Continued Pretraining of Gemma 2 2B using NeMo Framework Container.

This script is designed to run INSIDE the NeMo Framework container.
It uses NeMo 2.0 API for CPT on SEA-PILE Filipino data.

IMPORTANT: For distributed training, pre-convert the HF checkpoint first:
    ./run_in_docker.sh python scripts/convert_hf_to_nemo.py --model google/gemma-2-2b

Then run training with the pre-converted checkpoint:
    ./run_in_docker.sh torchrun --nproc_per_node=8 training/nemo/run_cpt.py \
        --resume-from /workspace/checkpoints/nemo/google_gemma-2-2b \
        --data-path /workspace/data/processed/vanilla/chunk_001_text_document ...

Usage:
    ./run_in_docker.sh torchrun --nproc_per_node=8 training/nemo/run_cpt.py --max-steps 100
"""

import argparse
import os
import sys
from pathlib import Path

# Prevent cache conflicts in distributed mode
os.environ.setdefault("TORCH_HOME", "/workspace/.cache/torch")
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/.cache/huggingface")

# Verify we're in the container environment
try:
    import nemo
    import nemo.collections.llm as llm
    from nemo import lightning as nl
    from nemo.collections.llm import PreTrainingDataModule
    from megatron.core.optimizer import OptimizerConfig
    from megatron.core.distributed import DistributedDataParallelConfig
    print(f"✓ Running in NeMo Framework {nemo.__version__}")
except ImportError as e:
    print(f"✗ Error: {e}")
    print("\nThis script must run inside the NeMo Framework container.")
    print("\nFor Singularity/Apptainer:")
    print("  ./run_in_singularity.sh python scripts/run_cpt_gemma3_1b_container.py")
    print("\nFor Enroot:")
    print("  ./run_in_enroot.sh python scripts/run_cpt_gemma3_1b_container.py")
    sys.exit(1)


# Note: Gemma3 monkey-patch removed since NeMo 2.0.0rc1 doesn't support Gemma3
# Use NeMo 2.1+ container for Gemma3 support


class EvaluationCallback(nl.pytorch.callbacks.Callback):
    """Callback to run evaluation after each checkpoint save."""
    
    def __init__(self, eval_script_path="/workspace/scripts/run_evaluation.py", benchmarks=None, eval_mode="mcq"):
        super().__init__()
        self.eval_script_path = eval_script_path
        # Use all MCQ benchmarks by default for comprehensive evaluation
        self.benchmarks = benchmarks or [
            "pacute-affixation-mcq",
            "pacute-composition-mcq", 
            "pacute-manipulation-mcq",
            "pacute-syllabification-mcq",
            "hierarchical-mcq",
            "langgame-mcq",
            "multi-digit-addition-mcq",
        ]
        self.eval_mode = eval_mode
        
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Run evaluation after checkpoint is saved."""
        if not os.getenv("RUN_EVAL_ON_CHECKPOINT", "false").lower() == "true":
            return
            
        # Only run on rank 0 to avoid duplicate evaluations
        if trainer.global_rank != 0:
            return
            
        print("\n" + "=" * 80)
        print("Running Evaluation After Checkpoint Save")
        print("=" * 80)
        
        # Get checkpoint path
        ckpt_dir = trainer.checkpoint_callback.dirpath
        current_step = trainer.global_step
        
        # Construct checkpoint path - find most recent checkpoint
        import glob
        ckpt_pattern = f"{ckpt_dir}/step={current_step}-*.ckpt"
        ckpt_files = glob.glob(ckpt_pattern)
        if not ckpt_files:
            print(f"⚠ Warning: No checkpoint found matching {ckpt_pattern}")
            return
        
        ckpt_path = ckpt_files[0]
        print(f"Checkpoint: {ckpt_path}")
        
        # Prepare output path for evaluation results
        eval_output = f"{ckpt_dir}/eval_step_{current_step}.json"
        
        # Run evaluation script
        import subprocess
        cmd = [
            "python",
            self.eval_script_path,
            "--model", ckpt_path,
            "--benchmarks", *self.benchmarks,
            "--eval-mode", self.eval_mode,
            "--output", eval_output,
        ]
        
        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            if result.returncode == 0:
                print(f"✓ Evaluation complete: {eval_output}")
                print(result.stdout)
                
                # Log results to WandB if available
                if trainer.logger:
                    import json
                    try:
                        with open(eval_output, 'r') as f:
                            eval_results = json.load(f)
                        # Log to WandB with step
                        for benchmark, metrics in eval_results.items():
                            for metric_name, value in metrics.items():
                                trainer.logger.experiment.log({
                                    f"eval/{benchmark}/{metric_name}": value,
                                    "step": current_step
                                })
                    except Exception as e:
                        print(f"⚠ Warning: Could not log eval results to WandB: {e}")
            else:
                print(f"✗ Evaluation failed with return code {result.returncode}")
                print(result.stderr)
        except subprocess.TimeoutExpired:
            print("✗ Evaluation timed out after 1 hour")
        except Exception as e:
            print(f"✗ Evaluation error: {e}")
        
        print("=" * 80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continued pretraining of Gemma 2 2B in NeMo container",
        allow_abbrev=False  # Prevent argument abbreviation conflicts with torchrun
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
        default=20000,
        help="Save checkpoint every N steps (default: 20000 for 100k training)",
    )
    parser.add_argument(
        "--run-eval-on-checkpoint",
        action="store_true",
        default=False,
        help="Run evaluation on PACUTE benchmark after saving each checkpoint",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Path to pre-converted NeMo checkpoint (use scripts/convert_hf_to_nemo.py first). Empty = train from scratch.",
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
        default=1000,  # Match checkpoint interval to avoid frequent saves
        help="Run validation every N steps",
    )
    
    args, unknown = parser.parse_known_args()
    if unknown:
        # Filter out empty strings from unknown args (torchrun artifact)
        unknown_filtered = [u for u in unknown if u.strip()]
        if unknown_filtered:
            print(f"Note: Ignoring unknown arguments: {unknown_filtered}")

    # Filter out empty strings from data_path (torchrun artifact)
    if args.data_path:
        args.data_path = [p for p in args.data_path if p.strip()]

    return args


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
    
    # Configure the data module with Gemma tokenizer
    # The data was preprocessed with Gemma tokenizer, so we must use the same tokenizer
    print(f"Configuring data module with {len(verified_paths)} data path(s)...")
    print("Loading Gemma 2 tokenizer via NeMo wrapper...")
    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer as NeMoAutoTokenizer
    nemo_tokenizer = NeMoAutoTokenizer("google/gemma-2-2b")

    data = PreTrainingDataModule(
        paths=verified_paths,  # Use the preprocessed data prefix(es) (without .bin/.idx)
        seq_length=args.seq_length,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        tokenizer=nemo_tokenizer,  # Use NeMo-wrapped Gemma tokenizer (matches preprocessing)
        num_workers=4,
        # Better handling of sequences and document boundaries
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,  # Mask loss at end-of-document tokens
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
            adam_eps=1e-5,  # Better for bf16 training (was 1e-8 by default)
            bf16=True,
            fp16=False,
            clip_grad=1.0,  # Explicit gradient clipping for stability
        ),
        lr_scheduler=nl.lr_scheduler.CosineAnnealingScheduler(
            warmup_steps=args.warmup_steps,
            constant_steps=0,
            min_lr=args.min_lr,
        ),
    )
    
    # Checkpoint configuration
    # NOTE: Checkpoints are 30-35GB each with distributed optimizer state
    # Save every 20k steps - 5 checkpoints total for 100k training
    print(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    checkpoint_callback = nl.ModelCheckpoint(
        save_top_k=-1,  # Keep all checkpoints (not just best)
        every_n_train_steps=args.checkpoint_interval,
        save_on_train_epoch_end=False,  # Save during training steps, not epoch end
        dirpath=args.checkpoint_dir,
        # Note: No 'monitor' parameter - we save every N steps regardless of metric
        # Simple filename without val_loss dependency
        filename="step={step}-consumed={consumed_samples}",
        verbose=True,  # Print when saving checkpoints
    )
    
    # Setup callbacks list
    callbacks_list = [checkpoint_callback]
    
    # Add evaluation callback if requested
    if args.run_eval_on_checkpoint:
        print("Evaluation will run after each checkpoint save")
        print("Benchmarks: All MCQ benchmarks (PACUTE, Hierarchical, LangGame, Multi-digit Addition)")
        eval_callback = EvaluationCallback(
            eval_script_path="/workspace/scripts/run_evaluation.py",
            benchmarks=None,  # Use default (all MCQ benchmarks)
            eval_mode="mcq"  # MCQ is faster than generative
        )
        callbacks_list.append(eval_callback)
    
    # Training configuration
    print(f"Configuring trainer with {args.devices} GPUs...")
    trainer = nl.Trainer(
        devices=args.devices,
        max_steps=args.max_steps,
        accelerator="gpu",
        enable_checkpointing=True,  # Explicitly enable checkpointing
        strategy=nl.MegatronStrategy(
            tensor_model_parallel_size=1,  # No model parallelism for 1B model
            pipeline_model_parallel_size=1,
            ddp=DistributedDataParallelConfig(
                # DDP settings for training stability (NeMo 2.3.0rc0 dev)
                check_for_nan_in_grad=True,
                grad_reduce_in_fp32=True,  # Reduce gradients in FP32 for stability
                overlap_grad_reduce=True,  # Overlap gradient reduction with computation
                overlap_param_gather=True,  # Overlap parameter gathering
                average_in_collective=True,  # Average gradients in collective ops
            ),
            gradient_as_bucket_view=True,  # Memory optimization
            ckpt_async_save=True,  # Async checkpoint saving
            ckpt_parallel_save=True,  # Parallel checkpoint saving
            ckpt_parallel_load=True,  # Parallel checkpoint loading
        ),
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        limit_val_batches=10,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        logger=wandb_logger,
        callbacks=callbacks_list,
    )
    
    # Train the model
    print("\n" + "=" * 80)
    print("Starting Continued Pretraining Run")
    print("=" * 80)
    if args.resume_from:
        print(f"Importing weights from HuggingFace: {args.resume_from}")
    else:
        print("Training from scratch (no checkpoint)")
    print(f"Training steps: {args.max_steps}")
    print(f"Batch size: {args.global_batch_size} (global), {args.micro_batch_size} (micro)")
    print(f"Sequence length: {args.seq_length}")
    print("=" * 80 + "\n")
    
    # Configure model (Gemma 2 2B - matches data tokenizer)
    print("Creating model from configuration...")

    # Set up checkpoint resumption for continued pretraining
    # IMPORTANT: Use pre-converted NeMo checkpoint, NOT HuggingFace model ID
    # Convert first with: ./run_in_docker.sh python scripts/convert_hf_to_nemo.py --model google/gemma-2-2b
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            print(f"✓ Loading pretrained model weights from: {args.resume_from}")
            print("  (Note: Optimizer state will be initialized fresh for continued pretraining)")
            # Load model with pretrained weights from the checkpoint
            model = llm.Gemma2Model(
                config=llm.Gemma2Config2B(
                    seq_length=args.seq_length,
                    vocab_size=256128,  # Gemma tokenizer vocab size
                ),
                # Load weights from checkpoint but don't restore training state
                # This is the correct approach for continued pretraining
            )
            # Don't use AutoResume - it tries to load optimizer state which causes key mismatches
            # Instead, we'll use llm.load() after creating the trainer
            resume_config = None
            use_pretrained_path = str(resume_path)
        else:
            print(f"✗ Error: Checkpoint path does not exist: {args.resume_from}")
            print("\nTo convert a HuggingFace checkpoint, run:")
            print("  ./run_in_docker.sh python scripts/convert_hf_to_nemo.py --model google/gemma-2-2b")
            sys.exit(1)
    else:
        print("Training from scratch (no checkpoint specified)")
        model = llm.Gemma2Model(
            config=llm.Gemma2Config2B(
                seq_length=args.seq_length,
                vocab_size=256128,  # Gemma tokenizer vocab size
            )
        )
        resume_config = None
        use_pretrained_path = None
    
    # Load pretrained weights if specified (for continued pretraining)
    if use_pretrained_path:
        print(f"\nLoading pretrained weights from: {use_pretrained_path}")
        print("(Optimizer and scheduler state will be initialized from scratch)")
        # Use NeMo's checkpoint format: weights/common.pt
        import torch
        weights_path = f"{use_pretrained_path}/weights/common.pt"
        if Path(weights_path).exists():
            checkpoint = torch.load(weights_path, map_location="cpu")
            # NeMo checkpoint format may have different keys
            if isinstance(checkpoint, dict):
                # Load the state dict - NeMo uses different keys depending on version
                model_state = checkpoint.get("state_dict", checkpoint)
                model.load_state_dict(model_state, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print("✓ Pretrained weights loaded successfully")
        else:
            print(f"⚠ Warning: Could not find weights at {weights_path}, training from scratch")

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
