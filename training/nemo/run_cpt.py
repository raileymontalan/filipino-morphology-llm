#!/usr/bin/env python3
"""
Continued Pretraining of Gemma 2 2B using NeMo Framework Container.

This script is designed to run INSIDE the NeMo Framework container.
It uses NeMo 2.0 API for CPT on SEA-PILE Filipino data.

IMPORTANT: For distributed training, pre-convert the HF checkpoint first:
    ./run_in_docker.sh python scripts/convert_hf_to_nemo.py --model google/gemma-3-1b

Then run training with the pre-converted checkpoint:
    ./run_in_docker.sh torchrun --nproc_per_node=8 training/nemo/run_cpt.py \
        --resume-from /workspace/checkpoints/nemo/google_gemma-3-1b \
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


# Apply Gemma3 monkey-patch for rotary_pos_cos_sin parameter
# This fixes: TypeError: Gemma3SelfAttention.forward() got an unexpected keyword argument 'rotary_pos_cos_sin'
try:
    from nemo.collections.llm.gpt.model.gemma3 import Gemma3SelfAttention
    
    print("Applying Gemma3SelfAttention monkey-patch for rotary_pos_cos_sin parameter...")
    
    # Store the original forward method
    _original_gemma3_forward = Gemma3SelfAttention.forward
    
    # Create a wrapper that accepts rotary_pos_cos_sin but ignores it
    def patched_gemma3_forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,  # <-- Added parameter (ignored by Gemma3)
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *args,
        **kwargs
    ):
        # Call the original forward with only the parameters it expects
        return _original_gemma3_forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            key_value_states=key_value_states,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            position_ids=position_ids,
            sequence_len_offset=sequence_len_offset,
            *args,
            **kwargs
        )
    
    # Apply the monkey-patch
    Gemma3SelfAttention.forward = patched_gemma3_forward
    print("✓ Gemma3SelfAttention monkey-patch applied successfully!")
    
except ImportError:
    print("Note: Gemma3SelfAttention not available in this NeMo version (monkey-patch skipped)")
except Exception as e:
    print(f"Warning: Could not apply Gemma3 monkey-patch: {e}")


# Evaluation Strategy: Checkpoints are saved during training.
# Run evaluation separately using scripts/evaluate_checkpoint.py
# This avoids callback complexity and allows parallel evaluation of multiple checkpoints.


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continued pretraining using NeMo container",
        allow_abbrev=False  # Prevent argument abbreviation conflicts with torchrun
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL", "cerebras/Cerebras-GPT-1.3B"),
        help="HuggingFace model ID to use for training",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=os.getenv("RESUME_FROM", ""),
        help="HuggingFace model ID or path to resume from. Empty = use --model",
    )
    
    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        nargs="+",
        default=os.getenv("DATA_PATH", "/workspace/data/processed/seapile-v2").split(":") if os.getenv("DATA_PATH") else ["/workspace/data/processed/seapile-v2"],
        help="Path prefix(es) for preprocessed Megatron binary files (without .bin/.idx extension).",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=int(os.getenv("SEQ_LENGTH", "2048")),
        help="Sequence length for training",
    )
    
    # Training arguments
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.getenv("MAX_STEPS", "100")),
        help="Maximum number of training steps",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=int(os.getenv("GBS", "256")),
        help="Global batch size across all GPUs",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=int(os.getenv("MBS", "2")),
        help="Micro batch size per GPU",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=int(os.getenv("DEVICES", "8")),
        help="Number of GPUs to use",
    )
    
    # Optimizer arguments
    parser.add_argument(
        "--lr",
        type=float,
        default=float(os.getenv("LR", "1e-4")),
        help="Learning rate",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=float(os.getenv("MIN_LR", "1e-5")),
        help="Minimum learning rate for scheduler",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=int(os.getenv("WARMUP_STEPS", "50")),
        help="Number of warmup steps",
    )
    
    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.getenv("CKPT_DIR", "/workspace/checkpoints"),
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=int(os.getenv("CKPT_INTERVAL", "20000")),
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--run-eval-on-checkpoint",
        action="store_true",
        default=os.getenv("RUN_EVAL_ON_CHECKPOINT", "false").lower() == "true",
        help="Run evaluation on benchmarks after saving each checkpoint",
    )
    
    # Logging arguments
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=os.getenv("WANDB_PROJECT", "filipino-morphology-cpt"),
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=os.getenv("WANDB_NAME", "cpt-run"),
        help="WandB run name",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=os.getenv("LOG_DIR", "/workspace/logs"),
        help="Directory for logs",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=int(os.getenv("LOG_EVERY_N_STEPS", "10")),
        help="Log metrics every N steps",
    )
    
    # Validation arguments
    parser.add_argument(
        "--val-check-interval",
        type=int,
        default=int(os.getenv("VAL_CHECK_INTERVAL", "1000")),
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


def get_model_config(model_name, seq_length):
    """Get NeMo model configuration based on model name.
    
    Uses built-in NeMo config classes when available, falls back to GPTConfig for others.
    """
    model_lower = model_name.lower()
    
    # Use NeMo's built-in config classes
    if "gemma-3-1b" in model_lower or "gemma3-1b" in model_lower:
        try:
            from nemo.collections.llm.gpt.model.gemma3 import Gemma3Config1B, Gemma3Model
            config = Gemma3Config1B(seq_length=seq_length)
            return config, Gemma3Model, "google/gemma-3-1b-pt"
        except ImportError:
            print("Warning: Gemma3 model not available in this NeMo version, using GPTConfig")
            # Fallback for older NeMo versions
            config = llm.GPTConfig(
                seq_length=seq_length,
                num_layers=26,
                hidden_size=2048,
                num_attention_heads=8,
                ffn_hidden_size=21504,
                vocab_size=256128,
            )
            return config, llm.GPTModel, "google/gemma-3-1b-pt"
    
    elif "gemma-2-2b" in model_lower or "gemma2-2b" in model_lower:
        config = llm.Gemma2Config2B(seq_length=seq_length)
        return config, llm.Gemma2Model, "google/gemma-2-2b"
    
    elif "qwen3-1.7b" in model_lower or "qwen/qwen3-1.7b" in model_lower:
        try:
            from nemo.collections.llm.gpt.model.qwen3 import Qwen3Config1P7B, Qwen3Model
            config = Qwen3Config1P7B(seq_length=seq_length)
            return config, Qwen3Model, "Qwen/Qwen3-1.7B-Base"
        except ImportError:
            print("Warning: Qwen3 model not available in this NeMo version, using GPTConfig")
            # Fallback for older NeMo versions
            config = llm.GPTConfig(
                seq_length=seq_length,
                num_layers=28,
                hidden_size=2048,
                num_attention_heads=16,
                ffn_hidden_size=11008,
                vocab_size=151936,
            )
            return config, llm.GPTModel, "Qwen/Qwen3-1.7B-Base"
    
    elif "cerebras" in model_lower and "1.3b" in model_lower:
        # No pre-defined config for Cerebras-GPT, use GPTConfig
        config = llm.GPTConfig(
            seq_length=seq_length,
            num_layers=24,
            hidden_size=2048,
            num_attention_heads=16,
            ffn_hidden_size=8192,
            vocab_size=50257,
        )
        return config, llm.GPTModel, model_name
    
    elif "gpt2-xl" in model_lower:
        # No pre-defined config for GPT-2 XL, use GPTConfig
        config = llm.GPTConfig(
            seq_length=seq_length,
            num_layers=48,
            hidden_size=1600,
            num_attention_heads=25,
            ffn_hidden_size=6400,
            vocab_size=50257,
        )
        return config, llm.GPTModel, model_name
    
    else:
        raise ValueError(f"Unsupported model: {model_name}. Add configuration in get_model_config().")


def main():
    args = parse_args()
    
    # Print configuration
    print("\n" + "=" * 80)
    print("Continued Pretraining Configuration")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Resume from: {args.resume_from if args.resume_from else args.model}")
    for arg, value in vars(args).items():
        if arg not in ['model', 'resume_from']:
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
    
    # Get model configuration
    print(f"\n=== Model Configuration ===")
    print(f"Detecting model configuration for: {args.model}")
    
    config, model_class, hf_model_name = get_model_config(args.model, args.seq_length)
    
    print(f"Config class: {config.__class__.__name__}")
    print(f"Model class: {model_class.__name__}")
    print(f"HuggingFace model: {hf_model_name}")
    print(f"Number of layers: {config.num_layers}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Number of attention heads: {config.num_attention_heads}")
    print(f"FFN hidden size: {config.ffn_hidden_size}")
    print(f"Vocab size: {config.vocab_size}")
    
    # Set up WandB logger
    print("\n=== Setting up WandB logger ===")
    wandb_logger = setup_wandb(args)
    
    # Configure the data module with correct tokenizer
    print(f"\n=== Configuring data module ===")
    print(f"Data paths: {len(verified_paths)}")
    print(f"Loading tokenizer: {hf_model_name}")
    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer as NeMoAutoTokenizer
    nemo_tokenizer = NeMoAutoTokenizer(hf_model_name)

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
    
    # Note: Evaluation runs separately after training completes
    # Use scripts/evaluate_checkpoint.py to evaluate saved checkpoints
    print("\nNote: Evaluation runs separately. Use scripts/evaluate_checkpoint.py after training.")
    
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
    print(f"Model: {args.model}")
    if args.resume_from:
        print(f"Resume from: {args.resume_from}")
    print(f"Training steps: {args.max_steps}")
    print(f"Batch size: {args.global_batch_size} (global), {args.micro_batch_size} (micro)")
    print(f"Sequence length: {args.seq_length}")
    print("=" * 80 + "\n")
    
    # Create model using the configuration from get_model_config
    print("Creating model from configuration...")
    
    # Determine resume source
    resume_source = args.resume_from if args.resume_from else args.model
    print(f"Resume source: {resume_source}")
    
    # Create model - NeMo will handle HuggingFace import automatically
    print(f"Creating {model_class.__name__} with config: {config.__class__.__name__}")
    model = model_class(config=config)
    
    # For HuggingFace models, import weights if needed
    resume_config = None
    if "/" in resume_source and not Path(resume_source).exists():
        # It's a HuggingFace model ID - import it
        print(f"\nImporting model weights from HuggingFace: {resume_source}...")
        try:
            restored_model = llm.import_ckpt(
                model=model,
                source=resume_source,
            )
            model = restored_model if restored_model is not None else model
            print("✓ Model imported from HuggingFace")
        except Exception as e:
            print(f"⚠ Warning: Could not import from HuggingFace: {e}")
            print("  Will try to load during training...")
    elif Path(resume_source).exists():
        # It's a local checkpoint
        print(f"Loading from local checkpoint: {resume_source}")
        resume_config = nl.AutoResume(
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
            path=resume_source,
        )
    else:
        print("Training from random initialization")
    
        print("Training from random initialization")
    
    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        optim=optimizer,
        resume=resume_config,
    )
    
    print("\n" + "=" * 80)
    print("✓ Training Run Completed!")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Steps: {args.max_steps}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"Logs saved to: {args.log_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
