#!/usr/bin/env python3
"""
Upload trained models to HuggingFace Hub.

Usage:
    python scripts/upload_to_huggingface.py --model vanilla-step4999
    python scripts/upload_to_huggingface.py --model stochastok-step4999
    python scripts/upload_to_huggingface.py --model patok-step4999
    python scripts/upload_to_huggingface.py --all  # Upload all models
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo


MODEL_CONFIGS = {
    "vanilla-step4999": {
        "local_path": "checkpoints/hf/gemma2-2b-vanilla-step4999",
        "repo_id": "davidafrica/gemma-2-2b-filipino-vanilla-cpt",
        "description": "Gemma 2 2B continued pretrained on SEA-PILE Filipino with vanilla BPE tokenization",
        "tokenization": "Standard BPE tokenization (baseline)",
        "expansion_rate": "0% (baseline)",
        "val_loss": "2.99",
    },
    "stochastok-step4999": {
        "local_path": "checkpoints/hf/gemma2-2b-stochastok-step4999",
        "repo_id": "davidafrica/gemma-2-2b-filipino-stochastok-cpt",
        "description": "Gemma 2 2B continued pretrained on SEA-PILE Filipino with stochastic expansion",
        "tokenization": "Stochastic token expansion (~9.3%)",
        "expansion_rate": "9.3%",
        "val_loss": "3.34",
    },
    "patok-step4999": {
        "local_path": "checkpoints/hf/gemma2-2b-patok-step4999",
        "repo_id": "davidafrica/gemma-2-2b-filipino-patok-cpt",
        "description": "Gemma 2 2B continued pretrained on SEA-PILE Filipino with morphology-aware tokenization",
        "tokenization": "Patok: Morphology-aware affix expansion (~46.5%)",
        "expansion_rate": "46.5%",
        "val_loss": "3.42",
    },
}


def create_model_card(config):
    """Create model card content for HuggingFace."""
    return f"""---
language:
- fil
- tl
license: gemma
base_model: google/gemma-2-2b
tags:
- filipino
- tagalog
- continued-pretraining
- morphology
- sea-pile
datasets:
- SEACrowd/sea_pile
pipeline_tag: text-generation
---

# {config['repo_id'].split('/')[-1]}

{config['description']}.

## Model Details

- **Base Model**: [google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b)
- **Language**: Filipino/Tagalog
- **Training Corpus**: [SEA-PILE v2 Filipino subset](https://huggingface.co/datasets/SEACrowd/sea_pile) (7.4GB)
- **Training Steps**: 5,000 steps (320,000 samples)
- **Tokenization Method**: {config['tokenization']}
- **Token Expansion Rate**: {config['expansion_rate']}
- **Final Validation Loss**: {config['val_loss']}

## Tokenization Approach

This model uses **{config['tokenization']}** as part of research on morphology-aware tokenization for Filipino.

Filipino is a morphologically rich language with extensive affixation (prefixes, infixes, suffixes). This model explores whether morphology-aware tokenization improves language model performance on Filipino morphological tasks.

## Training Details

- **Framework**: NVIDIA NeMo 2.0+
- **Training Data**: SEA-PILE v2 Filipino subset
- **Global Batch Size**: 256
- **Learning Rate**: 1e-4
- **Precision**: bfloat16
- **Hardware**: 8x NVIDIA GPUs

## Evaluation

This model was evaluated on 5 Filipino morphology benchmarks:
- **PACUTE**: Filipino-specific morphology (affixation, composition, manipulation, syllabification)
- **Hierarchical**: Multi-level morpheme understanding
- **CUTE**: Character understanding tasks
- **LangGame**: Subword understanding
- **Multi-digit Addition**: Numerical reasoning baseline

See evaluation results in the [main repository](https://github.com/DavidDemitriAfrica/filipino-morphology-llm).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "{config['repo_id']}",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{config['repo_id']}")

# Generate text
prompt = "Ang kabisera ng Pilipinas ay"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Citation

```bibtex
@misc{{gemma2-filipino-cpt-2025,
  title={{Morphology-Aware Tokenization for Filipino Language Models}},
  author={{Africa, David Demitri and
           Montalan, Jann Railey and
           Gamboa, Lance Calvin and
           Layacan, Jimson Paulo and
           Flores, Richell Isaiah and
           Susanto, Yosephine and
           Ngui, Jian Gang}},
  year={{2025}},
  publisher={{HuggingFace}},
  url={{https://huggingface.co/{config['repo_id']}}}
}}
```

## License

This model inherits the [Gemma License](https://www.kaggle.com/models/google/gemma/license/consent/verify/huggingface?returnModelRepoId=google/gemma-2-2b) from the base model.

## Acknowledgements

- Base model: [Google Gemma 2 2B](https://huggingface.co/google/gemma-2-2b)
- Training corpus: [SEA-PILE v2](https://huggingface.co/datasets/SEACrowd/sea_pile)
- Framework: [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
"""


def upload_model(model_name, token=None):
    """Upload a model to HuggingFace Hub."""
    if model_name not in MODEL_CONFIGS:
        print(f"❌ Unknown model: {model_name}")
        print(f"Available models: {', '.join(MODEL_CONFIGS.keys())}")
        return False

    config = MODEL_CONFIGS[model_name]
    local_path = Path(config["local_path"])

    if not local_path.exists():
        print(f"❌ Model not found at: {local_path}")
        print(f"Please run checkpoint conversion first:")
        print(f"  bash scripts/convert_all_checkpoints.sh")
        return False

    print(f"\n{'='*80}")
    print(f"Uploading: {model_name}")
    print(f"{'='*80}")
    print(f"Local path: {local_path}")
    print(f"Repository: {config['repo_id']}")
    print(f"Description: {config['description']}")
    print(f"{'='*80}\n")

    # Create API instance
    api = HfApi(token=token)

    # Create repository if it doesn't exist
    try:
        print(f"Creating repository: {config['repo_id']}...")
        create_repo(
            repo_id=config['repo_id'],
            repo_type="model",
            exist_ok=True,
            token=token,
        )
        print(f"✓ Repository created/verified")
    except Exception as e:
        print(f"⚠️  Repository creation: {e}")

    # Create and upload model card
    try:
        model_card_content = create_model_card(config)
        model_card_path = local_path / "README.md"

        print(f"Creating model card...")
        with open(model_card_path, "w", encoding="utf-8") as f:
            f.write(model_card_content)
        print(f"✓ Model card created")
    except Exception as e:
        print(f"❌ Error creating model card: {e}")
        return False

    # Upload model files
    try:
        print(f"Uploading model files to {config['repo_id']}...")
        api.upload_folder(
            folder_path=str(local_path),
            repo_id=config['repo_id'],
            repo_type="model",
            token=token,
            commit_message=f"Upload {model_name} (val_loss={config['val_loss']}, trained for 5000 steps)",
        )
        print(f"✓ Upload complete!")
        print(f"\n🎉 Model available at: https://huggingface.co/{config['repo_id']}")
        return True
    except Exception as e:
        print(f"❌ Error uploading: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload trained models to HuggingFace Hub")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        help="Model to upload (or 'all' for all models)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN environment variable)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Upload all models"
    )

    args = parser.parse_args()

    # Get token from args or environment
    token = args.token or None  # Let huggingface_hub get it from HF_TOKEN env var

    # Upload models
    if args.all or args.model == "all":
        print("\n" + "="*80)
        print("Uploading ALL models to HuggingFace Hub")
        print("="*80)

        success_count = 0
        for model_name in MODEL_CONFIGS.keys():
            if upload_model(model_name, token):
                success_count += 1

        print(f"\n{'='*80}")
        print(f"✓ Successfully uploaded {success_count}/{len(MODEL_CONFIGS)} models")
        print(f"{'='*80}")
        print(f"\nView your models at: https://huggingface.co/collections/davidafrica/patok")
    else:
        if not args.model:
            parser.print_help()
            return

        upload_model(args.model, token)


if __name__ == "__main__":
    main()
