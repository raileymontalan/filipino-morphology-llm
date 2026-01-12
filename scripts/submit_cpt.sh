#!/bin/bash
# Submission script for Continued Pretraining (CPT) jobs
# Supports both single job submission and parameter sweeps across models and tokenization methods

set -euo pipefail

# Check if this is a sweep request
if [ "${1:-}" = "sweep" ]; then
    echo "============================================================================"
    echo "CPT Parameter Sweep Mode"
    echo "============================================================================"
    echo "Will submit CPT jobs for all combinations of:"
    echo "  MODELS: google/gemma-3-1b-pt, openai-community/gpt2-xl, cerebras/Cerebras-GPT-1.3B, Qwen/Qwen3-1.7B-Base"
    echo "  TOKENIZATION_MODES: vanilla, stochastok, patok"
    echo ""
    echo "Tokenization parameters per mode:"
    echo "  vanilla: standard BPE"
    echo "  stochastok: expand_prop = 0.1, 0.2, ..., 1.0"
    echo "  patok: expand_prop = 0.1, 0.2"
    echo "         contract_prop = 0.90, 0.80, 0.50"
    echo "         affix_awareness = 0.95, 0.85, 0.75"
    echo "         affix_awareness_if_overlap = 0.75, 0.50"
    echo ""
    echo "Data: 50/50 mix of seapile-v2 and fineweb"
    echo "Training: 100k steps, warmup=500, checkpoint_interval=1000"
    echo ""

    # Define parameter arrays
    MODELS=(
        "google/gemma-3-1b-pt"
        "openai-community/gpt2-xl"
        "cerebras/Cerebras-GPT-1.3B"
        "Qwen/Qwen3-1.7B-Base"
    )
    
    MODES=("vanilla" "stochastok" "patok")

    # Stochastok expand props: 0.1 to 1.0 in steps of 0.1
    STOCHASTOK_EXPAND_PROPS=()
    for i in {1..10}; do
        STOCHASTOK_EXPAND_PROPS+=($(printf "%.1f" $(echo "scale=1; $i/10" | bc)))
    done

    # Patok parameters
    PATOK_EXPAND_PROPS=("0.1" "0.2")
    PATOK_CONTRACT_PROPS=("0.90" "0.80" "0.50")
    PATOK_AFFIX_AWARENESS=("0.95" "0.85" "0.75")
    PATOK_AFFIX_OVERLAP=("0.75" "0.50")

    total_jobs=0

    # Generate all combinations
    for model in "${MODELS[@]}"; do
        # Extract tokenizer name for path construction
        tokenizer="${model}"
        
        for mode in "${MODES[@]}"; do
            if [ "$mode" = "vanilla" ]; then
                # Vanilla: seapile-v2/<tokenizer>/vanilla
                tokenization_path="seapile-v2/${tokenizer}/vanilla"
                echo "Submitting CPT: ${model} | ${tokenization_path}"
                "$0" "${model}" "${tokenization_path}"
                ((total_jobs++))
                
            elif [ "$mode" = "stochastok" ]; then
                for expand_prop in "${STOCHASTOK_EXPAND_PROPS[@]}"; do
                    # Stochastok: seapile-v2/<tokenizer>/stochastok/expand_X.X
                    tokenization_path="seapile-v2/${tokenizer}/stochastok/expand_${expand_prop}"
                    echo "Submitting CPT: ${model} | ${tokenization_path}"
                    "$0" "${model}" "${tokenization_path}"
                    ((total_jobs++))
                done
                
            elif [ "$mode" = "patok" ]; then
                for expand_prop in "${PATOK_EXPAND_PROPS[@]}"; do
                    for contract_prop in "${PATOK_CONTRACT_PROPS[@]}"; do
                        for affix_awareness in "${PATOK_AFFIX_AWARENESS[@]}"; do
                            for affix_overlap in "${PATOK_AFFIX_OVERLAP[@]}"; do
                                # Patok: seapile-v2/<tokenizer>/patok/expand_X.X_contract_Y.Y_affix_Z.Z_overlap_W.W
                                tokenization_path="seapile-v2/${tokenizer}/patok/expand_${expand_prop}_contract_${contract_prop}_affix_${affix_awareness}_overlap_${affix_overlap}"
                                echo "Submitting CPT: ${model} | ${tokenization_path}"
                                "$0" "${model}" "${tokenization_path}"
                                ((total_jobs++))
                            done
                        done
                    done
                done
            fi
        done
    done

    echo ""
    echo "============================================================================"
    echo "CPT Sweep Complete!"
    echo "Total jobs submitted: $total_jobs"
    echo "============================================================================"
    exit 0
fi

# Default values (for single job submission)
MODEL_NAME="${1:-google/gemma-3-1b-pt}"
TOKENIZATION_PATH="${2:-seapile-v2/google-gemma-3-1b-pt/vanilla}"

# Verify preprocessed data exists for this tokenization
# Check both seapile-v2 and fineweb
TOKENIZER=$(echo "${TOKENIZATION_PATH}" | cut -d'/' -f2)
METHOD_PATH=$(echo "${TOKENIZATION_PATH}" | cut -d'/' -f3-)

SEAPILE_DATA="data/processed/${TOKENIZER}/seapile-v2/${METHOD_PATH}"
FINEWEB_DATA="data/processed/${TOKENIZER}/fineweb/${METHOD_PATH}"

missing_data=false
if [ ! -d "${SEAPILE_DATA}" ]; then
    echo "Warning: SEA-PILE data not found: ${SEAPILE_DATA}"
    missing_data=true
fi

if [ ! -d "${FINEWEB_DATA}" ]; then
    echo "Warning: FineWeb data not found: ${FINEWEB_DATA}"
    missing_data=true
fi

if [ "$missing_data" = true ]; then
    echo ""
    echo "Error: Preprocessed data not found for tokenization: ${TOKENIZATION_PATH}"
    echo ""
    echo "Please run preprocessing first:"
    echo "  bash scripts/submit_tokenizations.sh sweep"
    echo ""
    echo "Or preprocess specific configuration:"
    echo "  bash scripts/submit_tokenizations.sh seapile-v2 ${TOKENIZER} <mode> [params...]"
    echo "  bash scripts/submit_tokenizations.sh fineweb ${TOKENIZER} <mode> [params...]"
    exit 1
fi

# Check if both datasets have chunks
SEAPILE_CHUNKS=$(ls -1 "${SEAPILE_DATA}"/chunk_*.bin 2>/dev/null | wc -l)
FINEWEB_CHUNKS=$(ls -1 "${FINEWEB_DATA}"/chunk_*.bin 2>/dev/null | wc -l)

if [ "$SEAPILE_CHUNKS" -eq 0 ] || [ "$FINEWEB_CHUNKS" -eq 0 ]; then
    echo "Error: No chunks found for one or both datasets"
    echo "  SEA-PILE chunks: ${SEAPILE_CHUNKS}"
    echo "  FineWeb chunks: ${FINEWEB_CHUNKS}"
    exit 1
fi

echo "============================================================================"
echo "Submitting Continued Pretraining Job"
echo "============================================================================"
echo "Model: ${MODEL_NAME}"
echo "Tokenization: ${TOKENIZATION_PATH}"
echo "Data chunks found:"
echo "  SEA-PILE: ${SEAPILE_CHUNKS}"
echo "  FineWeb: ${FINEWEB_CHUNKS}"
echo "  Total: $((SEAPILE_CHUNKS + FINEWEB_CHUNKS))"
echo ""
echo "Training configuration:"
echo "  Steps: 100k"
echo "  Warmup: 500 steps"
echo "  Checkpoint interval: 1000 steps"
echo "  Data mix: 50/50 seapile-v2/fineweb"
echo ""

# Submit the job
qsub -v MODEL_NAME="${MODEL_NAME}",TOKENIZATION_PATH="${TOKENIZATION_PATH}" jobs/run_cpt.pbs

echo ""
echo "✓ Job submitted successfully!"
echo "Monitor with: qstat"
echo "============================================================================"
