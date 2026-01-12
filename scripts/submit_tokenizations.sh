#!/bin/bash
# Submission script for sequential preprocessing jobs
# Submits one job per parameter set that processes all chunks sequentially in a single container
# Much more efficient - avoids spinning up a new container for each chunk!

set -euo pipefail

# ============================================================================
# Load Environment
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"

if [ -f "${PROJECT_DIR}/.env" ]; then
    set +u  # Temporarily allow undefined variables
    source "${PROJECT_DIR}/.env"
    set -u
    echo "✓ Environment loaded from ${PROJECT_DIR}/.env"
else
    echo "Warning: .env file not found at ${PROJECT_DIR}/.env"
    echo "Some environment variables may not be set."
fi

# ============================================================================
# Verify Enroot Container Exists
# ============================================================================

CONTAINER_NAME="${CONTAINER_NAME:-nemo_framework}"

if ! command -v enroot &> /dev/null; then
    echo "Error: enroot not found. Please install enroot first."
    exit 1
fi

if ! enroot list | grep -q "^${CONTAINER_NAME}$"; then
    echo "============================================================================"
    echo "Error: Container '${CONTAINER_NAME}' not found"
    echo "============================================================================"
    echo ""
    echo "Available containers:"
    enroot list
    echo ""
    echo "To set up the container, run:"
    echo "  cd ${PROJECT_DIR}"
    echo "  bash training/nemo/setup/setup_enroot.sh"
    echo ""
    echo "Or download your own personal container:"
    echo "  bash scripts/setup_personal_container.sh 25.11"
    echo ""
    exit 1
fi

# Function to submit a single job
submit_job() {
    local dataset=$1
    local tokenizer=$2
    local tokenization_mode=$3
    local seed=${4:-42}
    local expand_prop=${5:-0.1}
    local contract_prop=${6:-0.9}
    local affix_awareness=${7:-0.95}
    local affix_awareness_if_overlap=${8:-0.75}
    
    # Check if dataset directory exists
    local chunk_dir="data/chunks/${dataset}"
    if [ ! -d "${chunk_dir}" ]; then
        echo "Error: Dataset directory not found: ${chunk_dir}"
        return 1
    fi
    
    # Count chunk files
    local num_chunks=$(ls -1 "${chunk_dir}"/chunk_*.jsonl 2>/dev/null | wc -l)
    if [ "$num_chunks" -eq 0 ]; then
        echo "Error: No chunk files found in ${chunk_dir}"
        return 1
    fi
    
    # Submit the sequential job (processes all chunks in one container)
    local job_id=$(qsub -v DATASET="${dataset}",TOKENIZER="${tokenizer}",TOKENIZATION_MODE="${tokenization_mode}",SEED="${seed}",EXPAND_PROP="${expand_prop}",CONTRACT_PROP="${contract_prop}",AFFIX_AWARENESS="${affix_awareness}",AFFIX_AWARENESS_IF_OVERLAP="${affix_awareness_if_overlap}" jobs/preprocess_data_sequential.pbs 2>&1)
    echo "${job_id}"
    
    return 0
}

# Check if this is a sweep request
if [ "${1:-}" = "sweep" ]; then
    set +e  # Disable exit on error for sweep mode
    
    echo "============================================================================"
    echo "Parameter Sweep Mode"
    echo "============================================================================"
    echo "Will submit jobs for all combinations of:"
    echo "  DATASET: seapile-v2, fineweb"
    echo "  TOKENIZER: openai-community/gpt2-xl, google/gemma-3-1b-pt, cerebras/Cerebras-GPT-1.3B, Qwen/Qwen3-1.7B-Base"
    echo "  TOKENIZATION_MODE: vanilla, stochastok, patok"
    echo "  SEED: 42"
    echo ""
    echo "Additional parameters per mode:"
    echo "  stochastok: EXPAND_PROP = 0.1, 0.2, ..., 1.0"
    echo "  patok: EXPAND_PROP = 0.1, 0.2"
    echo "         CONTRACT_PROP = 0.90, 0.80, 0.50"
    echo "         AFFIX_AWARENESS = 0.95, 0.85, 0.75"
    echo "         AFFIX_AWARENESS_IF_OVERLAP = 0.75, 0.50"
    echo ""

    # Define parameter arrays
    DATASETS=("fineweb")
    # DATASETS=("fineweb" "seapile-v2")
    TOKENIZERS=("cerebras/Cerebras-GPT-1.3B")
    # TOKENIZERS=("cerebras/Cerebras-GPT-1.3B" "google/gemma-3-1b-pt" "Qwen/Qwen3-1.7B-Base" "openai-community/gpt2-xl")
    MODES=("patok")
    # MODES=("stochastok" "patok" "vanilla")
    SEED="42"

    # Stochastok expand props: 0.1 to 1.0 in steps of 0.1
    STOCHASTOK_EXPAND_PROPS=()
    for i in {1..10}; do
        STOCHASTOK_EXPAND_PROPS+=($(echo "scale=1; $i/10" | bc))
    done

    # Patok parameters
    PATOK_EXPAND_PROPS=("0.1" "0.2")
    PATOK_CONTRACT_PROPS=("0.90" "0.80" "0.50")
    PATOK_AFFIX_AWARENESS=("0.95" "0.85" "0.75")
    PATOK_AFFIX_OVERLAP=("0.75" "0.50")

    total_jobs=0

    # Generate all combinations
    for dataset in "${DATASETS[@]}"; do
        for tokenizer in "${TOKENIZERS[@]}"; do
            for mode in "${MODES[@]}"; do
                if [ "$mode" = "vanilla" ]; then
                    echo "[$((total_jobs+1))] Submitting: $dataset | $tokenizer | $mode"
                    submit_job "$dataset" "$tokenizer" "$mode"
                    ((total_jobs++))
                elif [ "$mode" = "stochastok" ]; then
                    for expand_prop in "${STOCHASTOK_EXPAND_PROPS[@]}"; do
                        echo "[$((total_jobs+1))] Submitting: $dataset | $tokenizer | $mode | seed=$SEED | expand=$expand_prop"
                        submit_job "$dataset" "$tokenizer" "$mode" "$SEED" "$expand_prop"
                        ((total_jobs++))
                    done
                elif [ "$mode" = "patok" ]; then
                    for expand_prop in "${PATOK_EXPAND_PROPS[@]}"; do
                        for contract_prop in "${PATOK_CONTRACT_PROPS[@]}"; do
                            for affix_awareness in "${PATOK_AFFIX_AWARENESS[@]}"; do
                                for affix_overlap in "${PATOK_AFFIX_OVERLAP[@]}"; do
                                    echo "[$((total_jobs+1))] Submitting: $dataset | $tokenizer | $mode | seed=$SEED | expand=$expand_prop | contract=$contract_prop | affix=$affix_awareness | overlap=$affix_overlap"
                                    submit_job "$dataset" "$tokenizer" "$mode" "$SEED" "$expand_prop" "$contract_prop" "$affix_awareness" "$affix_overlap"
                                    ((total_jobs++))
                                done
                            done
                        done
                    done
                fi
            done
        done
    done

    echo ""
    echo "============================================================================"
    echo "Sweep Complete!"
    echo "Total jobs submitted: $total_jobs"
    echo "============================================================================"
    exit 0
fi

# Single job submission mode
DATASET="${1:-fineweb}"                 # seapile-v2, fineweb
TOKENIZER="${2:-google/gemma-3-1b-pt}"  # google/gemma-3-1b-pt, openai-community/gpt2-xl, cerebras/Cerebras-GPT-1.3B, Qwen/Qwen3-1.7B-Base
TOKENIZATION_MODE="${3:-vanilla}"       # vanilla, stochastok, patok
SEED="${4:-42}"                         # 42
EXPAND_PROP="${5:-0.1}"                 # 0.10, 0.20
CONTRACT_PROP="${6:-0.9}"               # 0.90, 0.80, 0.50
AFFIX_AWARENESS="${7:-0.95}"            # 0.95, 0.85, 0.75
AFFIX_AWARENESS_IF_OVERLAP="${8:-0.75}" # 0.75, 0.50

CHUNK_DIR="data/chunks/${DATASET}"
if [ ! -d "${CHUNK_DIR}" ]; then
    echo "Error: Dataset directory not found: ${CHUNK_DIR}"
    echo "Available datasets:"
    ls -1 "data/chunks/" 2>/dev/null || echo "No datasets found in data/chunks/"
    echo ""
    echo "Usage: $0 sweep"
    echo "       $0 [dataset] [tokenizer] [tokenization_mode] [seed] [expand_prop] [contract_prop] [affix_awareness] [affix_awareness_if_overlap]"
    echo ""
    echo "Sweep mode:"
    echo "  $0 sweep                    # Submit jobs for all parameter combinations"
    echo ""
    echo "Single job mode:"
    echo "  $0 fineweb google/gemma-3-1b-pt vanilla"
    echo "  $0 seapile-v2 google/gemma-3-1b-pt stochastok 42 0.15"
    echo "  $0 seapile-v2 google/gemma-3-1b-pt patok 42 0.1 0.9 0.95 0.75"
    exit 1
fi

NUM_CHUNKS=$(ls -1 "${CHUNK_DIR}"/chunk_*.jsonl 2>/dev/null | wc -l)
if [ "$NUM_CHUNKS" -eq 0 ]; then
    echo "Error: No chunk files found in ${CHUNK_DIR}"
    echo "Expected files like: chunk_0001.jsonl, chunk_0002.jsonl, etc."
    exit 1
fi

echo "============================================================================"
echo "Submitting Sequential Preprocessing Job"
echo "============================================================================"
echo "Dataset: ${DATASET}"
echo "Chunks found: ${NUM_CHUNKS}"
echo "Tokenizer: ${TOKENIZER}"
echo "Tokenization Mode: ${TOKENIZATION_MODE}"
if [ "${TOKENIZATION_MODE}" = "stochastok" ]; then
    echo "Random Seed: ${SEED}"
    echo "Expand Proportion: ${EXPAND_PROP}"
elif [ "${TOKENIZATION_MODE}" = "patok" ]; then
    echo "Random Seed: ${SEED}"
    echo "Expand Proportion: ${EXPAND_PROP}"
    echo "Contract Proportion: ${CONTRACT_PROP}"
    echo "Affix Awareness: ${AFFIX_AWARENESS}"
    echo "Affix Awareness If Overlap: ${AFFIX_AWARENESS_IF_OVERLAP}"
fi
echo ""

echo "Submitting job..."
submit_job "${DATASET}" "${TOKENIZER}" "${TOKENIZATION_MODE}" "${SEED}" "${EXPAND_PROP}" "${CONTRACT_PROP}" "${AFFIX_AWARENESS}" "${AFFIX_AWARENESS_IF_OVERLAP}"

echo ""
echo "✓ Job submitted successfully!"
echo "Monitor with: qstat"
echo "============================================================================"