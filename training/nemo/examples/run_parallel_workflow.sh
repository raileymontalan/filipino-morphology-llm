#!/bin/bash
# Example workflow: Parallel preprocessing + training with chunks
#
# This demonstrates the complete workflow from raw JSONL to training
# using automatic chunking (2^16 lines per chunk) and dynamic job arrays.
#
# The workflow:
# 1. Split JSONL into chunks of 65,536 lines each
# 2. Submit parallel preprocessing jobs (auto-detects chunk count)
# 3. Verify all chunks were processed successfully
# 4. Generate training data paths
# 5. Submit training job with all chunks

set -e

echo "=========================================="
echo "Parallel Data Preprocessing + Training"
echo "=========================================="
echo ""

# Configuration
INPUT_JSONL="/scratch_aisg/SPEC-SF-AISG/railey/data/corpora/seapile-v2.jsonl"
DATASET="seapile-v2"  # Dataset name (directory under data/chunks/)
TOKENIZER="google/gemma-3-1b-pt"
TOKENIZER_NAME=$(echo ${TOKENIZER} | sed 's/\//-/g')  # Convert slashes to dashes
TOKENIZATION_MODE="vanilla"  # or "stochastok" or "patok"
SEED="42"  # Random seed for stochastok/patok modes
EXPAND_PROP="0.1"  # For stochastok/patok modes
CONTRACT_PROP="0.9"  # For patok mode
AFFIX_AWARENESS="0.95"  # For patok mode
AFFIX_AWARENESS_IF_OVERLAP="0.75"  # For patok mode

# Set output directory based on tokenization mode
if [ "${TOKENIZATION_MODE}" = "stochastok" ]; then
    OUTPUT_DIR="/scratch_aisg/SPEC-SF-AISG/railey/data/processed/${DATASET}/${TOKENIZER_NAME}/stochastok/expand_${EXPAND_PROP}"
elif [ "${TOKENIZATION_MODE}" = "patok" ]; then
    OUTPUT_DIR="/scratch_aisg/SPEC-SF-AISG/railey/data/processed/${DATASET}/${TOKENIZER_NAME}/patok/expand_${EXPAND_PROP}_contract_${CONTRACT_PROP}_affix_${AFFIX_AWARENESS}_overlap_${AFFIX_AWARENESS_IF_OVERLAP}"
else
    OUTPUT_DIR="/scratch_aisg/SPEC-SF-AISG/railey/data/processed/${DATASET}/${TOKENIZER_NAME}/${TOKENIZATION_MODE}"
fi

CHUNK_DIR="/scratch_aisg/SPEC-SF-AISG/railey/data/chunks/${DATASET}"

echo "Configuration:"
echo "  Input: ${INPUT_JSONL}"
echo "  Dataset: ${DATASET}"
echo "  Tokenizer: ${TOKENIZER}"
echo "  Tokenization Mode: ${TOKENIZATION_MODE}"
if [ "${TOKENIZATION_MODE}" = "stochastok" ]; then
    echo "  Expand Proportion: ${EXPAND_PROP}"
    echo "  Random Seed: ${SEED}"
elif [ "${TOKENIZATION_MODE}" = "patok" ]; then
    echo "  Expand Proportion: ${EXPAND_PROP}"
    echo "  Contract Proportion: ${CONTRACT_PROP}"
    echo "  Affix Awareness: ${AFFIX_AWARENESS}"
    echo "  Random Seed: ${SEED}"
fi
echo "  Chunk dir: ${CHUNK_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo ""

# Step 1: Split JSONL into chunks
echo "Step 1: Splitting JSONL into chunks (2^16 lines each)..."
echo "----------------------------------------"
python training/nemo/data/split_jsonl.py \
    --input "${INPUT_JSONL}" \
    --output-dir "data/chunks" \

echo "✓ Split complete"
echo ""

# Count actual number of chunks created
NUM_CHUNKS=$(ls -1 "${CHUNK_DIR}"/chunk_*.jsonl 2>/dev/null | wc -l)
echo "Created ${NUM_CHUNKS} chunks"
echo ""

# Step 2: Submit parallel preprocessing jobs
echo "Step 2: Submitting parallel preprocessing jobs..."
echo "----------------------------------------"
if [ "${TOKENIZATION_MODE}" = "stochastok" ]; then
    PREPROCESS_JOB=$("../../scripts/submit_preprocessing.sh" "${DATASET}" "${TOKENIZER}" "${TOKENIZATION_MODE}" "${SEED}" "${EXPAND_PROP}")
elif [ "${TOKENIZATION_MODE}" = "patok" ]; then
    PREPROCESS_JOB=$("../../scripts/submit_preprocessing.sh" "${DATASET}" "${TOKENIZER}" "${TOKENIZATION_MODE}" "${SEED}" "${EXPAND_PROP}" "${CONTRACT_PROP}" "${AFFIX_AWARENESS}" "${AFFIX_AWARENESS_IF_OVERLAP}")
else
    PREPROCESS_JOB=$("../../scripts/submit_preprocessing.sh" "${DATASET}" "${TOKENIZER}" "${TOKENIZATION_MODE}")
fi
PREPROCESS_JOB_ID=$(echo $PREPROCESS_JOB | grep -o '[0-9]\+\[[0-9]*\]' | head -1)

echo "✓ Submitted job: ${PREPROCESS_JOB}"
echo ""
echo "Monitoring preprocessing progress:"
echo "  qstat -t ${PREPROCESS_JOB_ID}"
echo "  # Or watch continuously:"
echo "  watch -n 5 'qstat -t ${PREPROCESS_JOB_ID}'"
echo ""
echo "Waiting for preprocessing to complete..."
echo "(Press Ctrl+C if you want to monitor manually)"
echo ""

# Wait for preprocessing to complete
while qstat -t "${PREPROCESS_JOB_ID}" &> /dev/null; do
    sleep 30
    echo -n "."
done
echo ""
echo "✓ Preprocessing complete"
echo ""

# Step 3: Verify all chunks were created
echo "Step 3: Verifying preprocessed chunks..."
echo "----------------------------------------"
missing_chunks=0
for i in $(seq -f "%04g" 1 ${NUM_CHUNKS}); do
    chunk_prefix="${OUTPUT_DIR}/chunk_${i}"
    if [ ! -f "${chunk_prefix}.bin" ] || [ ! -f "${chunk_prefix}.idx" ]; then
        echo "✗ Missing: ${chunk_prefix}.{bin,idx}"
        missing_chunks=$((missing_chunks + 1))
    fi
done

if [ $missing_chunks -gt 0 ]; then
    echo ""
    echo "✗ Error: ${missing_chunks} chunks are missing!"
    echo "Check preprocessing logs in: /scratch_aisg/SPEC-SF-AISG/railey/logs/preprocessing/"
    exit 1
fi

echo "✓ All ${NUM_CHUNKS} chunks verified"
echo ""

# Step 4: Generate chunk paths for training
echo "Step 4: Generating chunk paths for training..."
echo "----------------------------------------"
if [ "${TOKENIZATION_MODE}" = "stochastok" ]; then
    CHUNK_PATHS=$(training/nemo/data/generate_chunk_paths.sh "${INPUT_JSONL}" "${TOKENIZER}" "${TOKENIZATION_MODE}" "${SEED}" "${EXPAND_PROP}")
elif [ "${TOKENIZATION_MODE}" = "patok" ]; then
    CHUNK_PATHS=$(training/nemo/data/generate_chunk_paths.sh "${INPUT_JSONL}" "${TOKENIZER}" "${TOKENIZATION_MODE}" "${SEED}" "${EXPAND_PROP}" "${CONTRACT_PROP}" "${AFFIX_AWARENESS}" "${AFFIX_AWARENESS_IF_OVERLAP}")
else
    CHUNK_PATHS=$(training/nemo/data/generate_chunk_paths.sh "${INPUT_JSONL}" "${TOKENIZER}" "${TOKENIZATION_MODE}")
fi
echo "✓ Chunk paths generated"
echo ""

# Step 5: Submit training job
echo "Step 5: Submitting training job with ${NUM_CHUNKS} chunks..."
echo "----------------------------------------"

TRAIN_JOB=$(qsub -v DATA_PATH="${CHUNK_PATHS}" jobs/run_cpt.pbs)

echo "✓ Submitted training job: ${TRAIN_JOB}"
echo ""
echo "=========================================="
echo "Workflow Complete!"
echo "=========================================="
echo ""
echo "Training job submitted: ${TRAIN_JOB}"
echo "Output will be in: ${OUTPUT_DIR}/"
echo ""
echo "Monitor training:"
echo "  qstat ${TRAIN_JOB}"
echo "  tail -f /scratch_aisg/SPEC-SF-AISG/railey/logs/${TRAIN_JOB}.OU"
echo ""
echo "View WandB logs:"
echo "  Check your WandB project at: https://wandb.ai"
echo ""
