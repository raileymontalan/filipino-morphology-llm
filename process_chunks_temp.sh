#!/bin/bash
set -euo pipefail

CHUNK_DIR="${1}"
OUTPUT_DIR="${2}"
TOKENIZER="${3}"
WORKERS="${4}"
JSON_KEY="${5}"
TOKENIZATION_MODE="${6}"
EXPAND_PROP="${7}"
CONTRACT_PROP="${8}"
AFFIX_AWARENESS="${9}"
AFFIX_AWARENESS_IF_OVERLAP="${10}"
SEED="${11}"
NUM_CHUNKS="${12}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "Starting sequential chunk processing..."
echo "Container started at: $(date)"
echo ""

SUCCESSFUL_CHUNKS=0
FAILED_CHUNKS=0
SKIPPED_CHUNKS=0

# Loop through all chunks
for i in $(seq 1 ${NUM_CHUNKS}); do
    CHUNK_NUM=$(printf "%04d" ${i})
    CHUNK_FILE="${CHUNK_DIR}/chunk_${CHUNK_NUM}.jsonl"
    OUTPUT_PREFIX="${OUTPUT_DIR}/chunk_${CHUNK_NUM}"
    
    echo "============================================================================"
    echo "Processing chunk ${i}/${NUM_CHUNKS}: chunk_${CHUNK_NUM}.jsonl"
    echo "============================================================================"
    
    # Check if chunk file exists
    if [ ! -f "${CHUNK_FILE}" ]; then
        echo "⚠ Warning: Chunk file not found: ${CHUNK_FILE}"
        echo "Skipping..."
        SKIPPED_CHUNKS=$((SKIPPED_CHUNKS + 1))
        continue
    fi
    
    # Check if output already exists (Python script renames to remove _text_document suffix)
    if [ -f "${OUTPUT_PREFIX}.bin" ] && [ -f "${OUTPUT_PREFIX}.idx" ]; then
        echo "✓ Output files already exist, skipping:"
        echo "  ${OUTPUT_PREFIX}.bin"
        echo "  ${OUTPUT_PREFIX}.idx"
        echo ""
        SKIPPED_CHUNKS=$((SKIPPED_CHUNKS + 1))
        continue
    fi
    
    # Build python command with proper argument handling
    echo "Running: python /workspace/training/nemo/data/preprocess_data.py --input ${CHUNK_FILE} --output-prefix ${OUTPUT_PREFIX} ..."
    echo ""
    
    # Run preprocessing with explicit arguments (not via variable expansion)
    set +e  # Don't exit on error, we want to count failures
    if [ "${TOKENIZATION_MODE}" = "stochastok" ]; then
        python /workspace/training/nemo/data/preprocess_data.py \
            --input "${CHUNK_FILE}" \
            --output-prefix "${OUTPUT_PREFIX}" \
            --tokenizer-model "${TOKENIZER}" \
            --workers "${WORKERS}" \
            --text-key "${JSON_KEY}" \
            --tokenization-mode "${TOKENIZATION_MODE}" \
            --expand-prop "${EXPAND_PROP}" \
            --seed "${SEED}"
        EXIT_CODE=$?
    elif [ "${TOKENIZATION_MODE}" = "patok" ]; then
        python /workspace/training/nemo/data/preprocess_data.py \
            --input "${CHUNK_FILE}" \
            --output-prefix "${OUTPUT_PREFIX}" \
            --tokenizer-model "${TOKENIZER}" \
            --workers "${WORKERS}" \
            --text-key "${JSON_KEY}" \
            --tokenization-mode "${TOKENIZATION_MODE}" \
            --expand-prop "${EXPAND_PROP}" \
            --contract-prop "${CONTRACT_PROP}" \
            --affix-awareness "${AFFIX_AWARENESS}" \
            --affix-awareness-if-overlap "${AFFIX_AWARENESS_IF_OVERLAP}" \
            --seed "${SEED}"
        EXIT_CODE=$?
    else
        python /workspace/training/nemo/data/preprocess_data.py \
            --input "${CHUNK_FILE}" \
            --output-prefix "${OUTPUT_PREFIX}" \
            --tokenizer-model "${TOKENIZER}" \
            --workers "${WORKERS}" \
            --text-key "${JSON_KEY}" \
            --tokenization-mode "${TOKENIZATION_MODE}"
        EXIT_CODE=$?
    fi
    set -e  # Re-enable exit on error
    
    # Check exit code
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Chunk ${i}/${NUM_CHUNKS} completed successfully"
        SUCCESSFUL_CHUNKS=$((SUCCESSFUL_CHUNKS + 1))
    else
        echo ""
        echo "✗ Chunk ${i}/${NUM_CHUNKS} failed"
        FAILED_CHUNKS=$((FAILED_CHUNKS + 1))
    fi
    
    echo ""
done

echo "============================================================================"
echo "Sequential Processing Complete!"
echo "============================================================================"
echo "Container finished at: $(date)"
echo ""
echo "Summary:"
echo "  Total chunks:      ${NUM_CHUNKS}"
echo "  Successful:        ${SUCCESSFUL_CHUNKS}"
echo "  Failed:            ${FAILED_CHUNKS}"
echo "  Skipped (existed): ${SKIPPED_CHUNKS}"
echo "============================================================================"

# Exit with error if any chunks failed
if [ ${FAILED_CHUNKS} -gt 0 ]; then
    exit 1
fi
