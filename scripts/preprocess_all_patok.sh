#!/bin/bash
# Preprocess all chunks for patok (morphology-aware) tokenization
# Runs 4 containers in parallel to balance speed vs resources

WORKSPACE="/home/ubuntu/filipino-morphology-llm"
TOKENIZER_MODEL="/workspace/data/tokenizer/gemma2_tokenizer.model"
HF_TOKENIZER="google/gemma-2-2b"
INPUT_DIR="/workspace/data/chunks/google-gemma-3-1b-pt"
OUTPUT_DIR="/workspace/data/processed/patok"
NEMO_IMAGE="nvcr.io/nvidia/nemo:24.07"
CONTRACT_PROP=0.3
EXPAND_PROP=0.3
AFFIX_AWARENESS=0.95
HF_CACHE="/home/ubuntu/.cache/huggingface"

# Create output directory
mkdir -p "$WORKSPACE/data/processed/patok"

# Function to preprocess a single chunk
preprocess_chunk() {
    local chunk_num=$1
    local chunk_file=$(printf "chunk_%04d.jsonl" $chunk_num)
    local output_prefix=$(printf "chunk_%04d" $chunk_num)

    echo "[$(date '+%H:%M:%S')] Starting patok chunk $chunk_num..."

    docker run --rm --gpus all --ipc=host \
        -v "$WORKSPACE:/workspace" \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -e HF_HOME=/root/.cache/huggingface \
        "$NEMO_IMAGE" \
        python /workspace/training/nemo/data/preprocess_data.py \
            --input "$INPUT_DIR/$chunk_file" \
            --output-prefix "$OUTPUT_DIR/$output_prefix" \
            --tokenizer-model "$TOKENIZER_MODEL" \
            --hf-tokenizer "$HF_TOKENIZER" \
            --tokenization-mode patok \
            --contract-prop $CONTRACT_PROP \
            --expand-prop $EXPAND_PROP \
            --affix-awareness $AFFIX_AWARENESS \
            --prefix-file /workspace/src/tokenization/affixes/prefix.txt \
            --infix-file /workspace/src/tokenization/affixes/infix.txt \
            --suffix-file /workspace/src/tokenization/affixes/suffix.txt \
            --workers 32 > /tmp/preprocess_patok_${chunk_num}.log 2>&1

    if [ $? -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Patok chunk $chunk_num completed"
    else
        echo "[$(date '+%H:%M:%S')] Patok chunk $chunk_num FAILED"
        tail -20 /tmp/preprocess_patok_${chunk_num}.log
    fi
}

export -f preprocess_chunk
export WORKSPACE TOKENIZER_MODEL HF_TOKENIZER INPUT_DIR OUTPUT_DIR NEMO_IMAGE CONTRACT_PROP EXPAND_PROP AFFIX_AWARENESS HF_CACHE

echo "=========================================="
echo "Starting patok preprocessing (20 chunks)"
echo "Parallel batches of 4"
echo "Contract: $CONTRACT_PROP, Expand: $EXPAND_PROP"
echo "Affix awareness: $AFFIX_AWARENESS"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Process all 20 chunks in parallel batches of 4
for batch_start in 1 5 9 13 17; do
    batch_end=$((batch_start + 3))
    if [ $batch_end -gt 20 ]; then
        batch_end=20
    fi

    echo ""
    echo "Processing batch: chunks $batch_start to $batch_end"

    for chunk in $(seq $batch_start $batch_end); do
        preprocess_chunk $chunk &
    done

    # Wait for this batch to complete
    wait
done

echo ""
echo "=========================================="
echo "Patok preprocessing complete!"
echo "End time: $(date)"
echo "=========================================="

# Verify outputs
echo ""
echo "Output files:"
ls -lh "$WORKSPACE/data/processed/patok/"*.bin 2>/dev/null | wc -l
echo "bin files created"
ls -lh "$WORKSPACE/data/processed/patok/"*.idx 2>/dev/null | wc -l
echo "idx files created"

# Fix permissions
sudo chown -R ubuntu:ubuntu "$WORKSPACE/data/processed/patok/" 2>/dev/null
