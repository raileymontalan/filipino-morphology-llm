#!/bin/bash
# Preprocess all chunks for vanilla baseline tokenization
# Runs 4 containers in parallel to balance speed vs resources

WORKSPACE="/home/ubuntu/filipino-morphology-llm"
TOKENIZER_MODEL="/workspace/data/tokenizer/gemma2_tokenizer.model"
INPUT_DIR="/workspace/data/chunks/google-gemma-3-1b-pt"
OUTPUT_DIR="/workspace/data/processed/vanilla"
NEMO_IMAGE="nvcr.io/nvidia/nemo:24.07"

# Create output directory
mkdir -p "$WORKSPACE/data/processed/vanilla"

# Function to preprocess a single chunk
preprocess_chunk() {
    local chunk_num=$1
    local chunk_file=$(printf "chunk_%04d.jsonl" $chunk_num)
    local output_prefix=$(printf "chunk_%04d" $chunk_num)

    echo "[$(date '+%H:%M:%S')] Starting chunk $chunk_num..."

    docker run --rm --gpus all --ipc=host \
        -v "$WORKSPACE:/workspace" \
        "$NEMO_IMAGE" \
        python /opt/megatron-lm/tools/preprocess_data.py \
            --input "$INPUT_DIR/$chunk_file" \
            --output-prefix "$OUTPUT_DIR/$output_prefix" \
            --tokenizer-type SentencePieceTokenizer \
            --tokenizer-model "$TOKENIZER_MODEL" \
            --json-keys text \
            --workers 32 \
            --append-eod > /tmp/preprocess_chunk_${chunk_num}.log 2>&1

    if [ $? -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Chunk $chunk_num completed successfully"
    else
        echo "[$(date '+%H:%M:%S')] Chunk $chunk_num FAILED"
        tail -20 /tmp/preprocess_chunk_${chunk_num}.log
    fi
}

export -f preprocess_chunk
export WORKSPACE TOKENIZER_MODEL INPUT_DIR OUTPUT_DIR NEMO_IMAGE

echo "=========================================="
echo "Starting vanilla preprocessing (20 chunks)"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Process chunks 2-20 (chunk 1 already done) in parallel batches
# Run 4 at a time to balance resources
for batch_start in 2 6 10 14 18; do
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
echo "Vanilla preprocessing complete!"
echo "End time: $(date)"
echo "=========================================="

# Verify outputs
echo ""
echo "Output files:"
ls -lh "$WORKSPACE/data/processed/vanilla/"*.bin 2>/dev/null | wc -l
echo "bin files created"
ls -lh "$WORKSPACE/data/processed/vanilla/"*.idx 2>/dev/null | wc -l
echo "idx files created"
