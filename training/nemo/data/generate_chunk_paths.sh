#!/bin/bash
# Helper script to generate DATA_PATH string for multiple chunks
# Chunks the input file every 2^16 lines
#
# Usage:
#   # Generate paths for chunks of 65536 lines each with default tokenizer
#   ./generate_chunk_paths.sh input_file.jsonl
#
#   # Generate paths for specific tokenizer and tokenization mode
#   ./generate_chunk_paths.sh input_file.jsonl google/gemma-3-1b-pt vanilla
#   ./generate_chunk_paths.sh input_file.jsonl google/gemma-3-1b-pt stochastok 0.15 12345
#
#   # Use in qsub:
#   DATA_PATH=$(./generate_chunk_paths.sh input_file.jsonl) qsub jobs/run_cpt.pbs

set -euo pipefail

INPUT_FILE=${1}
TOKENIZER=${2:-google/gemma-3-1b-pt}
TOKENIZATION_MODE=${3:-vanilla}
SEED=${4:-42}
EXPAND_PROP=${5:-0.1}
CONTRACT_PROP=${6:-0.9}
AFFIX_AWARENESS=${7:-0.95}
AFFIX_AWARENESS_IF_OVERLAP=${8:-0.75}

# Calculate number of chunks based on 2^16 lines per chunk
CHUNK_SIZE=65536
TOTAL_LINES=$(wc -l < "$INPUT_FILE")
NUM_CHUNKS=$(( (TOTAL_LINES + CHUNK_SIZE - 1) / CHUNK_SIZE ))

TOKENIZER_NAME=$(echo ${TOKENIZER} | sed 's/\//-/g')  # Convert slashes to dashes
# Extract dataset name from input file path
DATASET=$(basename "$INPUT_FILE" .jsonl)

# Construct path based on tokenization mode
if [ "${TOKENIZATION_MODE}" = "stochastok" ]; then
    SUBDIR="${TOKENIZER_NAME}/stochastok/expand_${EXPAND_PROP}"
elif [ "${TOKENIZATION_MODE}" = "patok" ]; then
    SUBDIR="${TOKENIZER_NAME}/patok/expand_${EXPAND_PROP}_contract_${CONTRACT_PROP}_affix_${AFFIX_AWARENESS}_overlap_${AFFIX_AWARENESS_IF_OVERLAP}"
else
    SUBDIR="${TOKENIZER_NAME}/${TOKENIZATION_MODE}"
fi

PREFIX="/workspace/data/processed/${DATASET}/${SUBDIR}/chunk"

paths=""
for i in $(seq -f "%04g" 1 $NUM_CHUNKS); do
    if [ -z "$paths" ]; then
        paths="${PREFIX}_${i}"
    else
        paths="${paths} ${PREFIX}_${i}"
    fi
done

echo "$paths"
