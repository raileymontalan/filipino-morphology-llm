#!/bin/bash

# ============================================================================
# Submit Parallel Evaluation Jobs (Run from login node) - TEMPLATE
# ============================================================================
#
# This is a TEMPLATE file. Copy to scripts/ and customize:
#   1. Replace YOUR_QUEUE_NAME with your cluster queue
#   2. Replace /path/to/your/logs/ with your log directory
#   3. Replace /path/to/your/temp/ with your temp directory
#   4. Replace /path/to/your/configs/ with your configs directory
#
# This script submits separate PBS jobs for EACH MODEL.
# Each model gets its own job with 1 GPU!
# Must be run from the login node (not within a PBS job)!
#
# Usage:
#   cp job_templates/submit_parallel_evaluation.template.sh scripts/submit_parallel_eval.sh
#   # Edit scripts/submit_parallel_eval.sh with your paths
#   bash scripts/submit_parallel_eval.sh
#
# To customize:
#   BENCHMARKS="pacute-affixation-mcq cute-gen" bash scripts/submit_parallel_eval.sh
#   MAX_SAMPLES=100 bash scripts/submit_parallel_eval.sh
#   ALL_MODELS="gpt2 gemma-2b" bash scripts/submit_parallel_eval.sh
#   MODEL_CONFIG="custom_models.yaml" bash scripts/submit_parallel_eval.sh
#
# ============================================================================

set -euo pipefail

# Get project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

cd ${PROJECT_DIR}

# ============================================================================
# Configuration
# ============================================================================

# Can be overridden via environment variables
BENCHMARKS="${BENCHMARKS:-pacute-affixation-mcq pacute-composition-mcq pacute-manipulation-mcq pacute-syllabification-mcq hierarchical-mcq langgame-mcq multi-digit-addition-mcq cute-gen pacute-affixation-gen pacute-composition-gen pacute-manipulation-gen pacute-syllabification-gen hierarchical-gen langgame-gen multi-digit-addition-gen}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/benchmark_evaluation}"
QUEUE="${QUEUE:-YOUR_QUEUE_NAME}"
WALLTIME="${WALLTIME:-08:00:00}"
MODEL_CONFIG="${MODEL_CONFIG:-/path/to/your/configs/models.yaml}"

# Load models from YAML config (can be overridden via ALL_MODELS env var)
if [ -z "${ALL_MODELS:-}" ]; then
    echo "Loading models from: ${MODEL_CONFIG}"
    # Extract model names from YAML (lines that start with spaces followed by model name and colon)
    ALL_MODELS=$(grep -E '^  [a-z0-9.-]+:$' ${MODEL_CONFIG} | sed 's/://g' | tr -d ' ' | tr '\n' ' ')
    echo "Found $(echo ${ALL_MODELS} | wc -w) models in config"
else
    echo "Using models from ALL_MODELS environment variable"
fi

# Temporary directory for job scripts
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEMP_DIR="/path/to/your/temp/eval_jobs_${TIMESTAMP}"
mkdir -p ${TEMP_DIR}

# Convert space-separated list to array
read -ra MODELS_ARRAY <<< "${ALL_MODELS}"
NUM_MODELS=${#MODELS_ARRAY[@]}

echo "============================================================================"
echo "Parallel Evaluation Job Submission"
echo "============================================================================"
echo "Model Config: ${MODEL_CONFIG}"
echo "Models: ${NUM_MODELS} total"
for model in "${MODELS_ARRAY[@]}"; do
    echo "  - ${model}"
done
echo ""
echo "Benchmarks: ${BENCHMARKS}"
echo "Max Samples: ${MAX_SAMPLES:-all}"
echo "Output: ${OUTPUT_DIR}"
echo "Temp scripts: ${TEMP_DIR}"
echo "============================================================================"
echo ""

# ============================================================================
# Helper Function to Create and Submit Job for Single Model
# ============================================================================

submit_eval_job() {
    local model_name=$1

    echo "Submitting: ${model_name}"

    local job_script="${TEMP_DIR}/eval_${model_name}.pbs"

    # Create job script
    cat > ${job_script} << EOF
#!/bin/bash
#PBS -N eval_${model_name}
#PBS -l select=1:ngpus=1
#PBS -l walltime=${WALLTIME}
#PBS -q ${QUEUE}
#PBS -j oe
#PBS -o /path/to/your/logs/

set -euo pipefail

cd ${PROJECT_DIR}

# Activate environment
source env/bin/activate

echo "============================================================================"
echo "Evaluating: ${model_name}"
echo "Benchmarks: ${BENCHMARKS}"
echo "============================================================================"

# Run evaluation
python scripts/run_evaluation.py \\
    --models ${model_name} \\
    --benchmarks ${BENCHMARKS} \\
    --output-dir ${OUTPUT_DIR} \\
    --device cuda $([ -n "${MAX_SAMPLES}" ] && echo "--max-samples ${MAX_SAMPLES}" || echo "")

exit_code=\$?

if [ \${exit_code} -eq 0 ]; then
    echo "✓ ${model_name} completed successfully"
else
    echo "✗ ${model_name} failed with exit code \${exit_code}"
fi

exit \${exit_code}
EOF

    # Submit job
    local job_id=$(qsub ${job_script})
    echo "  → Submitted: ${job_id}"
    echo "${job_id} ${model_name}" >> ${TEMP_DIR}/submitted_jobs.txt
    echo ""
}

# ============================================================================
# Submit All Jobs
# ============================================================================

echo "# Job submissions at ${TIMESTAMP}" > ${TEMP_DIR}/submitted_jobs.txt

# Submit one job per model
for model in "${MODELS_ARRAY[@]}"; do
    submit_eval_job "${model}"
done

# ============================================================================
# Summary
# ============================================================================

echo "============================================================================"
echo "Submitted ${NUM_MODELS} parallel evaluation jobs!"
echo "============================================================================"
echo ""
echo "Track progress:"
echo "  qstat -u \$USER"
echo ""
echo "View submitted jobs:"
echo "  cat ${TEMP_DIR}/submitted_jobs.txt"
echo ""
echo "Results will be saved to:"
echo "  ${OUTPUT_DIR}"
echo "============================================================================"
