#!/bin/bash

# ============================================================================
# Submit Parallel Evaluation Jobs (Run from login node)
# ============================================================================
#
# This script submits separate PBS jobs for each model group.
# Must be run from the login node (not within a PBS job)!
#
# Usage:
#   bash scripts/submit_parallel_eval.sh
#
# To customize:
#   BENCHMARKS="pacute cute" bash scripts/submit_parallel_eval.sh
#   MAX_SAMPLES=100 bash scripts/submit_parallel_eval.sh
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
BENCHMARKS="${BENCHMARKS:-pacute cute hierarchical langgame multi-digit-addition}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/benchmark_evaluation}"
QUEUE="${QUEUE:-AISG_debug}"
WALLTIME="${WALLTIME:-08:00:00}"

# Model groups
MODELS_GPT2="${MODELS_GPT2:-gpt2 gpt2-medium gpt2-large}"
MODELS_QWEN="${MODELS_QWEN:-qwen-2.5-0.5b qwen-2.5-0.5b-it qwen-2.5-1.5b qwen-2.5-1.5b-it}"
MODELS_CEREBRAS="${MODELS_CEREBRAS:-cerebras-gpt-111m cerebras-gpt-256m cerebras-gpt-590m}"
MODELS_LLAMA="${MODELS_LLAMA:-llama-3.2-1b llama-3.2-1b-it}"
MODELS_GEMMA="${MODELS_GEMMA:-gemma-2b gemma-2b-it}"

# Temporary directory for job scripts
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEMP_DIR="/scratch_aisg/SPEC-SF-AISG/railey/logs/eval_jobs_${TIMESTAMP}"
mkdir -p ${TEMP_DIR}

echo "============================================================================"
echo "Parallel Evaluation Job Submission"
echo "============================================================================"
echo "Benchmarks: ${BENCHMARKS}"
echo "Max Samples: ${MAX_SAMPLES:-all}"
echo "Output: ${OUTPUT_DIR}"
echo "Temp scripts: ${TEMP_DIR}"
echo "============================================================================"
echo ""

# ============================================================================
# Helper Function to Create and Submit Job
# ============================================================================

submit_eval_job() {
    local model_group=$1
    local models=$2
    
    echo "Submitting: ${model_group}"
    echo "  Models: ${models}"
    
    local job_script="${TEMP_DIR}/eval_${model_group}.pbs"
    
    # Create job script
    cat > ${job_script} << EOF
#!/bin/bash
#PBS -N eval_${model_group}
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=${WALLTIME}
#PBS -q ${QUEUE}
#PBS -j oe
#PBS -o /scratch_aisg/SPEC-SF-AISG/railey/logs/

set -euo pipefail

cd ${PROJECT_DIR}

# Activate environment
source env/bin/activate

echo "============================================================================"
echo "Evaluating: ${model_group}"
echo "Models: ${models}"
echo "Benchmarks: ${BENCHMARKS}"
echo "============================================================================"

# Run evaluation
python scripts/run_evaluation.py \\
    --models ${models} \\
    --benchmarks ${BENCHMARKS} \\
    --output-dir ${OUTPUT_DIR} \\
    --device cuda $([ -n "${MAX_SAMPLES}" ] && echo "--max-samples ${MAX_SAMPLES}" || echo "")

exit_code=\$?

if [ \${exit_code} -eq 0 ]; then
    echo "✓ ${model_group} completed successfully"
else
    echo "✗ ${model_group} failed with exit code \${exit_code}"
fi

exit \${exit_code}
EOF
    
    # Submit job
    local job_id=$(qsub ${job_script})
    echo "  → Submitted: ${job_id}"
    echo "${job_id} ${model_group}" >> ${TEMP_DIR}/submitted_jobs.txt
    echo ""
}

# ============================================================================
# Submit All Jobs
# ============================================================================

echo "# Job submissions at ${TIMESTAMP}" > ${TEMP_DIR}/submitted_jobs.txt

submit_eval_job "gpt2" "${MODELS_GPT2}"
submit_eval_job "qwen" "${MODELS_QWEN}"
submit_eval_job "cerebras" "${MODELS_CEREBRAS}"
submit_eval_job "llama" "${MODELS_LLAMA}"
submit_eval_job "gemma" "${MODELS_GEMMA}"

# ============================================================================
# Summary
# ============================================================================

echo "============================================================================"
echo "Submitted 5 parallel evaluation jobs!"
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
