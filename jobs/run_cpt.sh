#!/bin/bash

set -euo pipefail

# ============================================================================
# Distributed Training Launch Script
# ============================================================================
# This script is called from the PBS/SLURM job script and runs inside
# the Enroot container to launch distributed training with torchrun.
#
# Arguments:
#   $1: GPUs per node
#   $2: World size (total GPUs)
#   $3: Number of nodes
#   $4: Master address
#   $5: Master port
#   $6: Node rank (optional, defaults to 0)
#   $7: Local rank (optional, defaults to 0)
# ============================================================================

gpus_per_node=${1:-$(nvidia-smi -L | wc -l)}
world_size=${2:-$gpus_per_node}
num_node=${3:-1}
master_addr=${4:-'127.0.0.1'}
master_port=${5:-$((10000 + $RANDOM % 9000))}
node_rank=${6:-0}
local_rank=${7:-0}

# ============================================================================
# Activate Python Environment (if using venv inside container)
# ============================================================================

# Uncomment if you have a specific venv to activate inside the container
# source /opt/venv/bin/activate

cd ${JOB_WORK_DIR}

# ============================================================================
# Set Python Path (if needed for custom modules)
# ============================================================================

# Example: Add custom modules to Python path
# export PYTHONPATH="${MEGATRON_BRIDGE_DIR}:${MEGATRON_LM_DIR}:${PYTHONPATH}"

# ============================================================================
# CUDA Configuration
# ============================================================================

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ============================================================================
# Log Environment Information (Master node only)
# ============================================================================

if [ $node_rank -eq 0 ]; then
    sh_log_file=${LOG_DIR}/${node_rank}_sh.log
    mkdir -p "${LOG_DIR}"
    
    echo "============================================================================" > $sh_log_file
    echo "Environment Information" >> $sh_log_file
    echo "============================================================================" >> $sh_log_file
    echo "Current working directory: $(pwd)" >> $sh_log_file
    echo "PATH: $PATH" >> $sh_log_file
    echo "PYTHONPATH: ${PYTHONPATH:-not set}" >> $sh_log_file
    echo "" >> $sh_log_file
    
    echo "Python Information:" >> $sh_log_file
    echo "  which python: $(which python)" >> $sh_log_file
    echo "  python version: $(python --version)" >> $sh_log_file
    echo "" >> $sh_log_file
    
    echo "NeMo Information:" >> $sh_log_file
    python3 -c "import nemo; print('  NeMo version:', nemo.__version__)" >> $sh_log_file 2>&1 || echo "  Failed to import nemo" >> $sh_log_file
    python3 -c "import nemo; print('  NeMo location:', nemo.__file__)" >> $sh_log_file 2>&1 || echo "  Failed to get nemo location" >> $sh_log_file
    echo "" >> $sh_log_file
    
    echo "PyTorch Information:" >> $sh_log_file
    python3 -c "import torch; print('  PyTorch version:', torch.__version__)" >> $sh_log_file 2>&1 || echo "  Failed to import torch" >> $sh_log_file
    python3 -c "import torch; print('  CUDA available:', torch.cuda.is_available())" >> $sh_log_file 2>&1
    python3 -c "import torch; print('  CUDA version:', torch.version.cuda)" >> $sh_log_file 2>&1
    python3 -c "import torch; print('  Number of GPUs:', torch.cuda.device_count())" >> $sh_log_file 2>&1
    echo "" >> $sh_log_file
    
    echo "Megatron Core Information:" >> $sh_log_file
    python3 -c "import megatron.core; print('  megatron.core version:', megatron.core.__version__)" >> $sh_log_file 2>&1 || echo "  Failed to import megatron.core" >> $sh_log_file
    python3 -c "import megatron.core; print('  megatron.core location:', megatron.core.__file__)" >> $sh_log_file 2>&1 || echo "  Failed to get megatron.core location" >> $sh_log_file
    echo "" >> $sh_log_file
    
    echo "Environment Variables:" >> $sh_log_file
    env | sort >> ${LOG_DIR}/envvar/EnvVar_container.log
fi

# ============================================================================
# Torchrun Configuration
# ============================================================================

export TORCHELASTIC_EXIT_TIMEOUT=120

distributed_args=(
    --nnodes ${num_node}
    --node_rank ${node_rank}
    --nproc_per_node ${gpus_per_node}
    --rdzv_endpoint "${master_addr}:${master_port}"
    --rdzv_backend static
    --rdzv_conf "timeout=120,endpoint_timeout=120"
)

# ============================================================================
# Python Script Arguments
# ============================================================================

script_args=(
    "${PYTHON_SCRIPT}"
    --data-path "${DATA_PATH}"
    --seq-length "${SEQ_LENGTH}"
    --max-steps "${MAX_STEPS}"
    --global-batch-size "${GBS}"
    --micro-batch-size "${MBS}"
    --devices "${DEVICES}"
    --lr "${LR}"
    --min-lr "${MIN_LR}"
    --warmup-steps "${WARMUP_STEPS}"
    --checkpoint-dir "${CKPT_DIR}"
    --checkpoint-interval "${CKPT_INTERVAL}"
    --resume-from "${RESUME_FROM}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-name "${WANDB_NAME}"
    --log-dir "${LOG_DIR}"
    --log-every-n-steps "${LOG_EVERY_N_STEPS}"
    --val-check-interval "${VAL_CHECK_INTERVAL}"
)

# ============================================================================
# Launch Training
# ============================================================================

echo "============================================================================"
echo "Launching Distributed Training"
echo "============================================================================"
echo "Configuration:"
echo "  Nodes: ${num_node}"
echo "  Node rank: ${node_rank}"
echo "  GPUs per node: ${gpus_per_node}"
echo "  World size: ${world_size}"
echo "  Master address: ${master_addr}"
echo "  Master port: ${master_port}"
echo ""
echo "Training Parameters:"
echo "  Data path: ${DATA_PATH}"
echo "  Sequence length: ${SEQ_LENGTH}"
echo "  Max steps: ${MAX_STEPS}"
echo "  Global batch size: ${GBS}"
echo "  Micro batch size: ${MBS}"
echo "  Learning rate: ${LR}"
echo "============================================================================"
echo ""

if [ $node_rank -eq 0 ]; then
    # Master node: redirect output to both terminal and log file
    torchrun ${distributed_args[@]} ${script_args[@]} \
        2>&1 | tee ${LOG_DIR}/${node_rank}_python_master.log
else
    # Worker nodes: only save to log file
    torchrun ${distributed_args[@]} ${script_args[@]} \
        > ${LOG_DIR}/${node_rank}_python.log 2>&1
fi

echo ""
echo "============================================================================"
echo "Training Completed on Node ${node_rank}"
echo "============================================================================"
