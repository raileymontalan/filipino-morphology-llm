#!/bin/bash
# Setup script for NeMo Framework Container (Singularity/Apptainer)
# This script pulls and prepares the NeMo Framework container for training
#
# Usage: bash setup_singularity.sh [CONTAINER_CACHEDIR]
#   or:  CONTAINER_CACHEDIR=/path/to/cache bash setup_singularity.sh
#
# Arguments:
#   CONTAINER_CACHEDIR - Directory for container cache (required)
#               Should be on a filesystem with plenty of space (~20GB)
#
# Example:
#   bash setup_singularity.sh /scratch/myuser/container_cache

set -e  # Exit on error

echo "=============================================="
echo "Setting up NeMo Framework Container"
echo "=============================================="
echo ""

# Configuration
CONTAINER_IMAGE="nvcr.io/nvidia/nemo:25.11"
CONTAINER_NAME="nemo_framework"
SIF_FILE="nemo_framework_25.11.sif"

# Get cache directory from argument or environment variable
if [ -n "$1" ]; then
    CONTAINER_CACHEDIR="$1"
elif [ -n "$CONTAINER_CACHEDIR" ]; then
    CONTAINER_CACHEDIR="$CONTAINER_CACHEDIR"
else
    echo "Error: CONTAINER_CACHEDIR not specified!"
    echo ""
    echo "Usage:"
    echo "  bash setup_singularity.sh /path/to/cache"
    echo "  or"
    echo "  CONTAINER_CACHEDIR=/path/to/cache bash setup_singularity.sh"
    echo ""
    echo "Example:"
    echo "  bash setup_singularity.sh /scratch/\$USER/container_cache"
    echo ""
    echo "The cache directory should be on a filesystem with ~20GB free space."
    exit 1
fi

# Expand ~ and environment variables in path
CONTAINER_CACHEDIR=$(eval echo "$CONTAINER_CACHEDIR")

# Set cache directories to avoid filling up home directory
export APPTAINER_CACHEDIR="$CONTAINER_CACHEDIR"
export SINGULARITY_CACHEDIR="$CONTAINER_CACHEDIR"
export APPTAINER_TMPDIR="$CONTAINER_CACHEDIR/tmp"
export SINGULARITY_TMPDIR="$CONTAINER_CACHEDIR/tmp"

# Create cache directories
mkdir -p "$CONTAINER_CACHEDIR/tmp"
echo "✓ Using cache directory: $CONTAINER_CACHEDIR"

# Check if Singularity/Apptainer is available
if command -v singularity &> /dev/null; then
    CONTAINER_RUNTIME="singularity"
    echo "✓ Found Singularity: $(which singularity)"
elif command -v apptainer &> /dev/null; then
    CONTAINER_RUNTIME="apptainer"
    echo "✓ Found Apptainer: $(which apptainer)"
else
    echo "✗ Error: Neither Singularity nor Apptainer found"
    echo "  Please install one of these container runtimes"
    exit 1
fi

echo ""
echo "=============================================="
echo "Step 1: Pulling NeMo Framework Container"
echo "=============================================="
echo "Container: $CONTAINER_IMAGE"
echo ""

# Check if SIF file already exists
if [ -f "$SIF_FILE" ]; then
    echo "✓ Container image already exists: $SIF_FILE"
    echo "  To re-download, delete the file and run this script again"
else
    echo "Pulling container (this may take 10-15 minutes)..."
    $CONTAINER_RUNTIME pull "$SIF_FILE" "docker://$CONTAINER_IMAGE"
    echo "✓ Container pulled successfully"
fi

echo ""
echo "=============================================="
echo "Step 2: Verifying Container"
echo "=============================================="
echo ""

# Verify PyTorch and CUDA
echo "Checking PyTorch and CUDA availability..."
$CONTAINER_RUNTIME exec "$SIF_FILE" python -c "
import torch
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA version: {torch.version.cuda}')
    print(f'✓ Number of GPUs: {torch.cuda.device_count()}')
"

echo ""
echo "Checking NeMo Framework..."
$CONTAINER_RUNTIME exec "$SIF_FILE" python -c "
try:
    import nemo
    print(f'✓ NeMo version: {nemo.__version__}')
    
    import nemo.collections.llm as llm
    print('✓ NeMo LLM collections available')
    
    from megatron.core import parallel_state
    print('✓ Megatron-Core available')
    
    print('\n✓ All core components verified!')
except Exception as e:
    print(f'✗ Error: {e}')
    exit(1)
"

echo ""
echo "=============================================="
echo "Step 3: Creating Helper Scripts"
echo "=============================================="
echo ""

# Create run_in_singularity.sh helper script
cat > run_in_singularity.sh << 'EOF'
#!/bin/bash
# Helper script to run commands inside the NeMo Framework container (Singularity/Apptainer)
#
# Usage: ./run_in_singularity.sh python script.py [args]
#        ./run_in_singularity.sh bash
#
# Environment Variables:
#   BIND_MOUNTS - Comma-separated list of mount points (optional)
#                 Format: /host/path:/container/path
#   EXTRA_MOUNTS - Additional mounts to append (optional)
#
# Example:
#   BIND_MOUNTS="/scratch/$USER:/scratch" ./run_in_singularity.sh python train.py

SIF_FILE="nemo_framework_25.11.sif"

# Default bind mount: current directory to /workspace
if [ -z "$BIND_MOUNTS" ]; then
    BIND_MOUNTS="$(pwd):/workspace"
else
    # Prepend current directory if not already in BIND_MOUNTS
    if [[ ! "$BIND_MOUNTS" == *"$(pwd)"* ]]; then
        BIND_MOUNTS="$(pwd):/workspace,$BIND_MOUNTS"
    fi
fi

# Add additional mounts if specified
if [ ! -z "$EXTRA_MOUNTS" ]; then
    BIND_MOUNTS="$BIND_MOUNTS,$EXTRA_MOUNTS"
fi

# Run with Singularity/Apptainer
if command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
elif command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
else
    echo "Error: Neither Singularity nor Apptainer found"
    exit 1
fi

# Execute command in container
$CONTAINER_CMD exec \
    --nv \
    --bind "$BIND_MOUNTS" \
    --pwd /workspace \
    "$SIF_FILE" \
    "$@"
EOF

chmod +x run_in_singularity.sh
echo "✓ Created helper script: run_in_singularity.sh"

# Create SLURM job submission template
cat > jobs/submit_cpt_singularity.sh << 'EOF'
#!/bin/bash
#PBS -N gemma3_1b_cpt
#PBS -l select=1:ncpus=64:ngpus=8:mem=500gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o logs/${PBS_JOBID}.log

# Change to working directory
cd $PBS_O_WORKDIR

# Load modules if needed
# module load singularity

# ============================================================================
# CONFIGURATION - Modify these variables before submitting
# ============================================================================

# Container and script paths
SIF_FILE="nemo_framework_25.11.sif"
SCRIPT="scripts/run_cpt_gemma3_1b_container.py"

# Bind mounts - IMPORTANT: Update these paths for your system!
# Format: /host/path:/container/path
# Example: BIND_MOUNTS="$(pwd):/workspace,/scratch/$USER:/scratch"
BIND_MOUNTS="$(pwd):/workspace"
# Uncomment and modify the line below to add your data directories:
# BIND_MOUNTS="$BIND_MOUNTS,/path/to/your/data:/data"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO

# WandB API Key - REQUIRED: Set this before submitting!
# Option 1: Set in your environment before submitting
# Option 2: Uncomment and set directly (NOT RECOMMENDED for shared code)
# export WANDB_API_KEY="your-wandb-key-here"
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY is not set!"
    echo "Please set it before submitting:"
    echo "  export WANDB_API_KEY='your-key-here'"
    echo "  qsub jobs/submit_cpt_singularity.sh"
    exit 1
fi

# ============================================================================

# Run training in container
singularity exec \
    --nv \
    --bind "$BIND_MOUNTS" \
    --pwd /workspace \
    "$SIF_FILE" \
    python "$SCRIPT" \
    --devices 8 \
    --max-steps 100
EOF

chmod +x jobs/submit_cpt_singularity.sh
echo "✓ Created SLURM job template: jobs/submit_cpt_singularity.sh"

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Container: $SIF_FILE"
echo "Cache directory: $CONTAINER_CACHEDIR"
echo ""
echo "Next steps:"
echo ""
echo "1. Test the container:"
echo "   ./run_in_singularity.sh python --version"
echo ""
echo "2. Configure bind mounts (if needed):"
echo "   Edit run_in_singularity.sh or use environment variables:"
echo "   BIND_MOUNTS=\"/path/to/data:/data\" ./run_in_singularity.sh python --version"
echo ""
echo "3. Prepare your data:"
echo "   python src/data_preprocessing/prepare_seapile.py"
echo ""
echo "4. Set your WandB API key:"
echo "   export WANDB_API_KEY='your-key-here'"
echo ""
echo "5. Run training (100 steps for testing):"
echo "   ./run_in_singularity.sh python scripts/run_cpt_gemma3_1b_container.py --max-steps 100"
echo ""
echo "6. Or submit to SLURM (after editing the config in the file):"
echo "   export WANDB_API_KEY='your-key-here'"
echo "   qsub jobs/submit_cpt_singularity.sh"
echo ""
echo "IMPORTANT: Before sharing this code, ensure:"
echo "  - WANDB_API_KEY is set in your environment (not in files)"
echo "  - Bind mount paths in jobs/submit_cpt_singularity.sh are configured"
echo ""
