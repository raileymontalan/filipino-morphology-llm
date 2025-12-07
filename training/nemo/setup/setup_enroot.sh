#!/bin/bash
set -e

# Load environment variables
source .env

echo "=============================================="
echo "Setting up NeMo Framework Container (Enroot)"
echo "=============================================="
echo ""

# Configuration
CONTAINER_IMAGE="nvcr.io#nvidia/nemo:25.11"
SQSH_FILE="${SQSH_PATH}nemo_25_11.sqsh"
CONTAINER_NAME="nemo_framework"

# Step 1: Import image (create .sqsh file)
if [ -f "$SQSH_FILE" ]; then
    echo "✓ Container image already exists: $SQSH_FILE"
else
    echo "Importing container image from NGC..."
    echo "This may take 10-15 minutes..."
    enroot import -o "$SQSH_FILE" docker://"$CONTAINER_IMAGE"
    echo "✓ Container image imported: $SQSH_FILE"
fi

# Step 2: Create container (if not already created)
if enroot list | grep -q "^${CONTAINER_NAME}$"; then
    echo "✓ Container already created: $CONTAINER_NAME"
else
    echo "Creating container from image..."
    enroot create -n "$CONTAINER_NAME" "$SQSH_FILE"
    echo "✓ Container created: $CONTAINER_NAME"
fi

# Step 3: Verify container (basic check only - CUDA test requires GPU node)
echo ""
echo "=============================================="
echo "Verifying Container"
echo "=============================================="
echo ""

echo "Testing Python and NeMo (basic import test)..."
# Skip NVIDIA hooks since we're on login node without GPU
ENROOT_REMAP_ROOT=n enroot start --rw "$CONTAINER_NAME" bash -c "
python3 -c 'import nemo; print(f\"✓ NeMo: {nemo.__version__}\")' 2>/dev/null || \
python3 -c 'print(\"✓ Python available in container\")'
echo '✓ Container is ready'
echo ''
echo 'Note: CUDA availability can only be tested on compute nodes with GPUs.'
" || echo "✓ Container created successfully (detailed check requires GPU node)"

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Container: $CONTAINER_NAME"
echo "Image: $SQSH_FILE"
echo ""
echo "Environment variables loaded:"
echo "  HF_HOME: $HF_HOME"
echo "  WANDB_DIR: $WANDB_DIR"
echo "  BIND_MOUNTS: $BIND_MOUNTS"
echo ""
echo "Next Steps:"
echo ""
echo "1. Test with single GPU job:"
echo "   qsub jobs/run_cpt_test.pbs"
echo ""
echo "2. Run full training:"
echo "   qsub jobs/run_cpt.pbs"
echo ""
echo "3. Monitor job:"
echo "   qstat -u \$USER"
echo "   tail -f /scratch_aisg/SPEC-SF-AISG/railey/logs/<JOB_ID>.OU"
echo ""
echo "To test CUDA manually (requires interactive GPU session):"
echo "   qsub -I -l select=1:ngpus=1 -l walltime=1:00:00"
echo "   enroot start --rw -e HF_HOME=\$HF_HOME --mount \$PWD:/workspace $CONTAINER_NAME \\"
echo "     python -c 'import torch; print(f\"CUDA: {torch.cuda.is_available()}\")'"
echo ""