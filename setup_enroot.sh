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
SQSH_FILE="${ENROOT_PATH}nemo_25_11.sqsh"
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

# Step 3: Verify container
echo ""
echo "=============================================="
echo "Verifying Container"
echo "=============================================="
echo ""

echo "Testing Python and NeMo..."
enroot start --root --rw "$CONTAINER_NAME" python3 -c "
import torch
import nemo
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
print(f'✓ NeMo: {nemo.__version__}')
"

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
echo "To run Python scripts in the container:"
echo ""
echo "  # Direct enroot command:"
IFS=',' read -ra MOUNTS <<< "$BIND_MOUNTS"
MOUNT_STR="  enroot start --root --rw \\"
MOUNT_STR="$MOUNT_STR\n    --mount $PWD:/workspace \\"
for mount in "${MOUNTS[@]}"; do
    MOUNT_STR="$MOUNT_STR\n    --mount $mount \\"
done
echo -e "$MOUNT_STR"
echo "    $CONTAINER_NAME \\"
echo "    python /workspace/scripts/run_cpt_gemma3_1b_container.py --max-steps 100"
echo ""
echo "  # Or use the helper script (recommended):"
echo "  ./run_in_enroot.sh python scripts/run_cpt_gemma3_1b_container.py --max-steps 100"
echo ""