#!/bin/bash
# Helper script to run commands inside the NeMo Framework container (Enroot)
#
# Usage: ./run_in_enroot.sh python script.py [args]
#        ./run_in_enroot.sh bash
#
# Example:
#   ./run_in_enroot.sh python scripts/run_cpt_gemma3_1b_container.py --max-steps 100

echo "→ Loading environment variables..."
# Load environment variables
if [ -f .env ]; then
    source .env
    echo "  ✓ Environment variables loaded from .env"
else
    echo "  ⚠ Warning: .env file not found, using existing environment"
fi

CONTAINER_NAME="nemo_framework"

echo "→ Checking if container exists..."
# Check if container exists
if ! enroot list | grep -q "^${CONTAINER_NAME}$"; then
    echo "  ✗ Error: Container '$CONTAINER_NAME' not found!"
    echo "  Please run setup first: bash setup_enroot.sh"
    exit 1
fi
echo "  ✓ Container '$CONTAINER_NAME' found"

echo "→ Configuring mount points..."
# Build mount points from BIND_MOUNTS
MOUNT_ARGS=""

# Always mount current directory as /workspace
MOUNT_ARGS="--mount $PWD:/workspace"
echo "  • $PWD → /workspace"

# Parse BIND_MOUNTS (format: /host/path:/container/path,/host2:/container2)
if [ ! -z "$BIND_MOUNTS" ]; then
    IFS=',' read -ra MOUNTS <<< "$BIND_MOUNTS"
    for mount in "${MOUNTS[@]}"; do
        MOUNT_ARGS="$MOUNT_ARGS --mount $mount"
        echo "  • $mount"
    done
fi

echo "→ Passing environment variables to container..."
# Pass through important environment variables
ENV_VARS=""
[ ! -z "$HF_HOME" ] && ENV_VARS="$ENV_VARS --env HF_HOME=$HF_HOME" && echo "  • HF_HOME"
[ ! -z "$HF_TOKEN" ] && ENV_VARS="$ENV_VARS --env HF_TOKEN=$HF_TOKEN" && echo "  • HF_TOKEN"
[ ! -z "$WANDB_API_KEY" ] && ENV_VARS="$ENV_VARS --env WANDB_API_KEY=$WANDB_API_KEY" && echo "  • WANDB_API_KEY"
[ ! -z "$WANDB_DIR" ] && ENV_VARS="$ENV_VARS --env WANDB_DIR=$WANDB_DIR" && echo "  • WANDB_DIR"
[ ! -z "$CUDA_VISIBLE_DEVICES" ] && ENV_VARS="$ENV_VARS --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" && echo "  • CUDA_VISIBLE_DEVICES"

echo "→ Starting container and running command..."
echo "  Command: $@"
echo ""

# Run command in container
enroot start \
    --root \
    --rw \
    $MOUNT_ARGS \
    $ENV_VARS \
    "$CONTAINER_NAME" \
    "$@"
