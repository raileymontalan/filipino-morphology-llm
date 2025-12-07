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

echo "→ Checking container image..."
if [ ! -f "$SIF_FILE" ]; then
    echo "  ✗ Error: Container image not found: $SIF_FILE"
    echo "  Please run setup first: bash setup_singularity.sh"
    exit 1
fi
echo "  ✓ Container image found: $SIF_FILE"

echo "→ Configuring mount points..."
# Default bind mount: current directory to /workspace
if [ -z "$BIND_MOUNTS" ]; then
    BIND_MOUNTS="$(pwd):/workspace"
    echo "  • $(pwd) → /workspace (default)"
else
    # Prepend current directory if not already in BIND_MOUNTS
    if [[ ! "$BIND_MOUNTS" == *"$(pwd)"* ]]; then
        BIND_MOUNTS="$(pwd):/workspace,$BIND_MOUNTS"
        echo "  • $(pwd) → /workspace (prepended)"
    fi
    # Parse and display all mounts
    IFS=',' read -ra MOUNTS <<< "$BIND_MOUNTS"
    for mount in "${MOUNTS[@]}"; do
        echo "  • $mount"
    done
fi

# Add additional mounts if specified
if [ ! -z "$EXTRA_MOUNTS" ]; then
    BIND_MOUNTS="$BIND_MOUNTS,$EXTRA_MOUNTS"
    echo "  • Extra mounts: $EXTRA_MOUNTS"
fi

echo "→ Detecting container runtime..."
# Run with Singularity/Apptainer
if command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
    echo "  ✓ Using Singularity: $(which singularity)"
elif command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
    echo "  ✓ Using Apptainer: $(which apptainer)"
else
    echo "  ✗ Error: Neither Singularity nor Apptainer found"
    exit 1
fi

echo "→ Starting container and running command..."
echo "  Command: $@"
echo ""

# Execute command in container
$CONTAINER_CMD exec \
    --nv \
    --bind "$BIND_MOUNTS" \
    --pwd /workspace \
    "$SIF_FILE" \
    "$@"
