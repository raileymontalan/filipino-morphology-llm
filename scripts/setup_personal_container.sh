#!/bin/bash
# Setup a personal NeMo container image
# This creates a container owned by you, avoiding permission issues

set -euo pipefail

echo "============================================================================"
echo "Personal NeMo Container Setup"
echo "============================================================================"
echo ""

# Verify enroot is available
if ! command -v enroot &> /dev/null; then
    echo "Error: enroot not found. Please install enroot first."
    exit 1
fi

# Source environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"

if [ -f "${PROJECT_DIR}/.env" ]; then
    set +u
    source "${PROJECT_DIR}/.env"
    set -u
else
    echo "Error: .env file not found"
    exit 1
fi

echo "Environment loaded:"
echo "  ENROOT_CACHE_PATH: ${ENROOT_CACHE_PATH}"
echo "  ENROOT_PATH: ${ENROOT_PATH}"
echo ""

# Check which NeMo version to download
NEMO_VERSION="${1:-25.11}"
CONTAINER_IMAGE="nvcr.io/nvidia/nemo:${NEMO_VERSION}"
CONTAINER_NAME="nemo_personal_${NEMO_VERSION}"

echo "Container Configuration:"
echo "  Image: ${CONTAINER_IMAGE}"
echo "  Name: ${CONTAINER_NAME}"
echo ""

# Check if container already exists
if enroot list | grep -q "^${CONTAINER_NAME}$"; then
    echo "✓ Container '${CONTAINER_NAME}' already exists"
    echo "To remove it, run: enroot remove ${CONTAINER_NAME}"
    exit 0
fi

echo "Downloading container image from ${CONTAINER_IMAGE}..."
echo "This may take 5-10 minutes..."
echo ""

# Import the container (download from NVIDIA registry)
enroot import docker://${CONTAINER_IMAGE}

echo ""
echo "============================================================================"
echo "✓ Container Downloaded Successfully!"
echo "============================================================================"
echo ""
echo "Container name: ${CONTAINER_NAME}"
echo ""
echo "To use this container in your jobs:"
echo "1. Update your PBS script to use CONTAINER_NAME='${CONTAINER_NAME}'"
echo "2. Or set it when submitting: qsub -v CONTAINER_NAME='${CONTAINER_NAME}' job.pbs"
echo ""
echo "To create an instance from this image:"
echo "  enroot create --name my_nemo_instance ${CONTAINER_NAME}"
echo ""
echo "To list all available containers:"
echo "  enroot list"
echo ""
echo "============================================================================"
