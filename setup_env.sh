#!/bin/bash
# Setup script for local development environment (data preprocessing only)
# Usage: bash setup_env.sh
#
# NOTE: This environment is for data preprocessing and analysis ONLY.
# For training, use the NeMo Framework Container (see setup_container.sh)

set -e  # Exit on error

echo "=============================================="
echo "Setting up Local Development Environment"
echo "=============================================="
echo ""
echo "NOTE: This is for data preprocessing/analysis only."
echo "      For training, use: bash setup_container.sh"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed"
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Using uv: $(which uv)"
echo "uv version: $(uv --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "env" ]; then
    echo "Creating virtual environment..."
    uv venv env --python 3.11
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source env/bin/activate

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Install dependencies
echo ""
echo "=============================================="
echo "Installing dependencies from requirements.txt"
echo "=============================================="
echo ""
echo "Installing packages for data preprocessing and analysis..."
uv pip install -r requirements.txt

echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="

# Verify key imports
python -c "
import pandas
import numpy
import transformers
import wandb
print('✓ pandas:', pandas.__version__)
print('✓ numpy:', numpy.__version__)
print('✓ transformers:', transformers.__version__)
print('✓ wandb:', wandb.__version__)
print()
print('All checks passed!')
"

echo ""
echo "=============================================="
echo "Setup completed successfully!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source env/bin/activate"
echo ""
echo "For data preprocessing:"
echo "  python src/data_preprocessing/prepare_seapile.py"
echo ""
echo "For training, use the NeMo container:"
echo "  bash setup_container.sh"
echo ""
