"""
Centralized path setup for the filipino-morphology-llm project.

Use this instead of manual sys.path manipulation in every file.

Usage:
    # At the top of any script or test file
    from setup_paths import setup_project_paths
    setup_project_paths()

    # Now you can import from src modules
    from tokenization import MorphologyAwarePatokProcessor
    from evaluation import downstream
"""

import sys
from pathlib import Path


def setup_project_paths():
    """
    Add the project's src directory to Python path.

    This function:
    1. Finds the project root (looks for setup.py)
    2. Adds src/ directory to sys.path if not already present
    3. Works from any directory within the project
    """
    # Get the directory containing this file (src/)
    src_dir = Path(__file__).parent.resolve()

    # Add to path if not already there
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    return src_dir


def get_project_root():
    """
    Get the project root directory.

    Returns:
        Path: Path to project root (directory containing setup.py)
    """
    # Start from this file's directory and search upward
    current = Path(__file__).parent.resolve()

    while current != current.parent:
        if (current / "setup.py").exists():
            return current
        current = current.parent

    # Fallback: assume src is one level below root
    return Path(__file__).parent.parent.resolve()


# Auto-setup when imported
setup_project_paths()
