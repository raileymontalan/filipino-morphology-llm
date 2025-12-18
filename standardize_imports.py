#!/usr/bin/env python3
"""
Standardize all sys.path manipulations across the project.

This script:
1. Finds all Python files with sys.path manipulation
2. Updates them to use consistent pattern: Path(__file__).parent.parent / 'src'
3. Reports what was changed
"""

import re
from pathlib import Path

# Standard import pattern to use
STANDARD_PATTERN = """from pathlib import Path
import sys
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
"""

SCRIPT_PATTERN = """from pathlib import Path
import sys
# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
"""


def get_file_context(filepath):
    """Determine if file is a test, script, or other."""
    if "tests/" in str(filepath):
        return "test"
    elif "scripts/" in str(filepath):
        return "script"
    elif "src/" in str(filepath):
        return "src"
    return "other"


def update_sys_path(filepath):
    """Update a file's sys.path manipulation to standard pattern."""
    with open(filepath, "r") as f:
        content = f.read()

    original = content
    context = get_file_context(filepath)

    # Skip if already has setup_paths
    if "from setup_paths import" in content:
        return False, "Uses setup_paths (no change needed)"

    # Patterns to find and remove
    old_patterns = [
        (r'import sys\s*\n\s*sys\.path\.insert\(0,\s*[\'"]src[\'"]\s*\)', "simple src"),
        (r'sys\.path\.insert\(0,\s*[\'"]\.?[\'"]\s*\)', "dot path"),
        (
            r'import sys\s*\n.*\n\s*sys\.path\.insert\(0,\s*os\.path\.join\(os\.getcwd\(\),\s*[\'"]src[\'"]\)\)',
            "os.path.join",
        ),
        (
            r'sys\.path\.insert\(0,\s*str\(Path\(__file__\)\.parent\.parent\s*/\s*[\'"]src[\'"]\)\)',
            "existing standard",
        ),
        (
            r'sys\.path\.insert\(0,\s*str\(project_root\s*/\s*[\'"]src[\'"]\)\)',
            "project_root",
        ),
        (
            r'sys\.path\.insert\(0,\s*[\'"]\/workspace\/src[\'"]\)',
            "hardcoded /workspace",
        ),
    ]

    found_pattern = None
    for pattern, name in old_patterns:
        if re.search(pattern, content, re.MULTILINE | re.DOTALL):
            found_pattern = name
            # Remove the old pattern
            content = re.sub(pattern, "", content, flags=re.MULTILINE | re.DOTALL)
            break

    if not found_pattern:
        return False, "No sys.path found"

    # Find where to insert new import
    lines = content.split("\n")
    insert_idx = 0

    # Skip shebang
    if lines and lines[0].startswith("#!"):
        insert_idx = 1

    # Skip module docstring
    in_docstring = False
    quote_type = None
    for i in range(insert_idx, min(len(lines), 20)):  # Check first 20 lines
        line = lines[i].strip()

        if not in_docstring:
            if line.startswith('"""') or line.startswith("'''"):
                quote_type = '"""' if line.startswith('"""') else "'''"
                in_docstring = True
                if line.endswith(quote_type) and len(line) > 3:
                    # Single line docstring
                    insert_idx = i + 1
                    break
        else:
            if quote_type in line:
                insert_idx = i + 1
                break

        # If we hit an import, stop
        if not in_docstring and (line.startswith("import ") or line.startswith("from ")):
            insert_idx = i
            break

    # Insert the standard pattern
    if context in ["test", "other"]:
        pattern_to_use = STANDARD_PATTERN.strip() + "\n"
    else:
        pattern_to_use = SCRIPT_PATTERN.strip() + "\n"

    lines.insert(insert_idx, pattern_to_use)
    content = "\n".join(lines)

    # Clean up multiple blank lines
    content = re.sub(r"\n\n\n+", "\n\n", content)
    content = re.sub(r"\n+import sys", "\nimport sys", content)

    if content != original:
        with open(filepath, "w") as f:
            f.write(content)
        return True, f"Updated (was: {found_pattern})"

    return False, "No changes made"


def main():
    print("=" * 70)
    print("Standardizing sys.path manipulation across project")
    print("=" * 70)
    print()

    # Files to update
    files_to_check = []

    # All test files
    test_dir = Path("tests")
    if test_dir.exists():
        files_to_check.extend(test_dir.glob("test_*.py"))

    # Key script files
    script_files = [
        "scripts/build_tokenizer_expansions.py",
        "scripts/run_analysis.py",
        "scripts/run_evaluation.py",
        "scripts/generate_benchmarks.py",
    ]
    files_to_check.extend([Path(f) for f in script_files if Path(f).exists()])

    # src files that might have sys.path
    src_files = [
        "src/evaluation/downstream.py",
        "src/analysis/tokenization/compare_tokenizers.py",
    ]
    files_to_check.extend([Path(f) for f in src_files if Path(f).exists()])

    # Update each file
    updated_count = 0
    for filepath in sorted(files_to_check):
        updated, msg = update_sys_path(filepath)
        status = "✓" if updated else "○"
        print(f"{status} {filepath.name:40} {msg}")
        if updated:
            updated_count += 1

    print()
    print("=" * 70)
    print(f"Updated {updated_count} file(s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
