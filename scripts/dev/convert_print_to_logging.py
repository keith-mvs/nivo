#!/usr/bin/env python3
"""Convert print() statements to logging calls in the nivo codebase.

This script:
1. Adds logging import if not present
2. Adds logger = get_logger(__name__) if not present
3. Replaces print() with appropriate logging level

Run with: python scripts/dev/convert_print_to_logging.py --dry-run
"""

import re
import sys
from pathlib import Path
from typing import Tuple


def determine_log_level(content: str) -> str:
    """Determine appropriate log level based on content."""
    content_lower = content.lower()

    # Error patterns
    if any(x in content_lower for x in ['error', 'failed', 'exception', 'could not']):
        return 'error'

    # Warning patterns
    if any(x in content_lower for x in ['warning', 'warn', 'skipping', 'not available', 'not found', 'falling back']):
        return 'warning'

    # Debug patterns
    if any(x in content_lower for x in ['cache cleared', 'loading', 'loaded', 'deleting', 'deleted']):
        return 'debug'

    # Default to info
    return 'info'


def add_logging_import(content: str) -> str:
    """Add logging import if not present."""
    if 'from ..utils.logging_config import get_logger' in content:
        return content
    if 'from .utils.logging_config import get_logger' in content:
        return content
    if 'from ...utils.logging_config import get_logger' in content:
        return content

    # Find appropriate place to add import
    lines = content.split('\n')
    import_line = None
    last_from_import = 0

    for i, line in enumerate(lines):
        if line.startswith('from .'):
            last_from_import = i

    # Determine the right relative import path
    # This is a simple heuristic - may need adjustment for specific files
    import_stmt = 'from ..utils.logging_config import get_logger'

    if last_from_import > 0:
        # Insert after last relative import
        lines.insert(last_from_import + 1, import_stmt)
    else:
        # Find first import and add after
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                last_from_import = i
        if last_from_import > 0:
            lines.insert(last_from_import + 1, import_stmt)

    return '\n'.join(lines)


def add_logger_init(content: str) -> str:
    """Add logger initialization if not present."""
    if 'logger = get_logger(__name__)' in content:
        return content

    lines = content.split('\n')

    # Find where to insert logger init (after imports, before first class/def)
    insert_pos = 0
    in_docstring = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track docstrings
        if '"""' in stripped or "'''" in stripped:
            if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                in_docstring = not in_docstring

        if in_docstring:
            continue

        # Skip past imports and empty lines at top of file
        if stripped.startswith('import ') or stripped.startswith('from '):
            insert_pos = i + 1
        elif stripped == '' and insert_pos > 0:
            insert_pos = i + 1
        elif stripped.startswith('class ') or stripped.startswith('def '):
            break
        elif stripped.startswith('warnings.filterwarnings'):
            insert_pos = i + 1

    if insert_pos > 0:
        lines.insert(insert_pos, '\nlogger = get_logger(__name__)')

    return '\n'.join(lines)


def convert_print_to_logger(content: str) -> Tuple[str, int]:
    """Convert print() calls to logger calls."""
    count = 0

    # Pattern to match print statements
    # Handles: print(f"..."), print("..."), print(variable)
    pattern = r'(\s*)print\((.+)\)'

    def replace_print(match):
        nonlocal count
        indent = match.group(1)
        args = match.group(2)

        # Determine log level based on content
        level = determine_log_level(args)

        # Clean up the args (remove leading/trailing newlines in f-strings)
        args = args.strip()
        if args.startswith('f"\\n') or args.startswith("f'\\n"):
            args = 'f"' + args[4:]
        if args.startswith('"\\n') or args.startswith("'\\n"):
            args = '"' + args[3:]

        count += 1
        return f'{indent}logger.{level}({args})'

    new_content = re.sub(pattern, replace_print, content)
    return new_content, count


def process_file(filepath: Path, dry_run: bool = True) -> int:
    """Process a single file."""
    content = filepath.read_text(encoding='utf-8')

    # Skip if no print statements
    if 'print(' not in content:
        return 0

    # Skip test files
    if '/tests/' in str(filepath) or '\\tests\\' in str(filepath):
        return 0

    # Add imports
    new_content = add_logging_import(content)
    new_content = add_logger_init(new_content)

    # Convert prints
    new_content, count = convert_print_to_logger(new_content)

    if count > 0:
        print(f"  {filepath}: {count} print() -> logger")
        if not dry_run:
            filepath.write_text(new_content, encoding='utf-8')

    return count


def main():
    dry_run = '--dry-run' in sys.argv

    if dry_run:
        print("DRY RUN - no files will be modified\n")
    else:
        print("LIVE RUN - files will be modified\n")

    src_dir = Path(__file__).parent.parent.parent / 'src'

    total_count = 0
    for py_file in src_dir.rglob('*.py'):
        count = process_file(py_file, dry_run)
        total_count += count

    print(f"\nTotal: {total_count} print() statements {'would be ' if dry_run else ''}converted")

    if dry_run:
        print("\nRun without --dry-run to apply changes")


if __name__ == '__main__':
    main()
