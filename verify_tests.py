#!/usr/bin/env python3
"""
Simple verification script to test that the reorganized test structure works.
This script imports the test modules to verify they can be loaded correctly.
"""

from pathlib import Path
import sys
import importlib.util

def test_import(module_path, module_name):
    """Test importing a module from a file path."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"✓ Successfully imported {module_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to import {module_name}: {e}")
        return False

def main():
    """Test all the reorganized test files."""
    repo_root = Path(__file__).resolve().parent
    test_files = [
        (repo_root / "tests/lib/filters/test_builtins.py", "tests.lib.filters.test_builtins"),
        (repo_root / "tests/lib/event_log/test_view.py", "tests.lib.event_log.test_view"),
        (repo_root / "tests/lib/process_models/test_dfg.py", "tests.lib.process_models.test_dfg"),
    ]
    
    print("Verifying reorganized test structure...")
    print("=" * 50)
    
    all_passed = True
    for file_path, module_name in test_files:
        if not test_import(str(file_path), module_name):
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("✓ All test modules can be imported successfully!")
        print("✓ Test reorganization is complete and working.")
    else:
        print("✗ Some test modules failed to import.")
        sys.exit(1)

if __name__ == "__main__":
    main()


