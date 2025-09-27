#!/usr/bin/env python3
"""
Import fixer script that automatically corrects misplaced imports.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Import the analysis functions from import_fixer
from import_fixer import module_paths, find_misplaced_imports, analyze_file_imports

PROJECT_DIR = Path(__file__).parent.parent.resolve()


def fix_import_in_file(file_path: str, incorrect_import: str, correct_import: str) -> bool:
    """Fix a specific import in a file by replacing incorrect with correct import."""
    path = Path(file_path)
    
    if not path.exists():
        print(f"Warning: File {file_path} does not exist")
        return False
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern to match import statements
        patterns = [
            # from incorrect_import import ...
            (rf'^(\s*)from\s+{re.escape(incorrect_import)}\s+import\s+(.+)$', 
             rf'\1from {correct_import} import \2'),
            # import incorrect_import
            (rf'^(\s*)import\s+{re.escape(incorrect_import)}(\s|$)', 
             rf'\1import {correct_import}\2'),
            # import incorrect_import as alias
            (rf'^(\s*)import\s+{re.escape(incorrect_import)}\s+as\s+(.+)$', 
             rf'\1import {correct_import} as \2'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        if content != original_content:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        else:
            print(f"Warning: No changes made to {file_path} for import {incorrect_import}")
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    print("🔍 Analyzing imports...")
    all_paths, paths_being_imported, file_imports = module_paths()
    misplaced_imports = find_misplaced_imports(all_paths, paths_being_imported, file_imports)
    
    if not misplaced_imports:
        print("✅ No misplaced imports found!")
        return
    
    print(f"Found {len(misplaced_imports)} misplaced imports to fix.")
    
    # Group fixes by file to avoid duplicate processing
    fixes_by_file = {}
    for item in misplaced_imports:
        if 'correct_import' in item:  # Skip ambiguous cases
            file_path = item['file']
            if file_path not in fixes_by_file:
                fixes_by_file[file_path] = []
            fixes_by_file[file_path].append({
                'incorrect': item['incorrect_import'],
                'correct': item['correct_import'],
                'reason': item['reason']
            })
    
    print(f"\n🔧 Applying fixes to {len(fixes_by_file)} files...")
    
    fixed_count = 0
    failed_count = 0
    
    for file_path, fixes in fixes_by_file.items():
        print(f"\nProcessing: {file_path}")
        
        for fix in fixes:
            print(f"  Fixing: {fix['incorrect']} → {fix['correct']}")
            
            if fix_import_in_file(file_path, fix['incorrect'], fix['correct']):
                fixed_count += 1
                print(f"    ✅ Fixed")
            else:
                failed_count += 1
                print(f"    ❌ Failed")
    
    print(f"\n📊 Summary:")
    print(f"  ✅ Successfully fixed: {fixed_count}")
    print(f"  ❌ Failed to fix: {failed_count}")
    
    # Show remaining ambiguous cases
    ambiguous_cases = [item for item in misplaced_imports if 'possible_corrections' in item]
    if ambiguous_cases:
        print(f"\n⚠️  Ambiguous cases requiring manual review ({len(ambiguous_cases)}):")
        for item in ambiguous_cases:
            print(f"  File: {item['file']}")
            print(f"    Import: {item['incorrect_import']}")
            print(f"    Options: {item['possible_corrections']}")
    
    if fixed_count > 0:
        print(f"\n🎉 Import fixing complete! Re-run import_fixer.py to verify the fixes.")


if __name__ == "__main__":
    main()