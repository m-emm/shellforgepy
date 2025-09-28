#!/usr/bin/env python3
"""
Import fixer script that automatically corrects misplaced imports using robust AST-based regeneration.
"""

import ast
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Optional

# Import the analysis functions from import_fixer
from import_fixer import module_paths, find_misplaced_imports, analyze_file_imports
# Import the robust import regeneration functions
from import_regenerator import ImportSpec, regenerate_imports, ImportRegenerationError, ExoticImportError, find_invalid_imports

PROJECT_DIR = Path(__file__).parent.parent.resolve()


def determine_correct_import(incorrect_import: str, imported_names: List[str], adapter_exports: set) -> str:
    """Determine the correct import path based on what's being imported."""
    # If importing adapter functions, use shellforgepy.adapters.simple
    if any(name in adapter_exports for name in imported_names):
        return "shellforgepy.adapters.simple"
    
    # Otherwise, use the suggested correct import (which should be absolute shellforgepy.*)
    # Convert relative-style imports to absolute
    if not incorrect_import.startswith('shellforgepy.'):
        if '.' in incorrect_import:
            # Handle cases like "construct.alignment" -> "shellforgepy.construct.alignment"
            return f"shellforgepy.{incorrect_import}"
        else:
            # Handle single module names
            return f"shellforgepy.{incorrect_import}"
    
    return incorrect_import


def fix_invalid_import_in_file(file_path: str, imported_name: str, incorrect_module: str, correct_modules: List[str]) -> bool:
    """Fix an invalid import by changing the module it's imported from."""
    path = Path(file_path)
    
    if not path.exists():
        print(f"Warning: File {file_path} does not exist")
        return False
    
    if not correct_modules:
        print(f"Warning: No correct modules found for '{imported_name}'")
        return False
    
    # Choose the first correct module (could be enhanced with better heuristics)
    correct_module = correct_modules[0]
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Convert the correct module to relative import for simple.py files
        if file_path.endswith('/simple.py'):
            # Convert absolute path like 'shellforgepy.construct.named_part' to relative '.construct.named_part'
            if correct_module.startswith('shellforgepy.'):
                relative_module = '.' + correct_module[len('shellforgepy.'):]
            else:
                relative_module = correct_module
        else:
            relative_module = correct_module
        
        # Pattern to find and replace the specific import
        # Handle relative imports (from .module import name)
        if incorrect_module.startswith('.'):
            pattern = rf'^(\s*)from\s+{re.escape(incorrect_module)}\s+import\s+([^,\n]*{re.escape(imported_name)}[^,\n]*)'
            
            def replace_import(match):
                indent = match.group(1)
                import_list = match.group(2)
                # Split import list and remove the specific name
                imports = [imp.strip() for imp in import_list.split(',')]
                remaining_imports = [imp for imp in imports if imported_name not in imp]
                
                result = ""
                if remaining_imports:
                    # Keep the original import with remaining items
                    result += f"{indent}from {incorrect_module} import {', '.join(remaining_imports)}\n"
                
                # Add the corrected import
                result += f"{indent}from {relative_module} import {imported_name}"
                return result
            
            content = re.sub(pattern, replace_import, content, flags=re.MULTILINE)
        
        if content != original_content:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        else:
            print(f"Warning: No changes made to {file_path} for import {imported_name}")
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def fix_import_in_file(file_path: str, incorrect_import: str, correct_import: str, adapter_exports: set) -> bool:
    """Fix a specific import in a file by replacing incorrect with correct import."""
    path = Path(file_path)
    
    if not path.exists():
        print(f"Warning: File {file_path} does not exist")
        return False
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Parse the file to understand what's being imported
        try:
            tree = ast.parse(content)
            imported_names = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == incorrect_import:
                    for alias in node.names:
                        imported_names.append(alias.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == incorrect_import:
                            imported_names.append(alias.name)
            
            # Determine the actual correct import based on what's being imported
            actual_correct_import = determine_correct_import(incorrect_import, imported_names, adapter_exports)
            
        except:
            # If parsing fails, use the provided correct_import
            actual_correct_import = correct_import
        
        # Pattern to match import statements
        patterns = [
            # from incorrect_import import ...
            (rf'^(\s*)from\s+{re.escape(incorrect_import)}\s+import\s+(.+)$', 
             rf'\1from {actual_correct_import} import \2'),
            # import incorrect_import
            (rf'^(\s*)import\s+{re.escape(incorrect_import)}(\s|$)', 
             rf'\1import {actual_correct_import}\2'),
            # import incorrect_import as alias
            (rf'^(\s*)import\s+{re.escape(incorrect_import)}\s+as\s+(.+)$', 
             rf'\1import {actual_correct_import} as \2'),
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


def is_simple_py_file(file_path: str) -> bool:
    """Check if a file is a simple.py export facade that should be skipped."""
    return file_path.endswith('/simple.py')


def find_adapter_exports():
    """Find all exports from adapters/simple.py for special handling."""
    adapter_exports = set()
    
    adapter_simple_file = PROJECT_DIR / "src/shellforgepy/adapters/simple.py"
    
    if adapter_simple_file.exists():
        try:
            with open(adapter_simple_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the file to find __all__ exports
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == '__all__':
                            if isinstance(node.value, ast.List):
                                for elt in node.value.elts:
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                        adapter_exports.add(elt.value)
                                        
        except Exception as e:
            print(f"Warning: Could not parse {adapter_simple_file}: {e}")
    
    return adapter_exports


def main():
    print("🔍 Analyzing imports...")
    all_paths, paths_being_imported, file_imports = module_paths()
    misplaced_imports = find_misplaced_imports(all_paths, paths_being_imported, file_imports)
    
    # Also get invalid imports (importing non-existent names)
    invalid_imports = find_invalid_imports(all_paths, file_imports)
    
    if not misplaced_imports and not invalid_imports:
        print("✅ No import issues found!")
        return
    
    print(f"Found {len(misplaced_imports)} misplaced imports to fix.")
    print(f"Found {len(invalid_imports)} invalid imports to fix.")
    
    # Get adapter exports for special handling
    adapter_exports = find_adapter_exports()
    print(f"Found {len(adapter_exports)} adapter functions that should import from shellforgepy.adapters.simple")
    
    # Group invalid import fixes by file
    invalid_fixes_by_file = {}
    for item in invalid_imports:
        file_path = item['file']
        if file_path not in invalid_fixes_by_file:
            invalid_fixes_by_file[file_path] = []
        invalid_fixes_by_file[file_path].append({
            'imported_name': item['imported_name'],
            'incorrect_module': item['incorrect_module'],
            'correct_modules': item['correct_modules']
        })
    
    # Group misplaced import fixes by file to avoid duplicate processing, but skip simple.py files
    fixes_by_file = {}
    for item in misplaced_imports:
        if 'correct_import' in item:  # Skip ambiguous cases
            file_path = item['file']
            
            # Skip simple.py files - they are export facades and should not be "fixed"
            if is_simple_py_file(file_path):
                print(f"Skipping simple.py file: {file_path}")
                continue
                
            if file_path not in fixes_by_file:
                fixes_by_file[file_path] = []
            fixes_by_file[file_path].append({
                'incorrect': item['incorrect_import'],
                'correct': item['correct_import'],
                'reason': item['reason']
            })
    
    print(f"\n🔧 Applying invalid import fixes to {len(invalid_fixes_by_file)} files...")
    
    fixed_count = 0
    failed_count = 0
    
    # Process invalid import fixes first
    for file_path, fixes in invalid_fixes_by_file.items():
        print(f"\nProcessing invalid imports: {file_path}")
        
        for fix in fixes:
            correct_module_str = ', '.join(fix['correct_modules'][:3])  # Show first 3 options
            print(f"  Fixing: '{fix['imported_name']}' from {fix['incorrect_module']} → {correct_module_str}")
            
            if fix_invalid_import_in_file(file_path, fix['imported_name'], fix['incorrect_module'], fix['correct_modules']):
                fixed_count += 1
                print(f"    ✅ Fixed")
            else:
                failed_count += 1
                print(f"    ❌ Failed")
    
    print(f"\n🔧 Applying misplaced import fixes to {len(fixes_by_file)} files...")
    
    for file_path, fixes in fixes_by_file.items():
        print(f"\nProcessing: {file_path}")
        
        for fix in fixes:
            print(f"  Fixing: {fix['incorrect']} → {fix['correct']}")
            
            if fix_import_in_file(file_path, fix['incorrect'], fix['correct'], adapter_exports):
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