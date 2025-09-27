#!/usr/bin/env python3
"""
Fix relative imports by converting them to absolute imports.
Excludes simple.py files which are special.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set

PROJECT_DIR = Path(__file__).parent.parent.resolve()
SRC_DIR = PROJECT_DIR / "src"
TESTS_DIR = PROJECT_DIR / "tests"


class RelativeImportFinder(ast.NodeVisitor):
    """AST visitor to find relative imports in Python files."""
    
    def __init__(self):
        self.relative_imports = []
    
    def visit_ImportFrom(self, node):
        """Handle 'from module import name' statements."""
        if node.level > 0:  # This is a relative import
            self.relative_imports.append({
                'level': node.level,
                'module': node.module,
                'names': [alias.name for alias in node.names],
                'asnames': [(alias.name, alias.asname) for alias in node.names],
                'lineno': node.lineno,
                'col_offset': node.col_offset
            })
        self.generic_visit(node)


def find_relative_imports(file_path: Path) -> List[Dict]:
    """Find all relative imports in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        finder = RelativeImportFinder()
        finder.visit(tree)
        
        return finder.relative_imports
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return []


def get_module_path_from_file(file_path: Path, src_dir: Path) -> str:
    """Convert a file path to its module path."""
    try:
        relative_path = file_path.relative_to(src_dir)
        return ".".join(relative_path.with_suffix("").parts)
    except ValueError:
        # File is not under src_dir, might be in tests
        return None


def resolve_relative_import(file_path: Path, src_dir: Path, level: int, module: str) -> str:
    """Resolve a relative import to its absolute module path."""
    # Get the current module path
    current_module = get_module_path_from_file(file_path, src_dir)
    if not current_module:
        return None
    
    # Split into parts
    current_parts = current_module.split(".")
    
    # Go up 'level' directories
    if level > len(current_parts):
        return None
    
    # Calculate base path
    base_parts = current_parts[:-level] if level > 0 else current_parts
    
    # Add the imported module
    if module:
        target_parts = base_parts + module.split(".")
    else:
        target_parts = base_parts
    
    return ".".join(target_parts)


def fix_relative_imports_in_file(file_path: Path, src_dir: Path) -> bool:
    """Fix all relative imports in a single file."""
    # Skip simple.py files
    if file_path.name == "simple.py":
        print(f"Skipping {file_path} (simple.py files are special)")
        return False
    
    relative_imports = find_relative_imports(file_path)
    if not relative_imports:
        return False
    
    print(f"Processing {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Track changes
        changes_made = False
        
        # Process imports in reverse order (by line number) to avoid line number shifts
        for imp in sorted(relative_imports, key=lambda x: x['lineno'], reverse=True):
            line_idx = imp['lineno'] - 1  # Convert to 0-based index
            if line_idx >= len(lines):
                continue
                
            original_line = lines[line_idx]
            
            # Resolve the relative import
            absolute_module = resolve_relative_import(file_path, src_dir, imp['level'], imp['module'])
            if not absolute_module:
                print(f"  Warning: Could not resolve relative import on line {imp['lineno']}")
                continue
            
            # Build the new import statement
            import_names = []
            for name, asname in imp['asnames']:
                if asname:
                    import_names.append(f"{name} as {asname}")
                else:
                    import_names.append(name)
            
            if len(import_names) == 1:
                new_line = f"from {absolute_module} import {import_names[0]}\n"
            else:
                # Multi-line import
                indent = original_line[:len(original_line) - len(original_line.lstrip())]
                new_line = f"from {absolute_module} import (\n"
                for i, name in enumerate(import_names):
                    if i == len(import_names) - 1:
                        new_line += f"{indent}    {name}\n{indent})\n"
                    else:
                        new_line += f"{indent}    {name},\n"
            
            print(f"  Line {imp['lineno']}: {original_line.strip()}")
            print(f"         -> {new_line.strip()}")
            
            lines[line_idx] = new_line
            changes_made = True
        
        if changes_made:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False
    
    return False


def main():
    print("🔍 Finding files with relative imports...")
    
    files_to_process = []
    
    # Scan source files
    for file_path in SRC_DIR.rglob("*.py"):
        if file_path.name != "simple.py":  # Skip simple.py files
            relative_imports = find_relative_imports(file_path)
            if relative_imports:
                files_to_process.append(file_path)
    
    # Also scan test files
    if TESTS_DIR.exists():
        for file_path in TESTS_DIR.rglob("*.py"):
            if file_path.name != "simple.py":  # Skip simple.py files
                relative_imports = find_relative_imports(file_path)
                if relative_imports:
                    files_to_process.append(file_path)
    
    if not files_to_process:
        print("✅ No relative imports found (excluding simple.py files)!")
        return
    
    print(f"Found {len(files_to_process)} files with relative imports to fix:")
    for file_path in files_to_process:
        print(f"  {file_path}")
    
    print(f"\n🔧 Converting relative imports to absolute imports...")
    
    fixed_count = 0
    failed_count = 0
    
    for file_path in files_to_process:
        if fix_relative_imports_in_file(file_path, SRC_DIR):
            fixed_count += 1
        else:
            failed_count += 1
    
    print(f"\n📊 Summary:")
    print(f"  ✅ Files successfully fixed: {fixed_count}")
    print(f"  ❌ Files that couldn't be fixed: {failed_count}")
    print(f"  🔒 Files skipped (simple.py): excluded by design")
    
    if fixed_count > 0:
        print(f"\n🎉 Relative import fixing complete!")


if __name__ == "__main__":
    main()