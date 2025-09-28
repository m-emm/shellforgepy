#!/usr/bin/env python3
"""
Import fixer script that automatically corrects misplaced imports using robust AST-based regeneration.
"""

import ast
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

# Import the analysis functions from import_fixer
from import_fixer import module_paths, find_misplaced_imports, analyze_file_imports
# Import the robust import regeneration functions
from import_regenerator import ImportSpec, regenerate_imports, ImportRegenerationError, ExoticImportError

PROJECT_DIR = Path(__file__).parent.parent.resolve()


def is_simple_py_file(file_path: str) -> bool:
    """Check if a file is a simple.py export facade that should be skipped."""
    return file_path.endswith('/simple.py')


def find_adapter_exports() -> Set[str]:
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


def discover_all_symbol_locations() -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Discover where all symbols are defined in the codebase.
    
    Returns:
        - symbol_to_module: Dict mapping symbol -> primary module location
        - symbol_duplicates: Dict mapping symbol -> list of all modules where it's defined
    """
    symbol_to_modules = defaultdict(list)  # symbol -> [module1, module2, ...]
    
    # Scan source directory
    src_dir = PROJECT_DIR / "src"
    for py_file in src_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        # Convert file path to module path
        relative_path = py_file.relative_to(src_dir)
        module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
        module_path = ".".join(module_parts)
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Find all defined symbols
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    symbol_to_modules[node.name].append(module_path)
                elif isinstance(node, ast.FunctionDef):
                    symbol_to_modules[node.name].append(module_path)
                elif isinstance(node, ast.Assign):
                    # Handle variable assignments at module level
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            symbol_to_modules[target.id].append(module_path)
                            
        except Exception as e:
            print(f"Warning: Could not parse {py_file}: {e}")
    
    # Convert to final format
    symbol_to_module = {}
    symbol_duplicates = {}
    
    for symbol, modules in symbol_to_modules.items():
        # Remove duplicates while preserving order
        unique_modules = list(dict.fromkeys(modules))
        
        if len(unique_modules) > 1:
            symbol_duplicates[symbol] = unique_modules
            # For duplicates, prefer the first non-adapter module, or just the first one
            primary_module = unique_modules[0]
            for module in unique_modules:
                if 'adapter' not in module:
                    primary_module = module
                    break
            symbol_to_module[symbol] = primary_module
        else:
            symbol_to_module[symbol] = unique_modules[0]
    
    return symbol_to_module, symbol_duplicates


def extract_symbols_used_in_file(file_path: str) -> Set[str]:
    """Extract all symbols that are actually used in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        used_symbols = set()
        
        # Find all Name nodes that are being loaded (used)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_symbols.add(node.id)
        
        return used_symbols
        
    except Exception as e:
        print(f"Warning: Could not analyze {file_path}: {e}")
        return set()


def generate_correct_imports(file_path: str, symbol_to_module: Dict[str, str], adapter_exports: Set[str]) -> List[ImportSpec]:
    """Generate the correct import specifications for a file."""
    
    # Skip simple.py files - they use relative imports
    if is_simple_py_file(file_path):
        return []
    
    # Get symbols actually used in this file
    used_symbols = extract_symbols_used_in_file(file_path)
    
    # Generate import specs
    import_specs = []
    
    for symbol in used_symbols:
        # Skip built-in symbols and common names
        if symbol in {'True', 'False', 'None', 'self', 'cls'}:
            continue
        
        # Special handling for adapter symbols
        if symbol in adapter_exports:
            import_specs.append(ImportSpec(
                module="shellforgepy.adapters.simple",
                name=symbol,
                alias=None
            ))
            continue
        
        # Find where this symbol is defined
        if symbol in symbol_to_module:
            module = symbol_to_module[symbol]
            import_specs.append(ImportSpec(
                module=module,
                name=symbol,
                alias=None
            ))
    
    return import_specs


def run_isort_on_file(file_path: str) -> None:
    """Run isort on a file to canonicalize imports."""
    try:
        subprocess.run(['isort', file_path], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: isort failed on {file_path}: {e}")


def fix_imports_in_file(file_path: str, symbol_to_module: Dict[str, str], adapter_exports: Set[str]) -> bool:
    """
    Fix imports in a single file using robust AST-based regeneration.
    
    Returns True if file was modified, False otherwise.
    """
    try:
        # Generate correct import specifications
        import_specs = generate_correct_imports(file_path, symbol_to_module, adapter_exports)
        
        if not import_specs:
            return False
        
        # Use the robust import regenerator
        regenerate_imports(file_path, import_specs)
        
        # Run isort to canonicalize the imports
        run_isort_on_file(file_path)
        
        return True
        
    except ExoticImportError as e:
        print(f"Skipping {file_path}: {e}")
        return False
    except ImportRegenerationError as e:
        print(f"Failed to fix imports in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error fixing {file_path}: {e}")
        return False


def validate_simple_py_imports(symbol_to_module: Dict[str, str], symbol_duplicates: Dict[str, List[str]]) -> List[str]:
    """Validate that simple.py files don't have broken imports by analyzing their AST."""
    issues = []
    
    # Find simple.py files
    src_dir = PROJECT_DIR / "src"
    for simple_file in src_dir.rglob("simple.py"):
        print(f"🔍 Validating {simple_file}...")
        
        try:
            with open(simple_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Check each import statement
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    module_name = node.module
                    
                    # Convert relative imports to absolute
                    if node.level > 0:  # Relative import
                        # For simple.py files, figure out the absolute module name
                        relative_path = simple_file.relative_to(src_dir)
                        current_package_parts = list(relative_path.parts[:-1])  # Remove 'simple.py'
                        
                        if node.level == 1:  # from .module import ...
                            # current_package_parts already includes 'shellforgepy', so just join with module_name
                            absolute_module = ".".join(current_package_parts + [module_name])
                        else:
                            # Handle deeper relative imports if needed
                            package_up = current_package_parts[:-node.level+1] if len(current_package_parts) >= node.level-1 else []
                            absolute_module = ".".join(package_up + [module_name])
                    else:
                        absolute_module = module_name
                    
                    # Check if each imported name exists in the target module
                    for alias in node.names:
                        imported_name = alias.name
                        
                        # Check if this symbol is defined in the expected module
                        if imported_name in symbol_to_module:
                            actual_module = symbol_to_module[imported_name]
                            if actual_module != absolute_module:
                                # Check if this is a duplicate symbol case
                                if imported_name in symbol_duplicates:
                                    all_locations = symbol_duplicates[imported_name]
                                    if absolute_module in all_locations:
                                        # The import is actually valid - it's one of the valid locations
                                        issue = f"⚠️ {simple_file}: '{imported_name}' imported from '{absolute_module}' - symbol defined in multiple places: {all_locations}"
                                        issues.append(issue)
                                        print(f"  {issue}")
                                    else:
                                        issue = f"❌ {simple_file}: '{imported_name}' imported from '{absolute_module}' but actually defined in '{actual_module}' (also defined in: {all_locations})"
                                        issues.append(issue)
                                        print(f"  {issue}")
                                else:
                                    issue = f"❌ {simple_file}: '{imported_name}' imported from '{absolute_module}' but actually defined in '{actual_module}'"
                                    issues.append(issue)
                                    print(f"  {issue}")
                        else:
                            # Check if the symbol exists at all
                            if imported_name not in {'__all__', '*'}:  # Skip special imports
                                issue = f"❌ {simple_file}: '{imported_name}' imported from '{absolute_module}' but not found anywhere"
                                issues.append(issue)
                                print(f"  {issue}")
            
            if not any(issue.startswith(f"❌ {simple_file}:") for issue in issues):
                print(f"  ✅ {simple_file} imports are valid")
                
        except Exception as e:
            issue = f"❌ Could not validate {simple_file}: {e}"
            issues.append(issue)
            print(f"  {issue}")
    
    return issues


def main():
    print("🔍 Analyzing imports...")
    all_paths, paths_being_imported, file_imports = module_paths()
    misplaced_imports = find_misplaced_imports(all_paths, paths_being_imported, file_imports)
    
    # Get adapter exports for special handling
    adapter_exports = find_adapter_exports()
    print(f"Found {len(adapter_exports)} adapter functions that should import from shellforgepy.adapters.simple")
    
    # Discover where all symbols are defined
    print("🗺️ Discovering symbol locations...")
    symbol_to_module, symbol_duplicates = discover_all_symbol_locations()
    print(f"Mapped {len(symbol_to_module)} symbols to their modules")
    if symbol_duplicates:
        print(f"⚠️ Found {len(symbol_duplicates)} symbols defined in multiple places")
        print("\n📋 Symbols defined in multiple modules:")
        for symbol, modules in symbol_duplicates.items():
            print(f"  '{symbol}' defined in: {modules}")

    print("\n🔍 Validating simple.py files...")
    simple_py_issues = validate_simple_py_imports(symbol_to_module, symbol_duplicates)
    
    if not misplaced_imports and not simple_py_issues:
        print("✅ No misplaced imports found!")
        return
    
    print(f"Found {len(misplaced_imports)} misplaced imports to fix.")
    
    # Group fixes by file to avoid duplicate processing, but skip simple.py files
    files_to_fix = set()
    for item in misplaced_imports:
        if 'correct_import' in item:  # Skip ambiguous cases
            file_path = item['file']
            
            # Skip simple.py files - they are export facades and should not be "fixed"
            if is_simple_py_file(file_path):
                print(f"Skipping simple.py file: {file_path}")
                continue
                
            files_to_fix.add(file_path)
    
    if not files_to_fix:
        print("✅ All misplaced imports are in simple.py files (which is correct)")
        return
    
    print(f"\n🔧 Fixing imports in {len(files_to_fix)} files...")
    
    fixed_count = 0
    failed_count = 0
    
    for file_path in files_to_fix:
        print(f"\nProcessing: {file_path}")
        
        if fix_imports_in_file(file_path, symbol_to_module, adapter_exports):
            fixed_count += 1
            print(f"    ✅ Fixed")
        else:
            failed_count += 1
            print(f"    ❌ Failed or no changes needed")
    
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
    
    # Report simple.py issues
    if simple_py_issues:
        print(f"\n⚠️  Simple.py validation issues ({len(simple_py_issues)}):")
        for issue in simple_py_issues:
            print(f"  {issue}")
        print("\n💡 Note: simple.py files must be fixed manually to import from correct modules.")
    
    if fixed_count > 0:
        print(f"\n🎉 Import fixing complete! Re-run import_fixer.py to verify the fixes.")


if __name__ == "__main__":
    main()