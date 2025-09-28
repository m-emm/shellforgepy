#!/usr/bin/env python3
"""
Advanced import regenerator that:
1. Parses all imports using AST
2. Builds an internal model of what should be imported from where
3. Removes all existing imports from files
4. Regenerates imports from the internal model (flat, single-line)
5. Uses isort to canonicalize them
"""

import ast
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict

PROJECT_DIR = Path(__file__).parent.parent.resolve()
SRC_DIR = PROJECT_DIR / "src"

class ImportAnalyzer(ast.NodeVisitor):
    """AST visitor to extract all import usage and find where symbols are defined."""
    
    def __init__(self):
        self.imports_used = []  # What this file imports
        self.symbols_defined = []  # What this file defines (classes, functions, etc.)
        self.symbols_used = []  # What symbols are used in the code
        
    def visit_Import(self, node):
        for alias in node.names:
            self.imports_used.append({
                'type': 'import',
                'module': alias.name,
                'asname': alias.asname,
                'lineno': node.lineno
            })
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:  # Skip relative imports without module name
            for alias in node.names:
                self.imports_used.append({
                    'type': 'from_import',
                    'module': node.module,
                    'name': alias.name,
                    'asname': alias.asname,
                    'level': node.level,
                    'lineno': node.lineno
                })
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        self.symbols_defined.append({
            'type': 'class',
            'name': node.name,
            'lineno': node.lineno
        })
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        self.symbols_defined.append({
            'type': 'function',
            'name': node.name,
            'lineno': node.lineno
        })
        self.generic_visit(node)
        
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):  # Symbol is being used, not assigned
            self.symbols_used.append({
                'name': node.id,
                'lineno': node.lineno
            })
        self.generic_visit(node)


def discover_all_modules_and_symbols():
    """Discover all modules and what symbols they define."""
    modules = {}  # module_path -> {symbols: [...], file_path: ...}
    
    # Scan source directory
    for py_file in SRC_DIR.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        # Convert file path to module path
        relative_path = py_file.relative_to(SRC_DIR)
        module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
        module_path = ".".join(module_parts)
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            analyzer = ImportAnalyzer()
            analyzer.visit(tree)
            
            modules[module_path] = {
                'symbols': [s['name'] for s in analyzer.symbols_defined],
                'file_path': str(py_file),
                'imports_used': analyzer.imports_used,
                'symbols_used': [s['name'] for s in analyzer.symbols_used]
            }
            
        except Exception as e:
            print(f"Warning: Could not parse {py_file}: {e}")
    
    return modules


def build_symbol_to_module_map(modules):
    """Build a map from symbol name to the module(s) that define it."""
    symbol_map = defaultdict(list)
    
    for module_path, module_info in modules.items():
        for symbol in module_info['symbols']:
            symbol_map[symbol].append(module_path)
    
    return symbol_map


def is_simple_py_file(file_path: str) -> bool:
    """Check if a file is a simple.py export facade."""
    return file_path.endswith('/simple.py')


def find_correct_import_for_symbol(symbol: str, symbol_map: Dict[str, List[str]], adapter_symbols: Set[str]) -> Optional[str]:
    """Find the correct module to import a symbol from."""
    
    # Special handling for adapter symbols
    if symbol in adapter_symbols:
        return "shellforgepy.adapters.simple"
    
    # Find where this symbol is defined
    if symbol in symbol_map:
        candidates = symbol_map[symbol]
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            # Multiple definitions - prefer non-test modules
            non_test = [c for c in candidates if '/test' not in c]
            if len(non_test) == 1:
                return non_test[0]
            # If still ambiguous, return the first one
            return candidates[0]
    
    return None


def extract_symbols_used_in_file(file_path: str) -> Set[str]:
    """Extract all symbols that are actually used in a file (not just imported)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        analyzer = ImportAnalyzer()
        analyzer.visit(tree)
        
        # Get symbols that are actually used in the code
        used_symbols = set()
        for symbol_usage in analyzer.symbols_used:
            used_symbols.add(symbol_usage['name'])
        
        return used_symbols
        
    except Exception as e:
        print(f"Warning: Could not analyze {file_path}: {e}")
        return set()


def remove_all_imports_from_file(file_path: str) -> str:
    """Remove all import statements from a file and return the cleaned content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        lines = content.split('\n')
        
        # Collect line numbers of all import statements
        import_lines = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_lines.add(node.lineno - 1)  # Convert to 0-based
        
        # Remove import lines
        cleaned_lines = [line for i, line in enumerate(lines) if i not in import_lines]
        
        return '\n'.join(cleaned_lines)
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return ""


def generate_imports_for_file(file_path: str, modules: Dict, symbol_map: Dict, adapter_symbols: Set[str]) -> List[str]:
    """Generate the correct import statements for a file."""
    
    # Skip simple.py files - they use relative imports which should be handled separately
    if is_simple_py_file(file_path):
        return []
    
    # Get symbols actually used in this file
    used_symbols = extract_symbols_used_in_file(file_path)
    
    # Build imports
    imports = []
    import_map = defaultdict(list)  # module -> [symbols]
    
    for symbol in used_symbols:
        correct_module = find_correct_import_for_symbol(symbol, symbol_map, adapter_symbols)
        if correct_module and not correct_module.startswith('shellforgepy.'):
            correct_module = f"shellforgepy.{correct_module}"
        
        if correct_module:
            import_map[correct_module].append(symbol)
    
    # Generate import statements
    for module, symbols in import_map.items():
        for symbol in sorted(symbols):
            imports.append(f"from {module} import {symbol}")
    
    return imports


def get_adapter_symbols() -> Set[str]:
    """Get all symbols exported by the adapter simple.py."""
    adapter_simple_path = SRC_DIR / "shellforgepy/adapters/simple.py"
    if not adapter_simple_path.exists():
        return set()
    
    try:
        with open(adapter_simple_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Look for __all__ list
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == '__all__':
                        if isinstance(node.value, ast.List):
                            symbols = set()
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    symbols.add(elt.value)
                            return symbols
        
        # If no __all__, extract defined symbols
        analyzer = ImportAnalyzer()
        analyzer.visit(tree)
        return set(s['name'] for s in analyzer.symbols_defined)
        
    except Exception as e:
        print(f"Warning: Could not parse adapter simple.py: {e}")
        return set()


def run_isort_on_file(file_path: str):
    """Run isort on a file to canonicalize imports."""
    try:
        subprocess.run(['isort', file_path], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: isort failed on {file_path}: {e}")


def main():
    print("🔍 Discovering all modules and symbols...")
    modules = discover_all_modules_and_symbols()
    
    print(f"Found {len(modules)} modules")
    
    print("🗺️ Building symbol-to-module map...")
    symbol_map = build_symbol_to_module_map(modules)
    
    print("🔌 Getting adapter symbols...")
    adapter_symbols = get_adapter_symbols()
    print(f"Found {len(adapter_symbols)} adapter symbols")
    
    print("🔧 Regenerating imports for all files...")
    
    files_processed = 0
    files_with_changes = 0
    
    for module_path, module_info in modules.items():
        file_path = module_info['file_path']
        
        # Skip simple.py files - they have special import handling
        if is_simple_py_file(file_path):
            print(f"Skipping {file_path} (simple.py file)")
            continue
        
        print(f"Processing {file_path}...")
        
        # Generate new imports
        new_imports = generate_imports_for_file(file_path, modules, symbol_map, adapter_symbols)
        
        if new_imports:
            # Remove existing imports
            cleaned_content = remove_all_imports_from_file(file_path)
            
            # Add new imports at the top (after docstring if present)
            lines = cleaned_content.split('\n')
            
            # Find where to insert imports (after module docstring)
            insert_pos = 0
            if lines and lines[0].strip().startswith('"""'):
                # Find end of docstring
                for i, line in enumerate(lines[1:], 1):
                    if '"""' in line:
                        insert_pos = i + 1
                        break
            elif lines and lines[0].strip().startswith("'''"):
                # Find end of docstring
                for i, line in enumerate(lines[1:], 1):
                    if "'''" in line:
                        insert_pos = i + 1
                        break
            
            # Insert imports
            new_lines = lines[:insert_pos] + [''] + new_imports + [''] + lines[insert_pos:]
            new_content = '\n'.join(new_lines)
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            # Run isort to canonicalize
            run_isort_on_file(file_path)
            
            files_with_changes += 1
            print(f"  ✅ Updated with {len(new_imports)} imports")
        else:
            print(f"  ⚪ No imports needed")
        
        files_processed += 1
    
    print(f"\n📊 Summary:")
    print(f"  📁 Files processed: {files_processed}")
    print(f"  ✅ Files updated: {files_with_changes}")
    print(f"  ⚪ Files unchanged: {files_processed - files_with_changes}")


if __name__ == "__main__":
    main()