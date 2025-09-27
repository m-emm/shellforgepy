import ast
from collections import defaultdict
from pathlib import Path
from typing import Set, Dict, List, Tuple

PROJECT_DIR = Path(__file__).parent.parent.resolve()
SRC_DIR = PROJECT_DIR / "src"
TESTS_DIR = PROJECT_DIR / "tests"


class ImportVisitor(ast.NodeVisitor):
    """AST visitor to extract import information from Python files."""
    
    def __init__(self):
        self.imports = []
        self.from_imports = []
    
    def visit_Import(self, node):
        """Handle 'import module' statements."""
        for alias in node.names:
            self.imports.append({
                'module': alias.name,
                'asname': alias.asname,
                'type': 'import'
            })
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Handle 'from module import name' statements."""
        if node.module:  # Has a module name
            for alias in node.names:
                self.from_imports.append({
                    'module': node.module,
                    'name': alias.name,
                    'asname': alias.asname,
                    'type': 'from_import',
                    'level': node.level  # 0 = absolute, >0 = relative
                })
        elif node.level > 0:  # Relative import without module (from . import ...)
            for alias in node.names:
                self.from_imports.append({
                    'module': None,
                    'name': alias.name,
                    'asname': alias.asname,
                    'type': 'from_import',
                    'level': node.level
                })
        self.generic_visit(node)


def analyze_file_imports(file_path: Path) -> Tuple[List[Dict], List[Dict]]:
    """Parse a Python file and extract all import statements using AST."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        visitor = ImportVisitor()
        visitor.visit(tree)
        
        return visitor.imports, visitor.from_imports
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return [], []


def module_paths():
    """Collect all module paths and their imports."""
    print(f"Scanning source directory: {SRC_DIR}")
    print(f"Scanning tests directory: {TESTS_DIR}")
    
    all_paths = set()
    paths_being_imported = defaultdict(set)
    file_imports = {}  # Store detailed import info per file
    
    # Scan source files
    for path in SRC_DIR.rglob("*.py"):
        print(f"Processing: {path}")
        relative_path = path.relative_to(SRC_DIR)
        
        # Build module path (e.g., "shellforgepy.construct.construct_utils")
        module_path = ".".join(relative_path.with_suffix("").parts)
        all_paths.add(module_path)
        
        # Analyze imports in this file
        imports, from_imports = analyze_file_imports(path)
        file_imports[path.as_posix()] = {'imports': imports, 'from_imports': from_imports}
        
        # Process regular imports
        for imp in imports:
            paths_being_imported[imp['module']].add(path.as_posix())
        
        # Process from imports  
        for imp in from_imports:
            paths_being_imported[imp['module']].add(path.as_posix())
    
    # Scan test files
    if TESTS_DIR.exists():
        for path in TESTS_DIR.rglob("*.py"):
            print(f"Processing test: {path}")
            
            # Analyze imports in this file
            imports, from_imports = analyze_file_imports(path)
            file_imports[path.as_posix()] = {'imports': imports, 'from_imports': from_imports}
            
            # Process regular imports
            for imp in imports:
                paths_being_imported[imp['module']].add(path.as_posix())
            
            # Process from imports
            for imp in from_imports:
                paths_being_imported[imp['module']].add(path.as_posix())
                
    return sorted(all_paths), dict(paths_being_imported), file_imports


def find_misplaced_imports(all_paths, paths_being_imported, file_imports):
    """Find imports that should use full module paths but are using relative/incorrect paths."""
    misplaced_imports = []
    
    # Create lookups for finding correct imports
    module_by_basename = {}
    for full_path in all_paths:
        basename = full_path.split(".")[-1]
        if basename in module_by_basename:
            # Multiple modules with same basename - store as list
            if not isinstance(module_by_basename[basename], list):
                module_by_basename[basename] = [module_by_basename[basename]]
            module_by_basename[basename].append(full_path)
        else:
            module_by_basename[basename] = full_path
    
    # Check files for problematic relative imports
    for file_path, imports_data in file_imports.items():
        # Skip test files that import py_3d_construct_lib (already handled)
        if 'py_3d_construct_lib' in file_path or '/tests/' not in file_path:
            pass  # Process source files
        
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            continue
            
        try:
            relative_to_src = file_path_obj.relative_to(Path(file_path).parts[0])
        except:
            continue
            
        # Analyze each from_import in this file
        for imp in imports_data.get('from_imports', []):
            # Check for problematic relative imports (level > 0)
            if imp.get('level', 0) > 0 and imp.get('module'):
                # This is a relative import like "from .construct_utils import ..."
                imported_module = imp['module']
                
                # Look for a match in our known modules
                if imported_module in module_by_basename:
                    correct_module = module_by_basename[imported_module]
                    
                    if isinstance(correct_module, list):
                        # Multiple possibilities
                        misplaced_imports.append({
                            'file': file_path,
                            'incorrect_import': f".{imported_module}",
                            'possible_corrections': correct_module,
                            'reason': 'Relative import with multiple possible targets',
                            'import_statements': [f"from .{imported_module} import {imp['name']}"]
                        })
                    else:
                        # Single match found
                        misplaced_imports.append({
                            'file': file_path,
                            'incorrect_import': f".{imported_module}",
                            'correct_import': correct_module,
                            'reason': 'Relative import should be absolute',
                            'import_statements': [f"from .{imported_module} import {imp['name']}"]
                        })
    
    # Also check the original logic for other types of misplaced imports
    for imported_module, file_paths in paths_being_imported.items():
        # Skip external dependencies and already correct imports
        if imported_module in all_paths:
            continue
            
        # Check if this is a partial/incorrect import of an internal module
        module_parts = imported_module.split(".")
        last_part = module_parts[-1]
        
        # Look for matches by basename
        if last_part in module_by_basename:
            correct_module = module_by_basename[last_part]
            
            # Handle case where multiple modules have same basename
            if isinstance(correct_module, list):
                # Try to find the best match
                best_matches = []
                for candidate in correct_module:
                    if candidate.endswith("." + imported_module.replace(".", ".")):
                        best_matches.append(candidate)
                
                if len(best_matches) == 1:
                    correct_module = best_matches[0]
                else:
                    # Multiple or no matches - report all possibilities
                    for file_path in file_paths:
                        misplaced_imports.append({
                            'file': file_path,
                            'incorrect_import': imported_module,
                            'possible_corrections': correct_module,
                            'reason': 'Multiple modules with same basename'
                        })
                    continue
            
            # Found a single match - this is likely a misplaced import
            for file_path in file_paths:
                # Get specific import details for this file
                import_details = []
                if file_path in file_imports:
                    for imp in file_imports[file_path]['imports']:
                        if imp['module'] == imported_module:
                            import_details.append(f"import {imp['module']}")
                    for imp in file_imports[file_path]['from_imports']:
                        if imp['module'] == imported_module:
                            import_details.append(f"from {imp['module']} import {imp['name']}")
                
                misplaced_imports.append({
                    'file': file_path,
                    'incorrect_import': imported_module,
                    'correct_import': correct_module,
                    'import_statements': import_details,
                    'reason': 'Using short name instead of full module path'
                })
        
        # Also check for partial paths that could be extended
        for full_path in all_paths:
            if full_path.endswith("." + imported_module):
                for file_path in file_paths:
                    import_details = []
                    if file_path in file_imports:
                        for imp in file_imports[file_path]['imports']:
                            if imp['module'] == imported_module:
                                import_details.append(f"import {imp['module']}")
                        for imp in file_imports[file_path]['from_imports']:
                            if imp['module'] == imported_module:
                                import_details.append(f"from {imp['module']} import {imp['name']}")
                    
                    misplaced_imports.append({
                        'file': file_path,
                        'incorrect_import': imported_module,
                        'correct_import': full_path,
                        'import_statements': import_details,
                        'reason': 'Using partial path instead of full module path'
                    })
    
    return misplaced_imports


def main():
    all_paths, paths_being_imported, file_imports = module_paths()
    
    print("\n" + "="*50)
    print("DISCOVERED MODULES:")
    print("="*50)
    for path in all_paths:
        print(f"  {path}")
    
    # Find misplaced imports
    misplaced_imports = find_misplaced_imports(all_paths, paths_being_imported, file_imports)
    
    print("\n" + "="*60)
    print("MISPLACED IMPORTS (Need Fixing):")
    print("="*60)
    
    if not misplaced_imports:
        print("✓ No misplaced imports found!")
    else:
        for item in misplaced_imports:
            print(f"\nFile: {item['file']}")
            print(f"  ❌ Incorrect: {item['incorrect_import']}")
            if 'correct_import' in item:
                print(f"  ✅ Should be: {item['correct_import']}")
            elif 'possible_corrections' in item:
                print(f"  ✅ Possible corrections: {item['possible_corrections']}")
            print(f"  📝 Reason: {item['reason']}")
            if item.get('import_statements'):
                print(f"  📄 Statements: {item['import_statements']}")
    
    print("\n" + "="*50)
    print("COMPLETE IMPORT ANALYSIS:")
    print("="*50)
    
    # Build lookup for existing modules by their last component
    existing_paths_by_name = {p.split(".")[-1]: p for p in all_paths}
    
    internal_imports = 0
    external_imports = 0
    
    for imported_module, file_paths in sorted(paths_being_imported.items()):
        print(f"\nModule being imported: {imported_module}")
        print(f"  Used in files: {len(file_paths)} file(s)")
        
        # Check if this matches any of our internal modules
        module_parts = imported_module.split(".")
        is_internal = False
        for i in range(len(module_parts)):
            partial_name = ".".join(module_parts[i:])
            if partial_name in all_paths:
                print(f"  ✓ Maps to internal module: {partial_name}")
                internal_imports += 1
                is_internal = True
                break
        
        if not is_internal:
            # Check by last component name
            last_component = imported_module.split(".")[-1]
            if last_component in existing_paths_by_name:
                print(f"  ? Possible internal match: {existing_paths_by_name[last_component]}")
                # Don't count this as internal since it's ambiguous
            else:
                print(f"  ✗ External dependency: {imported_module}")
            external_imports += 1
    
    print(f"\n" + "="*50)
    print(f"SUMMARY:")
    print(f"  📦 Internal modules: {len(all_paths)}")
    print(f"  🔗 Total unique imports: {len(paths_being_imported)}")
    print(f"  🏠 Internal imports: {internal_imports}")
    print(f"  🌐 External imports: {external_imports}")
    print(f"  ⚠️  Misplaced imports: {len(misplaced_imports)}")
    print("="*50)


if __name__ == "__main__":
    main()