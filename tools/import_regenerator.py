#!/usr/bin/env python3
"""
Robust import regeneration utilities.

This module provides functions to reliably remove and regenerate imports in Python files.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass


@dataclass
class ImportSpec:
    """Specification for an import statement."""
    module: str
    name: Optional[str] = None  # None for "import module", string for "from module import name"
    alias: Optional[str] = None  # Optional alias for the import
    
    def __post_init__(self):
        # Validation: alias without name is only valid for "import module as alias"
        # For "from module import name as alias", both name and alias must be provided
        pass  # All combinations are valid in our design


class ImportRegenerationError(Exception):
    """Raised when import regeneration fails."""
    pass


class ExoticImportError(ImportRegenerationError):
    """Raised when file has exotic import patterns we don't support."""
    pass


def validate_file_structure(file_path: str) -> None:
    """
    Validate that file has the expected structure:
    - Optional module docstring
    - Import statements
    - Then code starts
    
    Raises ExoticImportError if structure is exotic.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        lines = content.split('\n')
        
        # Find all import statement line numbers
        import_lines = set()
        non_import_lines = set()
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_lines.add(node.lineno)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.Expr)):
                # Skip module-level docstrings (first Expr that's a string)
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                    if isinstance(node.value.value, str) and node.lineno <= 5:
                        continue  # This is likely a module docstring
                non_import_lines.add(node.lineno)
        
        # Check if imports are mixed with code
        if import_lines and non_import_lines:
            min_import = min(import_lines)
            max_import = max(import_lines)
            min_code = min(non_import_lines)
            
            # If there's code before imports or imports after code starts, it's exotic
            if min_code < min_import or max_import > min_code:
                # Check if this is just interspersed comments or blank lines
                code_before_imports = [line_no for line_no in non_import_lines if line_no < min_import]
                imports_after_code = [line_no for line_no in import_lines if line_no > min_code]
                
                if code_before_imports and imports_after_code:
                    raise ExoticImportError(
                        f"File {file_path} has exotic import structure: "
                        f"imports and code are interspersed"
                    )
        
    except SyntaxError as e:
        raise ImportRegenerationError(f"Cannot parse {file_path}: {e}")
    except Exception as e:
        raise ImportRegenerationError(f"Error validating {file_path}: {e}")


def extract_current_imports(file_path: str) -> List[ImportSpec]:
    """
    Extract all current import statements from a file.
    
    Returns list of ImportSpec objects representing current imports.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportSpec(
                        module=alias.name,
                        name=None,
                        alias=alias.asname
                    ))
            elif isinstance(node, ast.ImportFrom):
                if node.module:  # Skip relative imports without module
                    for alias in node.names:
                        imports.append(ImportSpec(
                            module=node.module,
                            name=alias.name,
                            alias=alias.asname
                        ))
        
        return imports
        
    except Exception as e:
        raise ImportRegenerationError(f"Failed to extract imports from {file_path}: {e}")


def remove_all_imports(file_path: str) -> str:
    """
    Remove all import statements from a file and return the cleaned content.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        File content with all imports removed
        
    Raises:
        ImportRegenerationError: If operation fails
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        lines = content.split('\n')
        
        # Collect line numbers of all import statements
        import_lines = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_lines.add(node.lineno - 1)  # Convert to 0-based indexing
        
        # Remove import lines
        cleaned_lines = [line for i, line in enumerate(lines) if i not in import_lines]
        
        return '\n'.join(cleaned_lines)
        
    except Exception as e:
        raise ImportRegenerationError(f"Failed to remove imports from {file_path}: {e}")


def generate_import_section(import_specs: List[ImportSpec]) -> str:
    """
    Generate a clean import section from import specifications.
    
    Args:
        import_specs: List of ImportSpec objects
        
    Returns:
        String containing properly formatted import statements
    """
    if not import_specs:
        return ""
    
    # Group imports by type and module
    simple_imports = []  # "import module"
    from_imports = {}    # "from module import ..." grouped by module
    
    for spec in import_specs:
        if spec.name is None:
            # Simple import: "import module" or "import module as alias"
            if spec.alias:
                simple_imports.append(f"import {spec.module} as {spec.alias}")
            else:
                simple_imports.append(f"import {spec.module}")
        else:
            # From import: "from module import name" or "from module import name as alias"
            if spec.module not in from_imports:
                from_imports[spec.module] = []
            
            if spec.alias:
                from_imports[spec.module].append(f"{spec.name} as {spec.alias}")
            else:
                from_imports[spec.module].append(spec.name)
    
    # Generate import lines
    import_lines = []
    
    # Add simple imports first
    for imp in sorted(simple_imports):
        import_lines.append(imp)
    
    # Add from imports
    for module in sorted(from_imports.keys()):
        names = sorted(from_imports[module])
        if len(names) == 1:
            import_lines.append(f"from {module} import {names[0]}")
        else:
            # Multi-line import
            import_lines.append(f"from {module} import (")
            for name in names[:-1]:
                import_lines.append(f"    {name},")
            import_lines.append(f"    {names[-1]}")
            import_lines.append(")")
    
    return '\n'.join(import_lines)


def find_import_insertion_point(content: str) -> int:
    """
    Find the appropriate line number to insert imports.
    
    Args:
        content: File content as string
        
    Returns:
        Line number (0-based) where imports should be inserted
    """
    lines = content.split('\n')
    
    # Skip shebang and encoding declarations
    insert_pos = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#!') or stripped.startswith('# -*- coding:') or stripped.startswith('# coding:'):
            insert_pos = i + 1
        else:
            break
    
    # Skip module docstring
    if insert_pos < len(lines):
        # Check for triple-quoted docstring
        docstring_start = None
        for i in range(insert_pos, min(insert_pos + 5, len(lines))):  # Check first few lines
            line = lines[i].strip()
            if line.startswith('"""') or line.startswith("'''"):
                docstring_start = i
                quote_type = '"""' if line.startswith('"""') else "'''"
                
                # Find end of docstring
                if line.count(quote_type) >= 2:  # Single line docstring
                    insert_pos = i + 1
                    break
                else:  # Multi-line docstring
                    for j in range(i + 1, len(lines)):
                        if quote_type in lines[j]:
                            insert_pos = j + 1
                            break
                break
    
    return insert_pos


def regenerate_imports(file_path: str, import_specs: List[ImportSpec]) -> None:
    """
    Completely regenerate the import section of a Python file.
    
    Args:
        file_path: Path to the Python file to modify
        import_specs: List of ImportSpec objects specifying desired imports
        
    Raises:
        ImportRegenerationError: If operation fails
        ExoticImportError: If file structure is not supported
    """
    # Validate file structure first
    validate_file_structure(file_path)
    
    try:
        # Remove all existing imports
        cleaned_content = remove_all_imports(file_path)
        
        # Generate new import section
        import_section = generate_import_section(import_specs)
        
        # Find where to insert imports
        insert_pos = find_import_insertion_point(cleaned_content)
        
        # Reconstruct file content
        lines = cleaned_content.split('\n')
        new_lines = lines[:insert_pos]
        
        # Add imports with proper spacing
        if import_section:
            if new_lines and new_lines[-1].strip():  # Add blank line after existing content
                new_lines.append('')
            new_lines.extend(import_section.split('\n'))
            new_lines.append('')  # Blank line after imports
        
        new_lines.extend(lines[insert_pos:])
        
        # Write back to file
        new_content = '\n'.join(new_lines)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
    except Exception as e:
        raise ImportRegenerationError(f"Failed to regenerate imports in {file_path}: {e}")


def main():
    """Test the import regeneration functions."""
    # Example usage
    import_specs = [
        ImportSpec("os"),
        ImportSpec("sys"),
        ImportSpec("pathlib", "Path"),
        ImportSpec("typing", "Dict"),
        ImportSpec("typing", "List"),
        ImportSpec("dataclasses", "dataclass"),
        ImportSpec("numpy", name=None, alias="np"),
    ]
    
    print("Import specs created:")
    for spec in import_specs:
        print(f"  {spec}")
    
    print("\nGenerated import section:")
    print(generate_import_section(import_specs))


if __name__ == "__main__":
    main()