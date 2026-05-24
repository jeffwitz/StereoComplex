#!/usr/bin/env python3
"""Pre-commit guard: check that docstring Parameters match the actual function signature.

Usage: python3 examples/notebooks/check_docstring_params.py [file...]
Exits with 1 and prints mismatches if any docstring invents parameter names.
"""
import ast, sys, os

def check_file(path):
    with open(path) as f:
        source = f.read()
    tree = ast.parse(source)
    errors = []
    
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node)
        if doc is None or 'Parameters' not in doc:
            continue
        
        # Get real parameter names (skip self/cls)
        real_args = {a.arg for a in node.args.args}
        real_args -= {'self', 'cls'}
        if not real_args:
            continue
        
        # Extract names mentioned in the Parameters section
        # Simple heuristic: look for "name : type" patterns
        params_section = doc.split('Parameters')[1] if 'Parameters' in doc else ''
        lines = params_section.split('\n')
        doc_params = set()
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('-'):
                continue
            # Match "name : type" or "name, name2 : type"
            if ':' in stripped:
                name_part = stripped.split(':')[0].strip()
                for name in name_part.split(','):
                    name = name.strip()
                    if name and not name[0].isdigit():
                        doc_params.add(name)
        
        # Check for invented parameters
        invented = doc_params - real_args
        if invented:
            errors.append(
                f"{path}:{node.lineno} {node.name}() invents: {sorted(invented)} "
                f"(real: {sorted(real_args)})"
            )
    
    return errors

if __name__ == '__main__':
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        # Default: scan src/ excluding api/
        import glob
        files = [f for f in glob.glob('src/stereocomplex/**/*.py', recursive=True)
                 if '/api/' not in f]
    
    all_errors = []
    for f in files:
        if os.path.isfile(f):
            all_errors.extend(check_file(f))
    
    if all_errors:
        print(f"DOCSTRING MISMATCH: {len(all_errors)} function(s) invent parameter names!\n")
        for e in all_errors:
            print(f"  {e}")
        print("\nFix: read the actual signature with ast.parse and use exact names.")
        sys.exit(1)
    else:
        print(f"OK: all docstring Parameters match actual signatures ({len(files)} files)")
        sys.exit(0)
