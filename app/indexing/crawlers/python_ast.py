# Python AST Crawler for CodeAtlas.
# This crawler accepts a project folder path, finds all Python files inside it,
# and uses Python's built-in ast module to extract structured information:
# function names, parameters, return types, docstrings, classes, imports,
# and function calls. This structured data is what we will later store in
# Neo4j (as nodes and relationships) and use to create smart text chunks
# for embeddings in ChromaDB.

import ast
from pathlib import Path


def read_file(file_path: str) -> str:
    """
    Reads a Python source file and returns its content as a string.

    Kept separate from parsing so each function has one purpose —
    reading is one job, parsing is another.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def find_python_files(project_path: str) -> list[str]:
    """
    Recursively finds all Python (.py) files inside a project folder.

    Uses pathlib.Path.rglob to walk the entire folder tree. Skips files
    inside __pycache__ folders since those are compiled bytecode files,
    not source code we want to index.

    Returns a sorted list of file paths as strings so the order is
    consistent across different operating systems.
    """
    project = Path(project_path)

    # rglob("*.py") walks all subfolders and returns every .py file it finds
    python_files = [
        str(file)
        for file in project.rglob("*.py")
        if "__pycache__" not in file.parts
        and ".venv" not in file.parts
    ]

    return sorted(python_files)


def extract_docstring(node: ast.FunctionDef | ast.ClassDef) -> str:
    """
    Extracts the docstring from a function or class node if one exists.

    In an AST, a docstring is the first statement in a function or class body
    and it is a string constant. Returns empty string if no docstring exists
    so callers never have to handle None.
    """
    docstring = ast.get_docstring(node)
    return docstring if docstring else ""


def extract_parameters(node: ast.FunctionDef) -> list[dict]:
    """
    Extracts all parameters from a function node, including their type hints.

    Skips 'self' because it is not a real parameter — it is just Python's way
    of referring to the class instance inside a method.

    Returns a list of dicts, one per parameter, with keys:
    - name: the parameter name
    - type: the type annotation as a string, or "unknown" if not declared
    """
    parameters = []

    for arg in node.args.args:
        # skip self — it is a class reference, not a meaningful parameter
        if arg.arg == "self":
            continue

        # ast.unparse converts the annotation node back to a readable string
        # e.g. the annotation node for "float" becomes the string "float"
        param_type = ast.unparse(arg.annotation) if arg.annotation else "unknown"
        parameters.append({
            "name": arg.arg,
            "type": param_type,
        })

    return parameters


def extract_return_type(node: ast.FunctionDef) -> str:
    """
    Extracts the return type annotation from a function node.

    Returns the type as a string (e.g. "float", "list[str]"), or
    "unknown" if no return type is declared. This matters for the
    Neo4j graph — knowing what a function returns helps trace data flow.
    """
    if node.returns:
        return ast.unparse(node.returns)
    return "unknown"


def extract_function_calls(node: ast.FunctionDef) -> list[str]:
    """
    Extracts all function calls made inside a function or method body.

    Walks every node inside the function body looking for ast.Call nodes.
    Each ast.Call represents one function being called. We extract the
    function name from node.func.

    Two cases exist for how a call looks in the AST:
    - Simple call: calculate_average(marks) → node.func is ast.Name, name is node.func.id
    - Method call: obj.method() → node.func is ast.Attribute, name is node.func.attr

    This is what creates CALLS relationships in Neo4j:
    get_student_report → CALLS → calculate_average
    """
    calls = []

    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            # simple function call: calculate_average(marks)
            if isinstance(child.func, ast.Name):
                calls.append(child.func.id)

            # method call on an object: obj.method()
            elif isinstance(child.func, ast.Attribute):
                calls.append(child.func.attr)

    # remove duplicates while preserving order
    seen = set()
    unique_calls = []
    for call in calls:
        if call not in seen:
            seen.add(call)
            unique_calls.append(call)

    return unique_calls


def extract_function_info(node: ast.FunctionDef, source_code: str) -> dict:
    """
    Extracts all meaningful information from a single function or method node.

    Returns a dict with name, docstring, parameters, return type, line number,
    calls made inside the body, and the actual source code of the function.

    The source code is included so the JSON is fully self-contained — the
    chunker never needs to read the original .py file.

    Centralising this here avoids duplicating the same logic in both
    extract_functions and extract_methods.
    """
    return {
        "name": node.name,
        "docstring": extract_docstring(node),
        "parameters": extract_parameters(node),
        "return_type": extract_return_type(node),
        "line_number": node.lineno,
        "calls": extract_function_calls(node),
        # ast.get_source_segment extracts the exact lines of source code
        # for this node — preserving indentation and formatting
        "source_code": ast.get_source_segment(source_code, node) or "",
    }


def extract_functions(tree: ast.Module, source_code: str) -> list[dict]:
    """
    Extracts only the top-level functions from a parsed AST module.

    Iterates tree.body (direct children of the file) rather than using
    ast.walk — this ensures we only get standalone functions, not methods
    that belong to a class. Class methods are handled by extract_methods.
    """
    functions = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions.append(extract_function_info(node, source_code))

    return functions


def extract_imports(tree: ast.Module) -> list[dict]:
    """
    Extracts all import statements from a parsed AST module.

    Handles two types of imports:
    - ast.Import: plain imports like `import ast` or `import os`
    - ast.ImportFrom: from-imports like `from utils import calculate_average`

    For Neo4j, ImportFrom is the most valuable because it tells us exactly
    which module a file depends on and which names it pulls from it.
    This is what creates the graph edge: File → IMPORTS → File.

    Returns a list of dicts, one per import statement, with keys:
    - type: "import" or "from_import"
    - module: the module being imported (e.g. "utils", "ast")
    - names: list of names being imported (e.g. ["calculate_average", "calculate_grade"])
    - line_number: where the import appears in the file
    """
    imports = []

    for node in tree.body:
        # plain import: `import ast`
        if isinstance(node, ast.Import):
            imports.append({
                "type": "import",
                "module": ", ".join(alias.name for alias in node.names),
                "names": [alias.name for alias in node.names],
                "line_number": node.lineno,
            })

        # from-import: `from utils import calculate_average, calculate_grade`
        elif isinstance(node, ast.ImportFrom):
            imports.append({
                "type": "from_import",
                "module": node.module,
                "names": [alias.name for alias in node.names],
                "line_number": node.lineno,
            })

    return imports


def extract_methods(class_node: ast.ClassDef, source_code: str) -> list[dict]:
    """
    Extracts all methods from a single class node.

    Iterates class_node.body (direct children of the class) so we only
    get methods that truly belong to this class, not nested functions
    inside those methods.
    """
    methods = []

    for node in class_node.body:
        if isinstance(node, ast.FunctionDef):
            methods.append(extract_function_info(node, source_code))

    return methods


def extract_classes(tree: ast.Module, source_code: str) -> list[dict]:
    """
    Extracts all classes from a parsed AST module, including their methods.

    For each class, returns a dict with:
    - name: class name
    - docstring: the class-level docstring
    - line_number: where the class starts in the file
    - source_code: the actual source code of the entire class
    - methods: list of method dicts extracted by extract_methods

    This is the foundation of the Neo4j graph edge: Class → HAS_METHOD → Method.
    """
    classes = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append({
                "name": node.name,
                "docstring": extract_docstring(node),
                "line_number": node.lineno,
                "source_code": ast.get_source_segment(source_code, node) or "",
                "methods": extract_methods(node, source_code),
            })

    return classes


def extract_module_level_variables(tree: ast.Module) -> list[dict]:
    """
    Extracts global variable assignments at the top level of a file.

    These are variables defined outside any function or class — for example:
        DEFAULT_PASS_MARK = 50
        system_profile = StudentProfile("System", 18)

    Since they belong to no function or class, they are tagged to the file
    itself in Neo4j: File → HAS_VARIABLE → variable_name

    Only captures simple assignments (ast.Assign) and annotated assignments
    (ast.AnnAssign). Skips functions, classes, and imports.

    Returns a list of dicts with keys:
    - name: the variable name
    - value: the assigned value as a readable string
    - line_number: where the assignment appears in the file
    """
    variables = []

    for node in tree.body:
        # simple assignment: DEFAULT_PASS_MARK = 50
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    variables.append({
                        "name": target.id,
                        "value": ast.unparse(node.value),
                        "line_number": node.lineno,
                    })

        # annotated assignment: count: int = 0
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value:
                variables.append({
                    "name": node.target.id,
                    "value": ast.unparse(node.value),
                    "line_number": node.lineno,
                })

    return variables


def extract_module_level_calls(tree: ast.Module) -> list[str]:
    """
    Extracts all function calls made at the module level — outside any
    function or class body.

    These calls are tagged to the file itself in Neo4j:
        File → CALLS → function_name

    Common examples:
        system_profile = StudentProfile("System", 18)  ← calls StudentProfile
        parser = argparse.ArgumentParser()              ← calls ArgumentParser

    Skips FunctionDef, ClassDef, and Import nodes since those are handled
    by their own extractors. Everything else (Assign, Expr, If) is searched
    for ast.Call nodes.
    """
    # node types that have their own extractors — skip them
    SKIP_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)

    calls = []

    for node in tree.body:
        if isinstance(node, SKIP_TYPES):
            continue

        # walk everything else and find all function calls
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)

    # remove duplicates while preserving order
    seen = set()
    unique_calls = []
    for call in calls:
        if call not in seen:
            seen.add(call)
            unique_calls.append(call)

    return unique_calls


def crawl_file(file_path: str) -> dict:
    """
    Runs all extractors on a single Python file and returns a structured dict.

    This is the final output of the crawler for one file — a complete picture
    of everything in that file that Neo4j and ChromaDB need:
    - file: the filename
    - path: the full file path
    - imports: all import statements (become IMPORTS relationships in Neo4j)
    - functions: all top-level functions with their calls (become Function nodes)
    - classes: all classes and their methods (become Class and Method nodes)
    """
    source_code = read_file(file_path)

    # Parse once, pass the same tree to all extractors — no double parsing
    tree = ast.parse(source_code)

    return {
        "file": Path(file_path).name,
        "path": file_path,
        "imports": extract_imports(tree),
        "module_variables": extract_module_level_variables(tree),
        "module_calls": extract_module_level_calls(tree),
        "functions": extract_functions(tree, source_code),
        "classes": extract_classes(tree, source_code),
    }


def print_crawler_output(result: dict) -> None:
    """
    Prints the crawl result for one file in a readable format.

    Shows imports first (graph edges), then top-level functions with their
    calls, then classes with their methods. This is only used for exploration
    and debugging — in the real pipeline this data goes to Neo4j and ChromaDB.
    """
    imports = result["imports"]
    module_variables = result["module_variables"]
    module_calls = result["module_calls"]
    functions = result["functions"]
    classes = result["classes"]

    print(f"\n{'='*60}")
    print(f"File         : {result['path']}")
    print(f"Imports      : {len(imports)}  |  Module Vars: {len(module_variables)}  |  Module Calls: {len(module_calls)}")
    print(f"Functions    : {len(functions)}  |  Classes: {len(classes)}")
    print(f"{'='*60}")

    if imports:
        print("\n--- Imports ---")
        for imp in imports:
            if imp["type"] == "from_import":
                print(f"  from {imp['module']} import {', '.join(imp['names'])}  (line {imp['line_number']})")
            else:
                print(f"  import {imp['module']}  (line {imp['line_number']})")

    if module_variables:
        print("\n--- Module-Level Variables (tagged to file) ---")
        for var in module_variables:
            print(f"  {var['name']} = {var['value']}  (line {var['line_number']})")

    if module_calls:
        print("\n--- Module-Level Calls (tagged to file) ---")
        print(f"  {', '.join(module_calls)}")

    if functions:
        print("\n--- Top-Level Functions ---")
        for func in functions:
            print(f"\nFunction : {func['name']} (line {func['line_number']})")
            print(f"Docstring: {func['docstring']}")
            print(f"Returns  : {func['return_type']}")
            print(f"Params   :")
            for param in func["parameters"]:
                print(f"           {param['name']} -> {param['type']}")
            print(f"Calls    : {', '.join(func['calls']) if func['calls'] else 'none'}")

    if classes:
        print("\n--- Classes ---")
        for cls in classes:
            print(f"\nClass    : {cls['name']} (line {cls['line_number']})")
            print(f"Docstring: {cls['docstring']}")
            print(f"Methods  : {len(cls['methods'])}")
            for method in cls["methods"]:
                print(f"\n  Method : {method['name']} (line {method['line_number']})")
                print(f"  Docs   : {method['docstring']}")
                print(f"  Returns: {method['return_type']}")
                print(f"  Params :")
                for param in method["parameters"]:
                    print(f"             {param['name']} -> {param['type']}")
                print(f"  Calls  : {', '.join(method['calls']) if method['calls'] else 'none'}")


if __name__ == "__main__":
    import sys

    # sys.argv[0] is always the script name itself.
    # sys.argv[1] is the project folder path the user passes.
    if len(sys.argv) < 2:
        print("Usage: uv run crawlers/python_ast.py <project_path>")
        sys.exit(1)

    PROJECT_PATH = sys.argv[1]

    python_files = find_python_files(PROJECT_PATH)

    print(f"\nProject    : {PROJECT_PATH}")
    print(f"Files found: {len(python_files)}")

    for file_path in python_files:
        result = crawl_file(file_path)
        print_crawler_output(result)
