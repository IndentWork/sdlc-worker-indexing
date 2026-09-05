"""
Graph loader — converts crawler output into graph nodes and edges in Cosmos DB.

Adapted from neo4j_loader/load_project.py in the prototype.
Only change: Neo4j replaced with Cosmos DB (cosmos.py).

Two-pass approach (same as prototype):
  Pass 1 — create all nodes (File, Function, Class, Method)
  Pass 2 — create all edges (CALLS, IMPORTS, HAS_METHOD)

Two passes are required because CALLS and IMPORTS are cross-file relationships.
All nodes must exist before any edge can reference them.

Node ID and edge ID follow the naming convention:
    {resource_code}:{github_org}:{project}:{repo}:{file}:{symbol}
"""
from app.services.cosmos import (
    upsert_node,
    upsert_edge,
    delete_node,
    get_node_ids_for_file,
)


def _build_id(
    resource_code: str, github_org: str, project: str, repo: str, file_path: str, *parts: str
) -> str:
    """
    Build a unique stable ID following the naming convention.
    Example: b310545b:sdlc-tenant:ecommerce:cart-service:cart.py:add_to_cart
    """
    return ":".join([resource_code, github_org, project, repo, file_path] + list(parts))


async def _upsert_file_node(
    crawl_result: dict,
    resource_code: str, github_org: str, project: str, repo: str,
) -> None:
    """Create or update the File node for a crawled file."""
    file_id = _build_id(resource_code, github_org, project, repo, crawl_result["path"])

    await upsert_node({
        "id":            file_id,
        "resource_code": resource_code,
        "github_org":    github_org,
        "project":       project,
        "repo":          repo,
        "type":          "File",
        "file":          crawl_result["path"],
        "name":          crawl_result["file"],
        "sha":           crawl_result.get("sha", ""),
    })


async def _upsert_function_node(
    func: dict,
    crawl_result: dict,
    resource_code: str, github_org: str, project: str, repo: str,
) -> str:
    """Create or update a Function node. Returns the node ID."""
    node_id = _build_id(resource_code, github_org, project, repo, crawl_result["path"], func["name"])

    await upsert_node({
        "id":            node_id,
        "resource_code": resource_code,
        "github_org":    github_org,
        "project":       project,
        "repo":          repo,
        "type":          "Function",
        "file":          crawl_result["path"],
        "name":          func["name"],
        "docstring":     func.get("docstring", ""),
        "return_type":   func.get("return_type", "unknown"),
        "parameters":    func.get("parameters", []),
        "calls":         func.get("calls", []),
        "line_number":   func.get("line_number", 0),
        "sha":           crawl_result.get("sha", ""),
    })

    return node_id


async def _upsert_class_and_methods(
    cls: dict,
    crawl_result: dict,
    resource_code: str, github_org: str, project: str, repo: str,
) -> list[str]:
    """
    Create or update a Class node and all its Method nodes.
    Returns list of all node IDs created (class + methods).
    """
    node_ids = []

    class_id = _build_id(resource_code, github_org, project, repo, crawl_result["path"], cls["name"])

    await upsert_node({
        "id":            class_id,
        "resource_code": resource_code,
        "github_org":    github_org,
        "project":       project,
        "repo":          repo,
        "type":          "Class",
        "file":          crawl_result["path"],
        "name":          cls["name"],
        "docstring":     cls.get("docstring", ""),
        "line_number":   cls.get("line_number", 0),
        "sha":           crawl_result.get("sha", ""),
    })
    node_ids.append(class_id)

    for method in cls.get("methods", []):
        method_id = _build_id(
            resource_code, github_org, project, repo,
            crawl_result["path"], cls["name"], method["name"]
        )

        await upsert_node({
            "id":            method_id,
            "resource_code": resource_code,
            "github_org":    github_org,
            "project":       project,
            "repo":          repo,
            "type":          "Method",
            "file":          crawl_result["path"],
            "class_name":    cls["name"],
            "name":          method["name"],
            "docstring":     method.get("docstring", ""),
            "return_type":   method.get("return_type", "unknown"),
            "parameters":    method.get("parameters", []),
            "calls":         method.get("calls", []),
            "line_number":   method.get("line_number", 0),
            "sha":           crawl_result.get("sha", ""),
        })
        node_ids.append(method_id)

        # HAS_METHOD edge — Class → Method
        edge_id = f"{class_id}:HAS_METHOD:{method_id}"
        await upsert_edge({
            "id":            edge_id,
            "resource_code": resource_code,
            "source_id":     class_id,
            "target_id":     method_id,
            "relationship":  "HAS_METHOD",
        })

    return node_ids


async def load_nodes(
    crawl_result: dict,
    resource_code: str, github_org: str, project: str, repo: str,
) -> dict:
    """
    Pass 1 — create all nodes for one file.

    Creates File, Function, Class, and Method nodes.
    Deletes stale nodes for functions/classes removed from the file.
    Does NOT create CALLS or IMPORTS — those need all nodes to exist first.

    Returns summary with status and node counts.
    """
    file_path = crawl_result["path"]

    # get existing node IDs before upserting — needed for stale detection
    existing_ids = set(await get_node_ids_for_file(resource_code, repo, file_path))
    new_ids: set = set()

    await _upsert_file_node(crawl_result, resource_code, github_org, project, repo)

    for func in crawl_result.get("functions", []):
        node_id = await _upsert_function_node(func, crawl_result, resource_code, github_org, project, repo)
        new_ids.add(node_id)

    for cls in crawl_result.get("classes", []):
        ids = await _upsert_class_and_methods(cls, crawl_result, resource_code, github_org, project, repo)
        new_ids.update(ids)

    # delete stale nodes — functions/classes removed from source
    stale_ids = existing_ids - new_ids
    for node_id in stale_ids:
        await delete_node(node_id, resource_code)

    return {
        "file":    file_path,
        "nodes":   len(new_ids),
        "deleted": len(stale_ids),
    }


async def load_edges(
    crawl_result: dict,
    resource_code: str, github_org: str, project: str, repo: str,
) -> None:
    """
    Pass 2 — create CALLS and IMPORTS edges for one file.

    Run only after all nodes for all files are created (Pass 1 complete).
    Both source and target nodes must exist before the edge is created.
    """
    file_path = crawl_result["path"]

    # IMPORTS edges — File → File
    for imp in crawl_result.get("imports", []):
        module = imp.get("module", "")
        if not module:
            continue

        source_id = _build_id(resource_code, github_org, project, repo, file_path)
        edge_id   = f"{source_id}:IMPORTS:{module}"

        await upsert_edge({
            "id":            edge_id,
            "resource_code": resource_code,
            "source_id":     source_id,
            "target_module": module,
            "relationship":  "IMPORTS",
        })

    # CALLS edges — Function/Method → Function/Method
    all_callables = (
        [(func, _build_id(resource_code, github_org, project, repo, file_path, func["name"]))
         for func in crawl_result.get("functions", [])]
        +
        [(method, _build_id(resource_code, github_org, project, repo, file_path, cls["name"], method["name"]))
         for cls in crawl_result.get("classes", [])
         for method in cls.get("methods", [])]
    )

    for item, caller_id in all_callables:
        for called_name in item.get("calls", []):
            edge_id = f"{caller_id}:CALLS:{called_name}"
            await upsert_edge({
                "id":           edge_id,
                "resource_code": resource_code,
                "source_id":    caller_id,
                "target_name":  called_name,
                "relationship": "CALLS",
            })
