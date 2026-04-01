from pathlib import Path

def execute_read_file(arguments: dict) -> dict:
    path_value = arguments.get("path", "")
    if not isinstance(path_value, str) or not path_value.strip():
        return {"success": False, "message": "read_file requires a non-empty 'path' string", "data": {}}

    target_path = Path(path_value)
    if not target_path.exists():
        return {"success": False, "message": f"File not found: {path_value}", "data": {}}

    content = target_path.read_text(encoding="utf-8")
    return {
        "success": True,
        "message": "File read successfully",
        "data": {"path": path_value, "content": content},
    }
