import httpx

def execute_fetch_data(arguments: dict) -> dict:
    url_value = arguments.get("url", "")
    if not isinstance(url_value, str) or not url_value.strip():
        return {"success": False, "message": "fetch_data requires a non-empty 'url' string", "data": {}}

    try:
        response = httpx.get(url_value, timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return {"success": False, "message": f"fetch_data failed: {exc}", "data": {"url": url_value}}

    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = response.json()
    else:
        payload = response.text[:5000]

    return {
        "success": True,
        "message": "Data fetched successfully",
        "data": {"url": url_value, "payload": payload},
    }
