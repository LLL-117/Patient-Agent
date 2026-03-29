"""
MCP Streamable HTTP 传输（规范 2025-11-25）：单端点 JSON-RPC，**非 SSE**、非 stdio。

- POST /mcp：请求体为 JSON-RPC 2.0；对带 `id` 的请求，响应为 **Content-Type: application/json** 的单条 JSON-RPC 结果（不建立 text/event-stream）。
- 对无 `id` 的通知：返回 **202 Accepted** 且无正文（如 notifications/initialized）。
- GET /mcp：本实现不提供 SSE 监听流，返回 **405 Method Not Allowed**（符合规范「要么 SSE 要么 405」之一）。
- DELETE /mcp：若带 MCP-Session-Id，可终止会话（可选）。

安全：校验 Origin（与规范一致）；生产环境请绑定 127.0.0.1 并加鉴权。
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.responses import Response as StarletteResponse

MCP_PROTOCOL_VERSION = "2025-11-25"
SERVER_NAME = "Patient Modular MCP"
SERVER_VERSION = "0.2.0"


def _jsonrpc_error(
  req_id: Optional[Union[str, int]],
  code: int,
  message: str,
  data: Any = None,
) -> Dict[str, Any]:
  err: Dict[str, Any] = {"code": code, "message": message}
  if data is not None:
    err["data"] = data
  out: Dict[str, Any] = {"jsonrpc": "2.0", "error": err}
  if req_id is not None:
    out["id"] = req_id
  return out


def _jsonrpc_ok(req_id: Union[str, int], result: Any) -> Dict[str, Any]:
  return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _validate_origin(request: Request) -> None:
  origin = request.headers.get("origin")
  if not origin:
    return
  allowed = (
    "http://127.0.0.1",
    "http://localhost",
    "https://127.0.0.1",
    "https://localhost",
    "null",
  )
  if any(origin == a or origin.startswith(a + ":") for a in allowed if a != "null"):
    return
  if origin == "null":
    return
  raise HTTPException(status_code=403, detail="invalid_origin")


def _parse_protocol_version(request: Request) -> Optional[str]:
  return (request.headers.get("mcp-protocol-version") or "").strip() or None


def _check_protocol_version(request: Request) -> None:
  v = _parse_protocol_version(request)
  if not v:
    return
  if v not in ("2025-11-25", "2025-03-26", "2025-06-18"):
    raise HTTPException(status_code=400, detail="unsupported MCP-Protocol-Version")


def _tool_specs_to_mcp_tools(specs: List[Any]) -> List[Dict[str, Any]]:
  out: List[Dict[str, Any]] = []
  for t in specs:
    out.append(
      {
        "name": t.name,
        "description": t.description,
        "inputSchema": t.input_schema,
      }
    )
  return out


class MCPSessionStore:
  """进程内会话（演示用）；生产可换 Redis 等。"""

  def __init__(self) -> None:
    self._ids: Dict[str, bool] = {}

  def create(self) -> str:
    sid = uuid.uuid4().hex
    self._ids[sid] = True
    return sid

  def valid(self, sid: Optional[str]) -> bool:
    if not sid:
      return False
    return sid in self._ids

  def drop(self, sid: str) -> None:
    self._ids.pop(sid, None)


def handle_mcp_jsonrpc(
  *,
  request: Request,
  body: Dict[str, Any],
  toolbox: Any,
  sessions: MCPSessionStore,
) -> Tuple[Optional[Dict[str, Any]], int, Dict[str, str]]:
  """
  处理一条 JSON-RPC。返回 (response_dict 或 None, http_status, extra_headers)。
  response_dict 为 None 且 status 202 表示无 JSON 正文。
  """
  _validate_origin(request)
  _check_protocol_version(request)

  if body.get("jsonrpc") != "2.0":
    raise HTTPException(status_code=400, detail="jsonrpc must be 2.0")

  has_id_key = "id" in body
  req_id = body.get("id")
  method = body.get("method")
  params = body.get("params") if isinstance(body.get("params"), dict) else {}

  # JSON-RPC Notification（不得含 id 字段）：须 202 无 body
  if not has_id_key:
    if method:
      return None, 202, {}
    raise HTTPException(status_code=400, detail="missing method")

  if req_id is None:
    raise HTTPException(status_code=400, detail="invalid id")

  if not method:
    return (
      _jsonrpc_error(req_id, -32600, "Invalid Request: missing method"),
      200,
      {},
    )

  session_hdr = (request.headers.get("mcp-session-id") or "").strip()

  if method == "initialize":
    result = {
      "protocolVersion": MCP_PROTOCOL_VERSION,
      "capabilities": {
        "tools": {"listChanged": False},
      },
      "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
    }
    sid = sessions.create()
    headers = {
      "MCP-Session-Id": sid,
      "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
    }
    return _jsonrpc_ok(req_id, result), 200, headers

  if method == "notifications/initialized":
    # 带 id 的误发：按请求处理
    return _jsonrpc_ok(req_id, {}), 200, {}

  if not sessions.valid(session_hdr):
    return (
      _jsonrpc_error(req_id, -32001, "Session invalid or missing; call initialize first"),
      400,
      {},
    )

  if method == "tools/list":
    specs = toolbox.list_specs()
    return (
      _jsonrpc_ok(req_id, {"tools": _tool_specs_to_mcp_tools(specs)}),
      200,
      {"MCP-Session-Id": session_hdr, "MCP-Protocol-Version": MCP_PROTOCOL_VERSION},
    )

  if method == "tools/call":
    name = params.get("name")
    arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
    if not name or not isinstance(name, str):
      return (
        _jsonrpc_error(req_id, -32602, "Invalid params: tools/call requires name"),
        200,
        {"MCP-Session-Id": session_hdr},
      )
    try:
      raw = toolbox.invoke(name, arguments)
    except Exception as e:
      return (
        _jsonrpc_ok(
          req_id,
          {
            "content": [{"type": "text", "text": str(e)}],
            "isError": True,
          },
        ),
        200,
        {"MCP-Session-Id": session_hdr, "MCP-Protocol-Version": MCP_PROTOCOL_VERSION},
      )
    text = json.dumps(raw, ensure_ascii=False)
    return (
      _jsonrpc_ok(
        req_id,
        {
          "content": [{"type": "text", "text": text}],
          "isError": False,
        },
      ),
      200,
      {"MCP-Session-Id": session_hdr, "MCP-Protocol-Version": MCP_PROTOCOL_VERSION},
    )

  return (
    _jsonrpc_error(req_id, -32601, f"Method not found: {method}"),
    200,
    {"MCP-Session-Id": session_hdr},
  )


async def run_mcp_post(
  request: Request,
  toolbox: Any,
  sessions: MCPSessionStore,
) -> Response:
  try:
    body = await request.json()
  except Exception:
    raise HTTPException(status_code=400, detail="invalid JSON body")
  if not isinstance(body, dict):
    raise HTTPException(status_code=400, detail="JSON body must be an object")

  payload, status, extra_headers = handle_mcp_jsonrpc(
    request=request, body=body, toolbox=toolbox, sessions=sessions
  )

  if status == 202:
    return StarletteResponse(status_code=202, headers=extra_headers)

  assert payload is not None
  headers = {"Content-Type": "application/json; charset=utf-8", **extra_headers}
  return JSONResponse(content=payload, status_code=status, headers=headers)


def mcp_endpoint_delete(request: Request, sessions: MCPSessionStore) -> Response:
  _validate_origin(request)
  sid = (request.headers.get("mcp-session-id") or "").strip()
  if not sid:
    raise HTTPException(status_code=400, detail="MCP-Session-Id required")
  sessions.drop(sid)
  return StarletteResponse(status_code=204)
