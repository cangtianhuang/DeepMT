"""
DeepMT Terminal — WebSocket 实时命令执行与输出流

路由:
  GET /terminal      → 终端页面
  WS  /terminal/ws  → WebSocket 命令流
"""

import asyncio
import os
import re
import shlex
import sys
from pathlib import Path

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from deepmt.ui.templating import templates

router = APIRouter()

# ── ANSI SGR → CSS 颜色（与 neon-lab 主题配套）──────────────────────────
_ANSI_FG: dict[str, str] = {
    "30": "#3d5570",  # black  → dim
    "31": "#ff5252",  # red    → danger
    "32": "#00e5b0",  # green  → success
    "33": "#ffcc00",  # yellow → warning
    "34": "#4d8bff",  # blue   → primary
    "35": "#c4b5fd",  # magenta
    "36": "#00e5ff",  # cyan   → accent
    "37": "#c8daf0",  # white  → text
    "90": "#546e7a",
    "91": "#ff7070",
    "92": "#4dffd9",
    "93": "#ffe066",
    "94": "#7ec8f4",
    "95": "#d4b8ff",
    "96": "#66eeff",
    "97": "#f0f6ff",
}

_ANSI_RE = re.compile(r"\x1b\[([0-9;]*)([A-Za-z])")


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _ansi_to_html(raw: str) -> str:
    """将含 ANSI SGR 转义码的文本转换为带样式 HTML（仅处理颜色/粗体/重置）。"""
    parts: list[str] = []
    open_spans = 0
    last = 0

    for m in _ANSI_RE.finditer(raw):
        parts.append(_escape_html(raw[last : m.start()]))
        last = m.end()

        if m.group(2) != "m":
            continue  # 非颜色控制符，跳过

        codes = m.group(1).split(";") if m.group(1) else ["0"]
        for code in codes:
            code = code.strip() or "0"
            if code == "0":
                if open_spans:
                    parts.append("</span>" * open_spans)
                    open_spans = 0
            elif code == "1":
                parts.append('<span style="font-weight:700;color:var(--text-bright)">')
                open_spans += 1
            elif code in _ANSI_FG:
                parts.append(f'<span style="color:{_ANSI_FG[code]}">')
                open_spans += 1

    parts.append(_escape_html(raw[last:]))
    if open_spans:
        parts.append("</span>" * open_spans)
    return "".join(parts)


# ── 页面路由 ─────────────────────────────────────────────────────────────

@router.get("/terminal", response_class=HTMLResponse, include_in_schema=False)
async def terminal_page(request: Request) -> HTMLResponse:
    """终端页面。"""
    return templates.TemplateResponse(
        request,
        "terminal.html",
        context={"active_page": "terminal"},
    )


# ── WebSocket 命令执行 ───────────────────────────────────────────────────

@router.websocket("/terminal/ws")
async def terminal_ws(ws: WebSocket) -> None:
    """
    WebSocket 端点：接收命令，流式返回输出。

    客户端 → 服务端消息：
      {"type": "run",  "cmd": "deepmt health check"}
      {"type": "kill"}
      {"type": "pong"}

    服务端 → 客户端消息：
      {"type": "start", "cmd": "..."}
      {"type": "line",  "html": "..."}
      {"type": "done",  "code": 0}
      {"type": "killed"}
      {"type": "error", "msg": "..."}
      {"type": "ping"}
    """
    await ws.accept()
    process: asyncio.subprocess.Process | None = None

    async def _send(payload: dict) -> None:
        try:
            await ws.send_json(payload)
        except Exception:
            pass

    try:
        while True:
            # 带超时的心跳：30 秒无消息则发 ping
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=30.0)
            except asyncio.TimeoutError:
                await _send({"type": "ping"})
                continue

            t = msg.get("type")

            if t == "pong":
                continue

            if t == "kill":
                if process and process.returncode is None:
                    try:
                        process.kill()
                    except Exception:
                        pass
                await _send({"type": "killed"})
                continue

            if t != "run":
                continue

            raw_cmd = msg.get("cmd", "").strip()
            if not raw_cmd:
                continue

            # ── 命令解析 ────────────────────────────────────────────
            try:
                parts = shlex.split(raw_cmd)
            except ValueError as exc:
                await _send({"type": "error", "msg": f"命令解析失败：{exc}"})
                continue

            if not parts:
                continue

            # 支持 "deepmt health check" 和 "health check" 两种写法
            if parts[0] == "deepmt":
                sub_args = parts[1:]
            else:
                sub_args = parts

            if not sub_args:
                continue

            await _send({"type": "start", "cmd": "deepmt " + " ".join(sub_args)})

            # ── 子进程 ──────────────────────────────────────────────
            try:
                env = {**os.environ, "PYTHONUNBUFFERED": "1"}
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "deepmt",
                    *sub_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=str(Path.cwd()),
                    env=env,
                )

                assert process.stdout is not None
                async for raw_bytes in process.stdout:
                    line = raw_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
                    await _send({"type": "line", "html": _ansi_to_html(line)})

                await process.wait()
                await _send({"type": "done", "code": process.returncode})

            except Exception as exc:
                await _send({"type": "error", "msg": str(exc)})

    except WebSocketDisconnect:
        pass
    finally:
        if process and process.returncode is None:
            try:
                process.kill()
            except Exception:
                pass
