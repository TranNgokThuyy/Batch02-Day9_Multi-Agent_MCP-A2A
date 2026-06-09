"""Small web chat server for the A2A Customer Agent.

Run:
    uv run python web_chat_server.py

Then open:
    http://localhost:8080
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
CUSTOMER_AGENT_URL = os.getenv("CUSTOMER_AGENT_URL", "http://localhost:10100")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("A2A_REQUEST_TIMEOUT", "600"))

app = FastAPI(title="Legal Multi-Agent Web Chat")


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Serve the chat UI."""
    return (APP_DIR / "web_chat.html").read_text(encoding="utf-8")


@app.get("/api/health")
async def health() -> dict:
    """Check whether the Customer Agent is reachable."""
    card_url = f"{CUSTOMER_AGENT_URL}/.well-known/agent.json"
    timeout = httpx.Timeout(10.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(card_url)
            response.raise_for_status()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Customer Agent is not reachable at {card_url}: {exc}",
            ) from exc
    card = response.json()
    return {
        "ok": True,
        "agent": card.get("name", "Customer Agent"),
        "url": CUSTOMER_AGENT_URL,
    }


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict:
    """Proxy a user message to the Customer Agent via A2A JSON-RPC."""
    text = request.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message is required.")

    payload = {
        "id": str(uuid4()),
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "kind": "message",
                "messageId": str(uuid4()),
                "parts": [{"kind": "text", "text": text}],
                "role": "user",
            }
        },
    }

    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(CUSTOMER_AGENT_URL, json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=504,
                detail=(
                    "Customer Agent timed out. Restart services so the latest "
                    "timeout/fallback fixes are loaded, or increase A2A_REQUEST_TIMEOUT."
                ),
            ) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    data = response.json()
    if "error" in data:
        raise HTTPException(status_code=502, detail=data["error"])

    return {
        "text": extract_text(data),
        "raw": data,
    }


def extract_text(data: dict) -> str:
    """Extract text from common A2A Task/Message response shapes."""
    result = data.get("result", {})
    chunks: list[str] = []

    for artifact in result.get("artifacts", []) or []:
        for part in artifact.get("parts", []) or []:
            if part.get("kind") == "text" and part.get("text"):
                chunks.append(part["text"])

    for part in result.get("parts", []) or []:
        if part.get("kind") == "text" and part.get("text"):
            chunks.append(part["text"])

    for message in result.get("history", []) or []:
        for part in message.get("parts", []) or []:
            if part.get("kind") == "text" and part.get("text"):
                chunks.append(part["text"])

    return "\n".join(chunks).strip() or "No text response received."


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
