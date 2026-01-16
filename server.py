import os
import json
import asyncio
from typing import Dict, Any, List

import httpx
from fastapi import FastAPI, Query, Body
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
INDEX_FILE = os.path.join(STATIC_DIR, "index.html")

load_dotenv(os.path.join(BASE_DIR, ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GOOGLE_CSE_CX = os.getenv("GOOGLE_CSE_CX", "").strip()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="SNG_WEB_BOT")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def home():
    if os.path.exists(INDEX_FILE):
        return FileResponse(INDEX_FILE)
    return JSONResponse(
        {"error": "index.html introuvable", "hint": "Crée static/index.html"},
        status_code=404,
    )


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
async def http_get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.get(url, params=params)
        return {"status": r.status_code, "json": r.json() if r.text else {}}


def missing_keys() -> List[str]:
    miss = []
    if not OPENAI_API_KEY:
        miss.append("OPENAI_API_KEY")
    if not GOOGLE_API_KEY:
        miss.append("GOOGLE_API_KEY")
    if not GOOGLE_CSE_CX:
        miss.append("GOOGLE_CSE_CX")
    return miss


@app.get("/api/health")
def health():
    return {"status": "ok", "missing": missing_keys(), "model": OPENAI_MODEL}


# ─────────────────────────────────────────────────────────────
# OPENAI CORE
# ─────────────────────────────────────────────────────────────
def openai_chat_once(message: str) -> str:
    if not OPENAI_API_KEY:
        return "⚠️ OPENAI_API_KEY manquante dans .env (mets ta clé puis relance le serveur)."

    # Import ici pour éviter crash si module pas installé
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Tu es SNGSLUISGUZMAN AI. Réponds en français, clairement et efficacement."},
            {"role": "user", "content": message},
        ],
        temperature=0.7,
    )
    return (resp.choices[0].message.content or "").strip()


# ─────────────────────────────────────────────────────────────
# CHAT (NON-STREAM) → utilisé par tests / et possible front
# ─────────────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(payload: Dict[str, Any] = Body(...)):
    message = (payload.get("message") or "").strip()
    if not message:
        return JSONResponse({"ok": False, "reply": "", "error": "message_vide"}, status_code=400)

    try:
        reply = openai_chat_once(message)
        return {"ok": True, "reply": reply, "error": None}
    except Exception as e:
        return JSONResponse(
            {"ok": False, "reply": "❌ Erreur serveur", "error": "openai_error", "details": str(e)},
            status_code=500
        )


# ─────────────────────────────────────────────────────────────
# CHAT STREAM (SSE) → EXACTEMENT ce que ton index.html attend
# Route attendue par ton front: POST /api/chat_stream
# Format SSE: data: {"type":"token","data":"..."}\n\n
# ─────────────────────────────────────────────────────────────
@app.post("/api/chat_stream")
async def chat_stream(payload: Dict[str, Any] = Body(...)):
    message = (payload.get("message") or "").strip()
    if not message:
        return JSONResponse({"detail": "message vide"}, status_code=400)

    async def event_gen():
        try:
            # 1) récupère réponse complète via OpenAI
            reply = openai_chat_once(message)
            if not reply:
                reply = "(vide)"

            # 2) stream “token” compatible avec ton parser JS
            #    (on découpe la réponse pour donner un effet streaming)
            chunk_size = 12
            for i in range(0, len(reply), chunk_size):
                chunk = reply[i:i + chunk_size]
                data = {"type": "token", "data": chunk}
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)

        except Exception as e:
            data = {"type": "token", "data": f"\n\n❌ Erreur serveur: {str(e)}"}
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ─────────────────────────────────────────────────────────────
# REGENERATE (alias)
# Ton front peut appeler /api/regenerate
# ─────────────────────────────────────────────────────────────
@app.post("/api/regenerate")
async def regenerate(payload: Dict[str, Any] = Body(...)):
    text = (payload.get("text") or "").strip()
    if not text:
        return JSONResponse({"ok": False, "new_text": "", "error": "text_vide"}, status_code=400)

    if not OPENAI_API_KEY:
        return {"ok": True, "new_text": "⚠️ OPENAI_API_KEY manquante dans .env."}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Réécris le message en gardant le sens, plus clair et plus utile."},
                {"role": "user", "content": text},
            ],
            temperature=0.8,
        )
        new_text = (resp.choices[0].message.content or "").strip()
        return {"ok": True, "new_text": new_text, "error": None}

    except Exception as e:
        return JSONResponse(
            {"ok": False, "new_text": "", "error": "openai_error", "details": str(e)},
            status_code=500
        )


# ─────────────────────────────────────────────────────────────
# GOOGLE CUSTOM SEARCH (WEB + IMAGES)
# ─────────────────────────────────────────────────────────────
@app.get("/api/google/search")
async def google_search(q: str = Query(..., min_length=1), num: int = 8):
    if not GOOGLE_API_KEY or not GOOGLE_CSE_CX:
        return JSONResponse(
            {"error": "Clés Google manquantes", "missing": [k for k in ["GOOGLE_API_KEY","GOOGLE_CSE_CX"] if not os.getenv(k)]},
            status_code=400,
        )

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": q,
        "num": max(1, min(int(num), 10)),
        "safe": "active",
    }
    data = await http_get_json(url, params)
    js = data["json"]

    items = []
    for it in js.get("items", []) or []:
        items.append({
            "title": it.get("title"),
            "link": it.get("link"),
            "snippet": it.get("snippet"),
        })

    return {"items": items, "raw": js}


@app.get("/api/google/images")
async def google_images(q: str = Query(..., min_length=1), num: int = 8):
    if not GOOGLE_API_KEY or not GOOGLE_CSE_CX:
        return JSONResponse(
            {"error": "Clés Google manquantes", "missing": [k for k in ["GOOGLE_API_KEY","GOOGLE_CSE_CX"] if not os.getenv(k)]},
            status_code=400,
        )

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": q,
        "searchType": "image",
        "num": max(1, min(int(num), 10)),
        "safe": "active",
    }
    data = await http_get_json(url, params)
    js = data["json"]

    items = []
    for it in js.get("items", []) or []:
        image = it.get("image", {}) or {}
        items.append({
            "title": it.get("title"),
            "link": it.get("link"),
            "thumbnail": image.get("thumbnailLink") or it.get("link"),
            "contextLink": image.get("contextLink"),
        })

    return {"items": items, "raw": js}


# ─────────────────────────────────────────────────────────────
# GOOGLE MAPS / PLACES
# ─────────────────────────────────────────────────────────────
@app.get("/api/google/place")
async def place_search(q: str = Query(..., min_length=2)):
    if not GOOGLE_API_KEY:
        return JSONResponse({"error": "GOOGLE_API_KEY manquante"}, status_code=400)

    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": q, "key": GOOGLE_API_KEY}
    data = await http_get_json(url, params)

    return {"raw": data["json"]}


@app.get("/api/google/place_details")
async def place_details(place_id: str = Query(..., min_length=5)):
    if not GOOGLE_API_KEY:
        return JSONResponse({"error": "GOOGLE_API_KEY manquante"}, status_code=400)

    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "key": GOOGLE_API_KEY,
        "fields": "name,formatted_address,geometry,photos,url,website,formatted_phone_number",
    }
    data = await http_get_json(url, params)
    js = data["json"]
    return {"result": js.get("result", {}), "raw": js}


@app.get("/api/google/place_photo")
async def place_photo(ref: str = Query(..., min_length=5), maxwidth: int = 800):
    if not GOOGLE_API_KEY:
        return JSONResponse({"error": "GOOGLE_API_KEY manquante"}, status_code=400)

    url = "https://maps.googleapis.com/maps/api/place/photo"
    params = {"photoreference": ref, "maxwidth": max(200, min(int(maxwidth), 1600)), "key": GOOGLE_API_KEY}

    async with httpx.AsyncClient(timeout=25, follow_redirects=False) as client:
        r = await client.get(url, params=params)
        if r.status_code in (301, 302, 303, 307, 308):
            loc = r.headers.get("location")
            if loc:
                return RedirectResponse(loc)
        return JSONResponse({"error": "Photo indisponible", "status": r.status_code}, status_code=400)
