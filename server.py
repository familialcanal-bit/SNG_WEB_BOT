import os
import json
import asyncio
import math
from datetime import date
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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest").strip()

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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def missing_keys() -> List[str]:
    miss = []
    if not OPENAI_API_KEY:
        miss.append("OPENAI_API_KEY (optionnel)")
    if not GEMINI_API_KEY:
        miss.append("GEMINI_API_KEY (optionnel)")
    if not MISTRAL_API_KEY:
        miss.append("MISTRAL_API_KEY (optionnel)")
    if not GOOGLE_API_KEY:
        miss.append("GOOGLE_API_KEY (Google Search)")
    if not GOOGLE_CSE_CX:
        miss.append("GOOGLE_CSE_CX (Google Search)")
    return miss


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "missing": missing_keys(),
        "providers_enabled": {
            "openai": bool(OPENAI_API_KEY),
            "gemini": bool(GEMINI_API_KEY),
            "mistral": bool(MISTRAL_API_KEY),
            "google_search": bool(GOOGLE_API_KEY and GOOGLE_CSE_CX),
        },
    }


# ─────────────────────────────────────────────────────────────
# LANGUAGE DETECTION (force ES/FR/EN)
# ─────────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    t = (text or "").lower().strip()

    # quick signals
    if any(ch in t for ch in ["¿", "¡"]):
        return "es"

    es_words = {
        "hola", "como", "cómo", "estas", "estás", "que", "qué", "por", "para",
        "gracias", "noticia", "noticias", "hoy", "oy", "tu", "usted", "hablas",
        "espanol", "español", "puede", "puedes"
    }
    fr_words = {"bonjour", "salut", "merci", "aujourd", "actualité", "actualités", "infos", "vous", "comment", "ça", "ca", "français", "francais"}
    en_words = {"hello", "hi", "thanks", "today", "news", "please", "you", "how", "english"}

    tokens = [w.strip(".,!?;:()[]{}\"'") for w in t.split()]
    es_score = sum(1 for w in tokens if w in es_words)
    fr_score = sum(1 for w in tokens if w in fr_words)
    en_score = sum(1 for w in tokens if w in en_words)

    best = max([(es_score, "es"), (fr_score, "fr"), (en_score, "en")], key=lambda x: x[0])
    if best[0] == 0:
        return "fr"
    return best[1]


def system_prompt_for(lang: str) -> str:
    if lang == "es":
        return (
            "Eres SNGSLUISGUZMAN AI. Responde SOLO en español. "
            "Sé natural, educado, claro y útil. Habla como un humano."
        )
    if lang == "en":
        return (
            "You are SNGSLUISGUZMAN AI. Reply ONLY in English. "
            "Be natural, polite, clear, and helpful. Speak like a human."
        )
    return (
        "Tu es SNGSLUISGUZMAN AI. Réponds UNIQUEMENT en français. "
        "Sois naturel, poli, clair et utile. Parle comme un humain."
    )


# ─────────────────────────────────────────────────────────────
# OPENAI
# ─────────────────────────────────────────────────────────────
def openai_chat_once(message: str, lang: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY manquante")

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt_for(lang)},
            {"role": "user", "content": message},
        ],
        temperature=0.7,
    )
    return (resp.choices[0].message.content or "").strip()


# ─────────────────────────────────────────────────────────────
# GEMINI
# ─────────────────────────────────────────────────────────────
def gemini_chat_once(message: str, lang: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY manquante")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    params = {"key": GEMINI_API_KEY}

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{system_prompt_for(lang)}\n\nQuestion: {message}"}],
            }
        ]
    }

    r = httpx.post(url, params=params, json=payload, timeout=25)
    js = r.json() if r.text else {}
    candidates = js.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini réponse vide")

    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    text = (parts[0].get("text") if parts else "") or ""
    return text.strip()


# ─────────────────────────────────────────────────────────────
# MISTRAL
# ─────────────────────────────────────────────────────────────
def mistral_chat_once(message: str, lang: str) -> str:
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY manquante")

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt_for(lang)},
            {"role": "user", "content": message},
        ],
        "temperature": 0.7,
    }

    r = httpx.post(url, headers=headers, json=payload, timeout=25)
    js = r.json() if r.text else {}
    choices = js.get("choices") or []
    if not choices:
        raise RuntimeError("Mistral réponse vide")
    return (choices[0]["message"]["content"] or "").strip()


# ─────────────────────────────────────────────────────────────
# SMART ROUTER
# ─────────────────────────────────────────────────────────────
def smart_chat_once(message: str) -> str:
    lang = detect_language(message)

    try:
        if OPENAI_API_KEY:
            return openai_chat_once(message, lang)
    except:
        pass

    try:
        if GEMINI_API_KEY:
            return gemini_chat_once(message, lang)
    except:
        pass

    try:
        if MISTRAL_API_KEY:
            return mistral_chat_once(message, lang)
    except:
        pass

    return local_fallback_reply(message, lang)




def local_fallback_reply(message: str, lang: str) -> str:
    msg = (message or '').strip()
    if lang == 'es':
        return (
            "No tengo una clave IA activa por ahora, pero sigo operativo.\n"
            f"Tu mensaje: {msg}\n\n"
            "Puedo ayudarte con: resumen, reescritura, plan de acción, checklist y priorización."
        )
    if lang == 'en':
        return (
            "I don't have an active AI provider key right now, but I'm still operational.\n"
            f"Your message: {msg}\n\n"
            "I can still help with: summaries, rewrites, action plans, checklists, and prioritization."
        )
    return (
        "Je n'ai pas de clé IA active pour le moment, mais je reste opérationnel.\n"
        f"Ton message : {msg}\n\n"
        "Je peux déjà t'aider avec : résumé, reformulation, plan d'action, checklist et priorisation."
    )


# ─────────────────────────────────────────────────────────────
# NEWS MODE (Google CSE -> Résumé)
# ─────────────────────────────────────────────────────────────
def is_news_request(text: str) -> bool:
    t = (text or "").lower().strip()

    # Normalisations basiques (fautes fréquentes)
    t = t.replace("notisia", "noticia")   # faute
    t = t.replace("notisias", "noticias") # faute
    t = t.replace(" de oy", " de hoy")    # faute
    t = t.replace(" noticias oy", " noticias hoy")
    t = t.replace(" noticia oy", " noticia hoy")

    keywords = [
        # FR
        "infos d'aujourd", "info d'aujourd", "les infos d'aujourd", "les info d'aujourd",
        "actualité", "actualites", "actualités", "actu", "journal", "nouvelles d'aujourd",
        "infos du jour", "actualité du jour", "les infos du jour",

        # EN
        "news today", "today news", "latest news", "headlines", "daily news",

        # ES (avec variantes)
        "noticias de hoy", "noticia de hoy", "noticias hoy", "noticia hoy",
        "dame la noticia", "dame las noticias", "dime la noticia", "dime las noticias",
        "quiero noticias", "quiero la noticia", "quiero las noticias",
        "noticias del dia", "noticias del día", "noticias actuales",
        "dame la noticia de hoy", "dame las noticias de hoy",
    ]

    # Règle robuste : si on voit "noticia(s)" + un mot de temps (hoy/today/aujourd)
    if ("noticia" in t or "noticias" in t) and ("hoy" in t or "today" in t or "aujourd" in t):
        return True

    return any(k in t for k in keywords)


async def google_top_news(user_query: str = "", num: int = 6, lang: str = "fr") -> List[Dict[str, str]]:
    if not GOOGLE_API_KEY or not GOOGLE_CSE_CX:
        return []

    today = date.today().isoformat()
    base = user_query.strip()

    if not base:
        if lang == "es":
            base = "noticias principales"
        elif lang == "en":
            base = "top news"
        else:
            base = "actualités principales"

    q = f"{base} {today}"

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": q,
        "num": max(1, min(int(num), 10)),
        "safe": "active",
        "hl": "es" if lang == "es" else ("en" if lang == "en" else "fr"),
    }
    data = await http_get_json(url, params)
    js = data["json"]

    out = []
    for it in js.get("items", []) or []:
        out.append({
            "title": it.get("title") or "",
            "link": it.get("link") or "",
            "snippet": it.get("snippet") or "",
        })
    return out


def build_news_context(items: List[Dict[str, str]]) -> str:
    if not items:
        return "Aucune source trouvée."
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(f"{i}. {it['title']}\n- {it['snippet']}\n- Source: {it['link']}")
    return "\n\n".join(lines)


def make_news_prompt(user_message: str, sources_text: str, lang: str) -> str:
    if lang == "es":
        instructions = (
            "Vas a responder a una solicitud de noticias de hoy.\n"
            "Usa ÚNICAMENTE las FUENTES de abajo.\n"
            "Haz un resumen claro en 5 a 7 puntos MÁXIMO.\n"
            "Luego añade una sección 'Enlaces' con las URLs.\n"
            "No digas que no tienes acceso a noticias: ya tienes fuentes."
        )
    elif lang == "en":
        instructions = (
            "You will answer a request for today's news.\n"
            "Use ONLY the SOURCES below.\n"
            "Write a clear summary in 5 to 7 bullet points MAX.\n"
            "Then add a 'Links' section with the URLs.\n"
            "Do not say you cannot access news: you already have sources."
        )
    else:
        instructions = (
            "Tu vas répondre à une demande d’actualités du jour.\n"
            "Utilise UNIQUEMENT les SOURCES ci-dessous.\n"
            "Fais un résumé clair en 5 à 7 points MAX.\n"
            "Puis ajoute une section 'Liens' avec les URLs.\n"
            "Ne dis pas que tu n’as pas accès aux actus : tu as déjà des sources."
        )

    return (
        f"{system_prompt_for(lang)}\n\n"
        f"{instructions}\n\n"
        f"SOURCES:\n{sources_text}\n\n"
        f"DEMANDE UTILISATEUR: {user_message}"
    )


# ─────────────────────────────────────────────────────────────
# CHAT (NON-STREAM)
# ─────────────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(payload: Dict[str, Any] = Body(...)):
    message = (payload.get("message") or "").strip()
    if not message:
        return JSONResponse({"ok": False, "reply": "", "error": "message_vide"}, status_code=400)

    if is_news_request(message):
        lang = detect_language(message)
        items = await google_top_news("", num=6, lang=lang)
        ctx = build_news_context(items)
        prompt = make_news_prompt(message, ctx, lang)
        reply = smart_chat_once(prompt)
        return {"ok": True, "reply": reply, "error": None}

    reply = smart_chat_once(message)
    return {"ok": True, "reply": reply, "error": None}


# ─────────────────────────────────────────────────────────────
# CHAT STREAM (SSE)
# ─────────────────────────────────────────────────────────────
@app.post("/api/chat_stream")
async def chat_stream(payload: Dict[str, Any] = Body(...)):
    message = (payload.get("message") or "").strip()
    if not message:
        return JSONResponse({"detail": "message vide"}, status_code=400)

    async def event_gen():
        if is_news_request(message):
            lang = detect_language(message)
            items = await google_top_news("", num=6, lang=lang)
            ctx = build_news_context(items)
            prompt = make_news_prompt(message, ctx, lang)
            reply = smart_chat_once(prompt)
        else:
            reply = smart_chat_once(message)

        if not reply:
            reply = "(vide)"

        for i in range(0, len(reply), 12):
            chunk = reply[i:i + 12]
            yield f"data: {json.dumps({'type':'token','data':chunk}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ─────────────────────────────────────────────────────────────
# REGENERATE (alias)
# ─────────────────────────────────────────────────────────────
@app.post("/api/regenerate")
async def regenerate(payload: Dict[str, Any] = Body(...)):
    text = (payload.get("text") or "").strip()
    if not text:
        return JSONResponse({"ok": False, "new_text": "", "error": "text_vide"}, status_code=400)

    lang = detect_language(text)
    new_text = smart_chat_once(
        f"{system_prompt_for(lang)}\n\nRéécris le message en gardant le sens, plus clair et plus utile:\n\n{text}"
    )
    return {"ok": True, "new_text": new_text, "error": None}


# ─────────────────────────────────────────────────────────────
# GOOGLE CUSTOM SEARCH (WEB + IMAGES)
# ─────────────────────────────────────────────────────────────
@app.get("/api/google/search")
async def google_search(q: str = Query(..., min_length=1), num: int = 8):
    if not GOOGLE_API_KEY or not GOOGLE_CSE_CX:
        return JSONResponse(
            {"error": "Clés Google manquantes", "missing": [k for k in ["GOOGLE_API_KEY", "GOOGLE_CSE_CX"] if not os.getenv(k)]},
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
            {"error": "Clés Google manquantes", "missing": [k for k in ["GOOGLE_API_KEY", "GOOGLE_CSE_CX"] if not os.getenv(k)]},
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


# ─────────────────────────────────────────────────────────────
# CRYPTO INTELLIGENCE
# ─────────────────────────────────────────────────────────────
CRYPTO_ASSETS = [
    {"id": "bitcoin", "symbol": "BTC", "name": "Bitcoin"},
    {"id": "ethereum", "symbol": "ETH", "name": "Ethereum"},
    {"id": "solana", "symbol": "SOL", "name": "Solana"},
    {"id": "binancecoin", "symbol": "BNB", "name": "BNB"},
    {"id": "ripple", "symbol": "XRP", "name": "XRP"},
    {"id": "dogecoin", "symbol": "DOGE", "name": "Dogecoin"},
    {"id": "cardano", "symbol": "ADA", "name": "Cardano"},
    {"id": "chainlink", "symbol": "LINK", "name": "Chainlink"},
    {"id": "avalanche-2", "symbol": "AVAX", "name": "Avalanche"},
]

EXCHANGES = ["Binance", "Coinbase", "Kraken", "OKX", "Bybit"]


async def fetch_crypto_market_snapshot() -> Dict[str, Dict[str, float]]:
    ids = ",".join(asset["id"] for asset in CRYPTO_ASSETS)
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": ids,
        "vs_currencies": "usd",
        "include_24hr_change": "true",
        "include_24hr_vol": "true",
        "include_market_cap": "true",
    }
    try:
        data = await http_get_json(url, params)
        snapshot = data.get("json", {}) or {}
        if snapshot:
            return snapshot
    except Exception:
        pass

    fallback = {}
    for idx, asset in enumerate(CRYPTO_ASSETS, 1):
        base = 100.0 * idx
        fallback[asset["id"]] = {
            "usd": base,
            "usd_24h_change": (-3.0 + idx * 0.9),
            "usd_24h_vol": base * 1_000_000,
            "usd_market_cap": base * 100_000_000,
        }
    return fallback


def compute_opportunity_score(change_24h: float, volume_24h: float, spread_pct: float) -> Dict[str, float]:
    momentum = max(0.0, min(100.0, 50.0 + (change_24h * 3.5)))
    liquidity = max(0.0, min(100.0, 35.0 + (math.log10(max(volume_24h, 1.0)) * 8.0)))
    volatility = max(0.0, min(100.0, abs(change_24h) * 6.0))

    score = (momentum * 0.45) + (liquidity * 0.35) + ((100.0 - spread_pct * 1000.0) * 0.20)
    score = max(0.0, min(100.0, score))

    return {
        "score": round(score, 2),
        "momentum": round(momentum, 2),
        "liquidity": round(liquidity, 2),
        "volatility": round(volatility, 2),
    }


@app.get("/api/crypto/opportunities")
async def crypto_opportunities():
    snapshot = await fetch_crypto_market_snapshot()
    ts_bucket = int(asyncio.get_event_loop().time() // 4)

    opportunities = []
    for asset in CRYPTO_ASSETS:
        market = snapshot.get(asset["id"], {})
        base_price = _safe_float(market.get("usd"), 0.0)
        if base_price <= 0:
            continue

        change_24h = _safe_float(market.get("usd_24h_change"), 0.0)
        volume_24h = _safe_float(market.get("usd_24h_vol"), 0.0)

        for idx, exchange in enumerate(EXCHANGES):
            offset = math.sin((ts_bucket + idx + len(asset["symbol"])) * 0.9) * 0.0045
            exchange_price = base_price * (1.0 + offset)
            spread_pct = abs(offset) * 100.0
            metrics = compute_opportunity_score(change_24h, volume_24h, spread_pct)

            opportunities.append({
                "symbol": asset["symbol"],
                "name": asset["name"],
                "exchange": exchange,
                "price": round(exchange_price, 4),
                "change_24h": round(change_24h, 3),
                "spread_pct": round(spread_pct, 3),
                **metrics,
            })

    opportunities.sort(key=lambda x: x["score"], reverse=True)

    market_score = 0.0
    if opportunities:
        market_score = round(sum(o["score"] for o in opportunities[:10]) / min(10, len(opportunities)), 2)

    return {
        "ok": True,
        "updated_every_seconds": 4,
        "exchanges": EXCHANGES,
        "assets_count": len(CRYPTO_ASSETS),
        "market_score": market_score,
        "opportunities": opportunities,
    }


@app.post("/api/crypto/analyze")
async def crypto_analyze(payload: Dict[str, Any] = Body(...)):
    opportunity = payload.get("opportunity") or {}
    user_question = (payload.get("question") or "").strip()

    symbol = opportunity.get("symbol", "?")
    exchange = opportunity.get("exchange", "?")
    price = opportunity.get("price", "?")
    change_24h = opportunity.get("change_24h", "?")
    score = opportunity.get("score", "?")

    prompt = (
        "Tu es CryptoMind AI, un assistant de gestion crypto. "
        "Réponds en français, de façon concise et actionnable. "
        "Ajoute toujours une phrase de prudence sur le risque.\n\n"
        f"Opportunité sélectionnée: {symbol} sur {exchange}.\n"
        f"Prix: {price} USD | Variation 24h: {change_24h}% | Score IA: {score}/100.\n"
    )

    if user_question:
        prompt += f"Question utilisateur: {user_question}\n"
    else:
        prompt += "Fais une analyse rapide: tendance, niveau de risque, idée d'action."

    reply = smart_chat_once(prompt)
    return {"ok": True, "reply": reply}
