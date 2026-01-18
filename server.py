import os
import base64
import json
import asyncio
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
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1").strip()
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts").strip()
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy").strip()

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
            "Sé natural, conversacional y muy útil. Explica con claridad, "
            "da pasos concretos cuando sea necesario y pide precisión si falta contexto. "
            "Mantén un tono amable y profesional."
        )
    if lang == "en":
        return (
            "You are SNGSLUISGUZMAN AI. Reply ONLY in English. "
            "Be natural, conversational, and highly helpful. Explain clearly, "
            "offer concrete steps when useful, and ask for clarification if context is missing. "
            "Keep a friendly, professional tone."
        )
    return (
        "Tu es SNGSLUISGUZMAN AI. Réponds UNIQUEMENT en français. "
        "Sois naturel, conversationnel et très utile. Explique clairement, "
        "donne des étapes concrètes si nécessaire et demande des précisions si le contexte manque. "
        "Garde un ton aimable et professionnel."
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


def openai_vision_once(prompt: str, image_data_url: str, lang: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY manquante")

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt_for(lang)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        temperature=0.7,
    )
    return (resp.choices[0].message.content or "").strip()


def openai_image_once(prompt: str, size: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY manquante")

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    resp = client.images.generate(
        model=OPENAI_IMAGE_MODEL,
        prompt=prompt,
        size=size,
        response_format="b64_json",
    )
    data = resp.data[0]
    b64_json = getattr(data, "b64_json", None) or data.get("b64_json")
    if not b64_json:
        raise RuntimeError("Image non disponible")
    return f"data:image/png;base64,{b64_json}"


def openai_tts_once(text: str, voice: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY manquante")

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    resp = client.audio.speech.create(
        model=OPENAI_TTS_MODEL,
        voice=voice or OPENAI_TTS_VOICE,
        input=text,
        response_format="mp3",
    )
    audio_bytes = None
    if hasattr(resp, "read"):
        audio_bytes = resp.read()
    elif hasattr(resp, "content"):
        audio_bytes = resp.content
    else:
        audio_bytes = bytes(resp)

    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:audio/mpeg;base64,{b64_audio}"


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

    return "⚠️ Aucune IA disponible."


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


def is_image_request(text: str) -> bool:
    t = (text or "").lower().strip()
    keywords = [
        "génère une image", "genere une image", "génère moi une image", "genere moi une image",
        "crée une image", "cree une image", "fais une image", "faire une image",
        "génère un visuel", "genere un visuel", "crée un visuel", "cree un visuel",
        "create an image", "generate an image", "make an image", "image generation",
        "crear una imagen", "genera una imagen", "generar una imagen", "haz una imagen",
    ]
    return any(k in t for k in keywords)


def normalize_image_prompt(text: str) -> str:
    t = (text or "").strip()
    lowers = t.lower()
    prefixes = [
        "génère une image", "genere une image", "génère moi une image", "genere moi une image",
        "crée une image", "cree une image", "fais une image", "faire une image",
        "génère un visuel", "genere un visuel", "crée un visuel", "cree un visuel",
        "create an image", "generate an image", "make an image", "image generation",
        "crear una imagen", "genera una imagen", "generar una imagen", "haz una imagen",
    ]
    for prefix in prefixes:
        if lowers.startswith(prefix):
            return t[len(prefix):].strip(" :,-")
    return t


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

    if is_image_request(message):
        prompt = normalize_image_prompt(message)
        if not prompt:
            return JSONResponse({"ok": False, "reply": "", "error": "prompt_vide"}, status_code=400)
        try:
            image_url = openai_image_once(prompt, "1024x1024")
        except Exception as exc:
            return JSONResponse({"ok": False, "reply": "", "error": str(exc)}, status_code=400)
        return {"ok": True, "image_url": image_url, "prompt": prompt, "error": None}

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
        if is_image_request(message):
            prompt = normalize_image_prompt(message)
            if not prompt:
                yield f"data: {json.dumps({'type':'error','data':'prompt_vide'}, ensure_ascii=False)}\n\n"
                return
            try:
                image_url = openai_image_once(prompt, "1024x1024")
            except Exception as exc:
                yield f"data: {json.dumps({'type':'error','data':str(exc)}, ensure_ascii=False)}\n\n"
                return
            payload = {"type": "image", "data": {"url": image_url, "prompt": prompt}}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            return
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
# IMAGE GENERATION
# ─────────────────────────────────────────────────────────────
@app.post("/api/image")
async def generate_image(payload: Dict[str, Any] = Body(...)):
    prompt = (payload.get("prompt") or "").strip()
    size = (payload.get("size") or "1024x1024").strip()
    if not prompt:
        return JSONResponse({"ok": False, "error": "prompt_vide"}, status_code=400)

    valid_sizes = {"256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"}
    if size not in valid_sizes:
        return JSONResponse({"ok": False, "error": "size_invalide"}, status_code=400)

    try:
        image_url = openai_image_once(prompt, size)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    return {"ok": True, "image_url": image_url, "error": None}


@app.post("/api/image_analyze")
async def analyze_image(payload: Dict[str, Any] = Body(...)):
    image_data_url = (payload.get("image_data_url") or "").strip()
    prompt = (payload.get("prompt") or "").strip()
    if not image_data_url:
        return JSONResponse({"ok": False, "error": "image_vide"}, status_code=400)
    if not prompt:
        prompt = "Décris cette image et réponds clairement."

    lang = detect_language(prompt)
    try:
        reply = openai_vision_once(prompt, image_data_url, lang)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    return {"ok": True, "reply": reply, "error": None}


@app.post("/api/tts")
async def tts(payload: Dict[str, Any] = Body(...)):
    text = (payload.get("text") or "").strip()
    voice = (payload.get("voice") or OPENAI_TTS_VOICE).strip()
    if not text:
        return JSONResponse({"ok": False, "error": "text_vide"}, status_code=400)
    if len(text) > 4000:
        return JSONResponse({"ok": False, "error": "text_trop_long"}, status_code=400)

    try:
        audio_url = openai_tts_once(text, voice)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    return {"ok": True, "audio_url": audio_url, "error": None}


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
