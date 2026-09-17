"""DeepSeek-R1 ga OpenAI-mos HTTP orqali murojaat.

Kalit bo‘lmasa, chaqiriq None qaytaradi — agentlar qoida/shablon ishlatadi.
Klinik tashxis o‘rnini bosmaydi.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv

    # Kalit venv ichida emas — loyiha ildizidagi .env (gitga kirmaydi)
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass


def _sozlama() -> Dict[str, str]:
    """Muhit o‘zgaruvchilaridan API manzili va model nomini o‘qiydi.

    Returns:
        base_url, api_key, model. Kalit bo‘sh bo‘lishi mumkin.
    """
    kalit = (os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    baza = (
        os.getenv("DEEPSEEK_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.deepseek.com"
    ).rstrip("/")
    model = (
        os.getenv("DEEPSEEK_MODEL")
        or os.getenv("LLM_MODEL")
        or "deepseek-reasoner"
    )
    vision = (os.getenv("FELLOW_VISION_MODEL") or os.getenv("DEEPSEEK_VISION_MODEL") or "").strip()
    return {"base_url": baza, "api_key": kalit, "model": model, "vision_model": vision}


def vlm_sozlama(rol: str) -> Optional[Dict[str, str]]:
    """MDT rollari (MedGemma / Qwen2.5-VL) uchun OpenAI-mos ulanish.

    Args:
        rol: ``medgemma`` yoki ``qwen_vl``.

    Returns:
        base_url, api_key, model. Yo‘q bo‘lsa None — shablon ishlatiladi.
    """
    asos = _sozlama()
    if rol == "medgemma":
        maxsus_baza = (os.getenv("MEDGEMMA_BASE_URL") or "").strip().rstrip("/")
        maxsus_kalit = (os.getenv("MEDGEMMA_API_KEY") or "").strip()
        maxsus_model = (os.getenv("MEDGEMMA_MODEL") or "").strip()
    elif rol == "qwen_vl":
        maxsus_baza = (os.getenv("QWEN_VL_BASE_URL") or "").strip().rstrip("/")
        maxsus_kalit = (os.getenv("QWEN_VL_API_KEY") or "").strip()
        maxsus_model = (os.getenv("QWEN_VL_MODEL") or "").strip()
    else:
        return None
    model = maxsus_model or asos.get("vision_model") or ""
    if maxsus_baza:
        return {
            "base_url": maxsus_baza,
            "api_key": maxsus_kalit or asos["api_key"] or "local",
            "model": model or rol,
        }
    # Faqat maxsus VLM modeli yoki FELLOW_VISION_MODEL — DeepSeek-reasoner tasvirni olmaydi
    if model and asos["api_key"]:
        past = model.lower()
        if "reasoner" in past:
            return None
        return {"base_url": asos["base_url"], "api_key": asos["api_key"], "model": model}
    return None


def llm_mavjud() -> bool:
    """API kaliti bor-yo‘qligini bildiradi.

    Returns:
        True — chaqiriq uriniladi; False — zaxira qoidalar.
    """
    return bool(_sozlama()["api_key"])


def _host_qisqa(url: str) -> str:
    """Base URL dan hostni oladi (kalit/so‘rov parametri yo‘q).

    Args:
        url: OpenAI-mos baza.

    Returns:
        host yoki bo‘sh satr.
    """
    if not url:
        return ""
    qism = url.split("://", 1)[-1]
    return qism.split("/", 1)[0].split("@")[-1]


def model_qisqacha() -> Dict[str, Any]:
    """Audit va UI uchun model nomlari (API kalitsiz).

    Returns:
        chat/fellow/VLM nomlari va ulanish holati. Tashxis emas.
    """
    soz = _sozlama()
    ulangan = bool(soz["api_key"])
    med = vlm_sozlama("medgemma")
    qwen = vlm_sozlama("qwen_vl")
    chat = soz["model"] if ulangan else "shablon"
    vision = (soz.get("vision_model") or soz["model"]) if ulangan else "shablon"
    return {
        "llm_ulangan": ulangan,
        "chat": chat,
        "fellow_vision": vision,
        "medgemma": (med or {}).get("model") if med else "shablon",
        "qwen_vl": (qwen or {}).get("model") if qwen else "shablon",
        "chat_host": _host_qisqa(soz["base_url"]) if ulangan else "",
    }


def llm_chat(
    xabarlar: List[Dict[str, Any]],
    temperatura: float = 0.2,
    max_token: int = 1200,
    model: Optional[str] = None,
    baza_url: Optional[str] = None,
    api_kalit: Optional[str] = None,
    timeout: float = 90.0,
) -> Optional[str]:
    """Chat completion yuboradi; xatoda None (agent to‘xtamaydi).

    Args:
        xabarlar: role/content (content matn yoki multimodal qismlar ro‘yxati).
        temperatura: Generatsiya tasodifiyligi.
        max_token: Javob uzunligi chegarasi.
        model: Bo‘sh bo‘lsa DEEPSEEK_MODEL; tasvir uchun vision model.
        baza_url: Ixtiyoriy OpenAI-mos server (MedGemma/Qwen vLLM).
        api_kalit: Ixtiyoriy kalit; mahalliy vLLM da ``local``.
        timeout: HTTP kutish (soniya); VLM uchun uzaytiriladi.

    Returns:
        Model matni yoki None. Tashxis sifatida ishlatilmasin.
    """
    soz = _sozlama()
    kalit = soz["api_key"] if api_kalit is None else (api_kalit or "").strip()
    baza = (baza_url or soz["base_url"] or "").rstrip("/")
    if not kalit or not baza:
        return None
    tanasi = {
        "model": model or soz["model"],
        "messages": xabarlar,
        "temperature": temperatura,
        "max_tokens": max_token,
    }
    talab = urllib.request.Request(
        f"{baza}/v1/chat/completions",
        data=json.dumps(tanasi).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {kalit}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(talab, timeout=timeout) as javob:
            yuk = json.loads(javob.read().decode("utf-8"))
        tanlovlar = yuk.get("choices") or []
        if not tanlovlar:
            return None
        return (tanlovlar[0].get("message") or {}).get("content")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError):
        return None


def llm_vision_model() -> Optional[str]:
    """Fellow multimodal chaqiriq uchun model nomini beradi.

    Returns:
        Vision model yoki asosiy chat model (kalit bo‘lsa). Yo‘q bo‘lsa None.
    """
    soz = _sozlama()
    if not soz["api_key"]:
        return None
    return soz.get("vision_model") or soz["model"]


def llm_json(
    tizim: str,
    foydalanuvchi: str,
    zaxira: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Modeldan JSON kutadi; bo‘lmasa zaxira lug‘atini qaytaradi.

    Args:
        tizim: System prompt (rol va JSON sxema).
        foydalanuvchi: Klinik kontekst.
        zaxira: API yo‘q yoki parse xatosida.

    Returns:
        Lug‘at. Tashxis o‘rnini bosmaydi.
    """
    matn = llm_chat(
        [
            {"role": "system", "content": tizim},
            {"role": "user", "content": foydalanuvchi},
        ]
    )
    if not matn:
        return dict(zaxira or {})
    tozalangan = matn.strip()
    if tozalangan.startswith("```"):
        qatorlar = tozalangan.split("\n")
        tozalangan = "\n".join(qatorlar[1:-1] if qatorlar[-1].startswith("```") else qatorlar[1:])
    bosh = tozalangan.find("{")
    oxir = tozalangan.rfind("}")
    if bosh < 0 or oxir <= bosh:
        return dict(zaxira or {})
    try:
        obyekt = json.loads(tozalangan[bosh : oxir + 1])
        return obyekt if isinstance(obyekt, dict) else dict(zaxira or {})
    except json.JSONDecodeError:
        return dict(zaxira or {})
