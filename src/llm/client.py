"""DeepSeek-R1 ga OpenAI-mos HTTP orqali murojaat.

Kalit bo‘lmasa, chaqiriq None qaytaradi — agentlar qoida/shablon ishlatadi.
Klinik tashxis o‘rnini bosmaydi.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


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
    return {"base_url": baza, "api_key": kalit, "model": model}


def llm_mavjud() -> bool:
    """API kaliti bor-yo‘qligini bildiradi.

    Returns:
        True — chaqiriq uriniladi; False — zaxira qoidalar.
    """
    return bool(_sozlama()["api_key"])


def llm_chat(
    xabarlar: List[Dict[str, str]],
    temperatura: float = 0.2,
    max_token: int = 1200,
) -> Optional[str]:
    """Chat completion yuboradi; xatoda None (agent to‘xtamaydi).

    Args:
        xabarlar: role/content juftlari (system, user, assistant).
        temperatura: Generatsiya tasodifiyligi.
        max_token: Javob uzunligi chegarasi.

    Returns:
        Model matni yoki None. Tashxis sifatida ishlatilmasin.
    """
    soz = _sozlama()
    if not soz["api_key"]:
        return None
    tanasi = {
        "model": soz["model"],
        "messages": xabarlar,
        "temperature": temperatura,
        "max_tokens": max_token,
    }
    talab = urllib.request.Request(
        f"{soz['base_url']}/v1/chat/completions",
        data=json.dumps(tanasi).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {soz['api_key']}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(talab, timeout=90) as javob:
            yuk = json.loads(javob.read().decode("utf-8"))
        tanlovlar = yuk.get("choices") or []
        if not tanlovlar:
            return None
        return (tanlovlar[0].get("message") or {}).get("content")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError):
        return None


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
