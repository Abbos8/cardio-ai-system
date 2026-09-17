"""Har tahlil uchun qisqa audit: qaysi model, qaysi CardiacRAG C.

Bemorga oid identifikator, EKG signal va API kalit yozilmaydi.
Fayl `data/audit/` da (git emas). Klinik tashxis emas.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm.client import model_qisqacha
from xavfsizlik.himoya import matndan_sirni_kes

KLINIK_OGOHLANTIRISH = (
    "Bu tizim shifokor o‘rnini bosmaydi. Tashxis va davolash faqat shifokorga tegishli. "
    "Demo: klinik production emas."
)
KLINIK_PRODUCTION = False

_C_PREFIKS = re.compile(r"^\[([^\]]+)\]")
_ILDIG = Path(__file__).resolve().parents[2]
ODATIY_AUDIT_DIR = _ILDIG / "data" / "audit"


def c_manbalar(dalillar: Optional[List[str]]) -> List[str]:
    """CardiacRAG C bo‘laklaridan manba identifikatorini oladi.

    Args:
        dalillar: retrieve() matnlari (`[jild/stem] ...` yoki seed).

    Returns:
        Takrorsiz manba id lari. To‘liq chunk matni emas.
    """
    ids: List[str] = []
    for bolak in dalillar or []:
        satr = str(bolak).strip()
        mos = _C_PREFIKS.match(satr)
        nom = mos.group(1) if mos else "seed/cardiology"
        if nom not in ids:
            ids.append(nom)
    return ids


def audit_katalog() -> Path:
    """Audit JSON yoziladigan jild (gitignore).

    Returns:
        AUDIT_DIR yoki data/audit.
    """
    maxsus = (os.getenv("AUDIT_DIR") or "").strip()
    return Path(maxsus) if maxsus else ODATIY_AUDIT_DIR


def audit_yozilsinmi() -> bool:
    """Muhitda diskka yozish yoqilganmi.

    Returns:
        Default True; AUDIT_YOZ=0 da False.
    """
    qiymat = (os.getenv("AUDIT_YOZ") or "1").strip().lower()
    return qiymat not in {"0", "false", "yoq", "off"}


def audit_yig(holat: Dict[str, Any]) -> Dict[str, Any]:
    """Graf holatidan PII-siz audit yozuvini yig‘adi.

    Args:
        holat: ChiefCardiologist tugagan (yoki oraliq) holat.

    Returns:
        Model, C manbalari, qadamlar. Signal/kalit yo‘q.
    """
    fellow = holat.get("fellow_natija") or {}
    mdt = holat.get("mdt_natija") or {}
    echo = holat.get("echo_natija") or {}
    model = model_qisqacha()
    c_ids = c_manbalar(holat.get("rag_dalillar") or [])
    qadamlar = [matndan_sirni_kes(str(q)) for q in (holat.get("qadam_tarixi") or [])]
    return {
        "id": str(uuid.uuid4()),
        "vaqt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "klinik_production": KLINIK_PRODUCTION,
        "ogohlantirish": KLINIK_OGOHLANTIRISH,
        "modellar": model,
        "C": c_ids,
        "C_soni": len(holat.get("rag_dalillar") or []),
        "fellow_manba": fellow.get("manba") or None,
        "fellow_model": fellow.get("model") or (
            "shablon" if fellow.get("manba") == "shablon" else model.get("fellow_vision")
        ),
        "mdt_medgemma_manba": mdt.get("medgemma_manba") or None,
        "mdt_qwen_manba": mdt.get("qwen_manba") or None,
        "mdt_medgemma_model": mdt.get("medgemma_model") or model.get("medgemma"),
        "mdt_qwen_model": mdt.get("qwen_model") or model.get("qwen_vl"),
        "echo_manba": echo.get("model") or None,
        "murakkablik": holat.get("murakkablik"),
        "amal": holat.get("amal"),
        "qadamlar": qadamlar,
    }


def audit_saqla(yozuv: Dict[str, Any], katalog: Optional[Path] = None) -> Optional[Path]:
    """Audit JSON ni mahalliy diskka yozadi (git emas).

    Args:
        yozuv: audit_yig natijasi.
        katalog: Ixtiyoriy jild.

    Returns:
        Yozilgan fayl yo‘li yoki None (o‘chirilgan / xato).
    """
    if not audit_yozilsinmi():
        return None
    joy = Path(katalog) if katalog is not None else audit_katalog()
    try:
        joy.mkdir(parents=True, exist_ok=True)
        nom = f"{yozuv.get('vaqt', 'x')}_{yozuv.get('id', 'n')[:8]}.json".replace(":", "")
        yol = joy / nom
        yol.write_text(json.dumps(yozuv, ensure_ascii=False, indent=2), encoding="utf-8")
        return yol
    except OSError:
        return None
