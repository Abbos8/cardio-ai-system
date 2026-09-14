"""Lab technician: laboratoriya va dori tarixini matn + token ko‘rinishiga keltiradi.

Natija klinik tashxis o‘rnini bosmaydi.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Ko‘rsatkich → (birlik, qisqa klinik izoh — tashxis emas)
LAB_IZOH: Dict[str, Tuple[str, str]] = {
    "troponin_i": ("ng/L", "yurak shikastlanishi biomarkeri"),
    "troponin": ("ng/L", "yurak shikastlanishi biomarkeri"),
    "nt_probnp": ("pg/mL", "yurak yetishmovchiligi markeri"),
    "bnp": ("pg/mL", "natriuretik peptid"),
    "kreatinin": ("µmol/L", "buyrak funksiyasi"),
    "kaliy": ("mmol/L", "elektrolit, aritmiya xavfi bilan bog‘liq"),
    "natriy": ("mmol/L", "elektrolit"),
    "glyukoza": ("mmol/L", "qon shakari"),
    "ldl": ("mmol/L", "lipid profili"),
    "hdl": ("mmol/L", "lipid profili"),
    "hba1c": ("%", "uzoq muddatli glyukoza"),
    "gemoglobin": ("g/L", "qon"),
    "crp": ("mg/L", "yallig‘lanish"),
}


def _son(qiymat: Any) -> Optional[float]:
    """Laboratoriya qiymatini float ga o‘giradi.

    Args:
        qiymat: Forma yoki JSON dan kelgan son.

    Returns:
        Float yoki None.
    """
    if qiymat is None or qiymat == "":
        return None
    try:
        return float(qiymat)
    except (TypeError, ValueError):
        return None


def _tokenlar(matn: str) -> List[str]:
    """Matndan qisqa tibbiy tokenlar ajratadi.

    Args:
        matn: Lab yoki dori satri.

    Returns:
        Kichik harfli tokenlar. RAG/fellow so‘rovi uchun.
    """
    qismlar = re.findall(r"[A-Za-zА-Яа-яЁёўғҳқ0-9_\-]+", matn.lower())
    return [t for t in qismlar if len(t) > 1]


def process_lab(bemor: Dict[str, Any]) -> Dict[str, Any]:
    """Laboratoriya va dorilarni matn hamda token ro‘yxatiga aylantiradi.

    Args:
        bemor: laboratoriya lug‘ati, dorilar/meds, klinik_savol.

    Returns:
        ok, matn, tokenlar, qiymatlar. Tashxis qo‘yilmaydi.
    """
    lab = bemor.get("laboratoriya") or bemor.get("lab") or {}
    if not isinstance(lab, dict):
        lab = {}
    qatorlar: List[str] = []
    qiymatlar: Dict[str, float] = {}
    tokenlar: List[str] = ["laboratory"]

    for nom, xom in lab.items():
        son = _son(xom)
        if son is None:
            continue
        qiymatlar[str(nom)] = son
        birlik, izoh = LAB_IZOH.get(str(nom).lower(), ("", "laboratoriya ko‘rsatkichi"))
        qatorlar.append(f"{nom}={son} {birlik} ({izoh})".strip())
        tokenlar.extend(_tokenlar(str(nom)))

    dorilar = bemor.get("dorilar") or bemor.get("meds") or bemor.get("medication_history") or ""
    if isinstance(dorilar, list):
        dorilar = ", ".join(str(d) for d in dorilar)
    dori_matn = str(dorilar).strip()
    if dori_matn:
        qatorlar.append(f"Medication history: {dori_matn}")
        tokenlar.extend(_tokenlar(dori_matn))

    if not qatorlar:
        return {
            "ok": False,
            "xabar": "Laboratoriya qiymatlari kiritilmagan.",
            "matn": "",
            "tokenlar": [],
            "qiymatlar": {},
        }

    matn = "Lab technician: " + "; ".join(qatorlar)
    return {
        "ok": True,
        "xabar": "Laboratoriya va dori tarixi matn/token ko‘rinishiga keltirildi. Tashxis emas.",
        "matn": matn,
        "tokenlar": sorted(set(tokenlar)),
        "qiymatlar": qiymatlar,
    }
