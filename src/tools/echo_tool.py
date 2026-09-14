"""Echocardiography technician: DICOM/video dan 11 ta standart ko‘rinishni tasniflash.

Hozirgi bosqich: fayl bor-yo‘qligi va nom bo‘yicha taxminiy yorliq.
To‘liq DICOM klassifikator ulanmaguncha tashxis qo‘yilmaydi.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

# Workflow dagi 11 ta standart echokardiografik ko‘rinish
ECHO_KORINISHLAR: List[str] = [
    "A2C",
    "A4C",
    "A3C",
    "PLAX",
    "PSAX-AV",
    "PSAX-MV",
    "PSAX-PM",
    "PSAX-APEX",
    "SUBCOSTAL-4C",
    "SUBCOSTAL-IVC",
    "SSN",
]


def _fayl_yol(bemor: Dict[str, Any]) -> Optional[Path]:
    """Bemor yozuvidan echo fayl yo‘lini oladi.

    Args:
        bemor: echo_fayl, dicom, echo_path maydonlari.

    Returns:
        Path yoki None.
    """
    for kalit in ("echo_fayl", "dicom", "echo_path", "echo_video"):
        qiymat = bemor.get(kalit)
        if qiymat:
            return Path(str(qiymat))
    return None


def classify_echo_views(bemor: Dict[str, Any]) -> Dict[str, Any]:
    """Echo faylidan 11 ko‘rinish yorlig‘ini taxmin qiladi (nom/heuristic).

    Args:
        bemor: DICOM yoki video yo‘li.

    Returns:
        ok, korinishlar, xabar. Klinik tasnif o‘rnini bosmaydi.
    """
    yol = _fayl_yol(bemor)
    if yol is None:
        return {
            "ok": False,
            "xabar": "Echokardiogramma fayli yo‘q — tasnif o‘tkazib yuborildi.",
            "korinishlar": [],
            "model": None,
        }
    if not yol.exists():
        return {
            "ok": False,
            "xabar": f"Echo fayli topilmadi: {yol}. DICOM klassifikator keyingi bosqichda ulanadi.",
            "korinishlar": [],
            "model": None,
        }

    nom = yol.name.upper()
    topilgan: List[str] = [k for k in ECHO_KORINISHLAR if k.replace("-", "") in nom.replace("-", "")]
    if not topilgan:
        topilgan = ["ANIQLANMAGAN"]
    return {
        "ok": True,
        "xabar": (
            "Fayl qabul qilindi; to‘liq 11-ko‘rinishli DICOM model hali ulanmagan. "
            "Yorliqlar fayl nomidan taxminiy. Tashxis emas."
        ),
        "korinishlar": topilgan,
        "model": "heuristic-filename",
        "fayl": str(yol),
    }
