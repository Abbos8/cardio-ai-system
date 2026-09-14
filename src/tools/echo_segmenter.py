"""Echocardiography segmenter: chap qorincha (LV) konturini maskalash interfeysi.

To‘liq piksel-ma-piksel model ulanmaguncha xavfsiz rad javobi qaytariladi.
Tashxis o‘rnini bosmaydi.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def segment_lv(kadrlar: Optional[Any] = None, bemor: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Chap qorincha maskasini hisoblaydi yoki model yo‘qligini bildiradi.

    Args:
        kadrlar: Video/kadr massivi (T, H, W) yoki None.
        bemor: echo_kadrlar yoki echo_fayl maydonlari.

    Returns:
        ok, maska (yoki None), xabar. Klinik o‘lchov emas.
    """
    bemor = bemor or {}
    if kadrlar is None:
        kadrlar = bemor.get("echo_kadrlar")
    if kadrlar is None:
        return {
            "ok": False,
            "xabar": "LV segmentatsiya uchun kadrlar yo‘q. Model hali ulanmagan.",
            "maska": None,
            "model": None,
        }
    try:
        massiv = np.asarray(kadrlar)
    except Exception as exc:
        return {
            "ok": False,
            "xabar": f"Kadrlar o‘qilmadi: {exc}",
            "maska": None,
            "model": None,
        }
    return {
        "ok": False,
        "xabar": (
            f"Kadr shakli {tuple(massiv.shape)} qabul qilindi, lekin LV segmentatsiya "
            "og‘irliklari ulanmagan — yolg‘on maska yaratilmaydi."
        ),
        "maska": None,
        "shakl": list(massiv.shape),
        "model": None,
    }
