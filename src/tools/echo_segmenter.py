"""Echocardiography segmenter: chap qorincha (LV) kavagini piksel maska qiladi.

Kavak kadrning qorong‘i bo‘shlig‘idan olinadi (OpenCV). Klinik U-Net og‘irligi emas.
EF hisoblanmaydi; tashxis o‘rnini bosmaydi. Kadr yo‘q bo‘lsa maska yaratilmaydi.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from tools.echo_tool import _yuklamalarni_yig
from tools.echo_yuklash import _kulrang, echo_yollardan_oqish

APICAL = {"A4C", "A2C", "A3C"}
SAX = {"PSAX-AV", "PSAX-MV", "PSAX-PM", "PSAX-APEX"}


def overlay_qur(kadr: np.ndarray, maska: np.ndarray) -> np.ndarray:
    """Kulrang kadr ustiga cyan maska va sariq kontur chizadi.

    Args:
        kadr: uint8 (H, W).
        maska: 0/1 yoki 0/255.

    Returns:
        RGB uint8 overlay. Ko‘rish uchun, o‘lchov emas.
    """
    g = _kulrang(kadr)
    m = (np.asarray(maska) > 0).astype(np.uint8)
    if m.shape != g.shape:
        if cv2 is not None:
            m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            m = np.zeros_like(g, dtype=np.uint8)
    rgb = np.stack([g, g, g], axis=-1).astype(np.float32)
    cyan = np.array([40.0, 200.0, 220.0])
    rgb[m > 0] = rgb[m > 0] * 0.45 + cyan * 0.55
    if cv2 is not None:
        kontur, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rgb_u = np.clip(rgb, 0, 255).astype(np.uint8)
        cv2.drawContours(rgb_u, kontur, -1, (255, 220, 40), 1)
        return rgb_u
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _png_bayt(rgb: np.ndarray) -> Optional[bytes]:
    """RGB kadrni PNG baytiga siqadi (UI/graf uchun).

    Args:
        rgb: (H, W, 3) uint8.

    Returns:
        PNG yoki None.
    """
    if cv2 is None:
        return None
    bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return None
    return bytes(buf)


def _fan_maska(g: np.ndarray) -> np.ndarray:
    """Ultratovush sektori taxminiy maskasi.

    Args:
        g: Kulrang kadr.

    Returns:
        bool (H, W). Sektor tashqarisi olinmaydi.
    """
    h, w = g.shape
    yy, xx = np.indices((h, w))
    fan = np.abs(xx - w / 2.0) < (0.10 * w + yy * 0.62)
    if cv2 is not None:
        yorug = (g > np.percentile(g, 18)).astype(np.uint8)
        yorug = cv2.dilate(yorug, np.ones((7, 7), np.uint8), iterations=1)
        return fan & (yorug > 0)
    return fan & (g > 12)


def _kavak_maska(g: np.ndarray, korinish: Optional[str]) -> Tuple[Optional[np.ndarray], str]:
    """Qorong‘i kamerani (taxminiy LV) ajratadi.

    Args:
        g: Kulrang echo kadr.
        korinish: A4C/A2C/... yoki None.

    Returns:
        (maska uint8 0/1, izoh). Topilmasa (None, sabab).
    """
    if cv2 is None:
        return None, "opencv yo‘q"
    h, w = g.shape
    blur = cv2.GaussianBlur(g, (5, 5), 0)
    fan = _fan_maska(blur).astype(np.uint8)
    if int(fan.sum()) < 80:
        return None, "ultratovush sektori topilmadi"

    # Bo‘shliqlar qorong‘i
    fan_pix = blur[fan > 0]
    chegara = int(np.percentile(fan_pix, 32))
    qorongi = ((blur < chegara) & (fan > 0)).astype(np.uint8)
    yadro = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    qorongi = cv2.morphologyEx(qorongi, cv2.MORPH_OPEN, yadro, iterations=1)
    qorongi = cv2.morphologyEx(qorongi, cv2.MORPH_CLOSE, yadro, iterations=2)

    n, yorliq, stats, centroids = cv2.connectedComponentsWithStats(qorongi, connectivity=8)
    fan_may = float(fan.sum())
    nomzod: List[Tuple[float, int]] = []
    for i in range(1, n):
        may = float(stats[i, cv2.CC_STAT_AREA])
        ulush = may / fan_may
        if ulush < 0.012 or ulush > 0.42:
            continue
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        ww, hh = float(stats[i, cv2.CC_STAT_WIDTH]), float(stats[i, cv2.CC_STAT_HEIGHT])
        dumaloq = may / (max(np.pi * (ww / 2) * (hh / 2), 1.0))
        k = (korinish or "").upper()
        ball = ulush
        if k in APICAL:
            ball += 0.35 * (cy / h) + 0.25 * (cx / w)
        elif k in SAX:
            ball += 0.45 * min(dumaloq, 1.2)
        else:
            ball += 0.2 * (cy / h)
        nomzod.append((ball, i))

    if not nomzod:
        return None, "yetarli kattalikdagi qorong‘i kavak yo‘q"

    _, eng = max(nomzod, key=lambda t: t[0])
    maska = (yorliq == eng).astype(np.uint8)
    # Teshiklarni to‘ldirish
    kontur, _ = cv2.findContours(maska, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    toldir = np.zeros_like(maska)
    cv2.drawContours(toldir, kontur, -1, 1, thickness=cv2.FILLED)
    return toldir, "kavak ajratildi"


def _kadrlar_yig(
    bemor: Dict[str, Any],
    echo_natija: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Bemor fayllaridan kadr + ko‘rinish yorlig‘ini yig‘adi.

    Args:
        bemor: echo_fayllar/bayt/yo‘l/echo_kadrlar.
        echo_natija: Technician tasnifi (ixtiyoriy).

    Returns:
        [{kadr, korinish, nom}].
    """
    chiq: List[Dict[str, Any]] = []
    yozuvlar = {z.get("fayl"): z for z in (echo_natija or {}).get("yozuvlar") or []}
    yuk = _yuklamalarni_yig(bemor)
    if yuk:
        oqish = echo_yollardan_oqish(yuk)
        for yoz in oqish.get("yozuvlar") or []:
            nom = yoz.get("nom") or ""
            kor = (yozuvlar.get(nom) or {}).get("asosiy")
            for kadr in (yoz.get("kadrlar") or [])[:3]:
                chiq.append({"kadr": _kulrang(kadr), "korinish": kor, "nom": nom})
    if not chiq and bemor.get("echo_kadrlar") is not None:
        try:
            massiv = np.asarray(bemor.get("echo_kadrlar"))
            if massiv.ndim == 2:
                chiq.append({"kadr": _kulrang(massiv), "korinish": None, "nom": "echo_kadrlar"})
            elif massiv.ndim >= 3:
                for i in range(min(massiv.shape[0], 4)):
                    chiq.append({"kadr": _kulrang(massiv[i]), "korinish": None, "nom": f"kadr{i}"})
        except Exception:
            pass
    return chiq


def segment_lv(
    kadrlar: Optional[Any] = None,
    bemor: Optional[Dict[str, Any]] = None,
    echo_natija: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Chap qorincha taxminiy maskasini kadr(lar)dan hisoblaydi.

    Args:
        kadrlar: (T,H,W) yoki (H,W); berilsa bemor faylidan oldin ishlatiladi.
        bemor: echo_fayllar, echo_bayt, echo_kadrlar.
        echo_natija: 11 ko‘rinish yorlig‘i (A4C afzal).

    Returns:
        ok, overlay_png, maydon_px, ulush, xabar. Klinik EF/tashxis emas.
        Kadr yo‘q yoki kavak yo‘q — ok=False, maska yo‘q.
    """
    bemor = bemor or {}
    if echo_natija is None:
        echo_natija = bemor.get("echo_natija")
    if cv2 is None:
        return {
            "ok": False,
            "xabar": "opencv o‘rnatilmagan — LV maska hisoblanmadi.",
            "maska": None,
            "model": None,
        }

    royxat = _kadrlar_yig(bemor, echo_natija)
    if kadrlar is not None:
        try:
            massiv = np.asarray(kadrlar)
            if massiv.ndim == 2:
                royxat = [{"kadr": _kulrang(massiv), "korinish": None, "nom": "kadr"}] + royxat
            elif massiv.ndim >= 3:
                qosh = [
                    {"kadr": _kulrang(massiv[i]), "korinish": None, "nom": f"kadr{i}"}
                    for i in range(min(int(massiv.shape[0]), 4))
                ]
                royxat = qosh + royxat
        except Exception as exc:
            if not royxat:
                return {
                    "ok": False,
                    "xabar": f"Kadrlar o‘qilmadi: {exc}",
                    "maska": None,
                    "model": None,
                }

    if not royxat:
        return {
            "ok": False,
            "xabar": "LV segmentatsiya uchun kadrlar yo‘q.",
            "maska": None,
            "model": None,
        }

    def tartib(element: Dict[str, Any]) -> Tuple[int, int]:
        k = str(element.get("korinish") or "")
        if k == "A4C":
            return (0, 0)
        if k in APICAL:
            return (1, 0)
        if k in SAX:
            return (2, 0)
        return (3, 0)

    royxat.sort(key=tartib)
    oxirgi_sabab = "kavak topilmadi"
    for element in royxat:
        kadr = element["kadr"]
        kor = element.get("korinish")
        maska, sabab = _kavak_maska(kadr, kor)
        if maska is None:
            oxirgi_sabab = sabab
            continue
        maydon = int(maska.sum())
        ulush = round(maydon / float(maska.size), 4)
        overlay = overlay_qur(kadr, maska)
        png = _png_bayt(overlay)
        xabar = (
            f"LV kontur (algoritmik kavak, U-Net emas): ko‘rinish={kor or 'noma’lum'}, "
            f"fayl={element.get('nom')}, {maydon} px ({ulush:.1%} kadr). "
            "EF hisoblanmaydi, tashxis emas."
        )
        return {
            "ok": True,
            "xabar": xabar,
            "maska": maska,
            "overlay_png": png,
            "maydon_px": maydon,
            "ulush": ulush,
            "shakl": list(maska.shape),
            "korinish": kor,
            "fayl": element.get("nom"),
            "model": "opencv-kavak",
        }

    return {
        "ok": False,
        "xabar": (
            f"Kadr bor ({len(royxat)} ta), lekin LV kavagi ajratilmadi ({oxirgi_sabab}). "
            "Yolg‘on maska berilmaydi."
        ),
        "maska": None,
        "model": "opencv-kavak",
        "kadrlar_soni": len(royxat),
    }
