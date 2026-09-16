"""Echocardiography technician: DICOM/video dan 11 ta standart ko‘rinishni tasniflash.

Kadrlar pydicom/opencv orqali olinadi; yorliq DICOM tegi yoki geometrik model.
Tashxis qo‘yilmaydi.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tools.echo_view_model import ECHO_KORINISHLAR, kadrlar_tasnif, tegdan_korinish
from tools.echo_yuklash import echo_fayldan_kadrlar, echo_yollardan_oqish


def _fayl_yol(bemor: Dict[str, Any]) -> Optional[Path]:
    """Bemor yozuvidan echo fayl yo‘lini oladi.

    Args:
        bemor: echo_fayl, dicom, echo_path maydonlari.

    Returns:
        Path yoki None.
    """
    for kalit in ("echo_fayl", "dicom", "echo_path", "echo_video"):
        qiymat = bemor.get(kalit)
        if qiymat and not isinstance(qiymat, (bytes, bytearray)):
            return Path(str(qiymat))
    return None


def _yuklamalarni_yig(bemor: Dict[str, Any]) -> List[Tuple[str, bytes]]:
    """UI/agentdan kelgan echo baytlarini (nom, bayt) ro‘yxatiga yig‘adi.

    Args:
        bemor: echo_bayt, echo_fayllar, yo‘l.

    Returns:
        Yuklangan juftliklar. DICOM/video tahlili uchun.
    """
    juft: List[Tuple[str, bytes]] = []
    roy = bemor.get("echo_fayllar")
    if isinstance(roy, (list, tuple)):
        for element in roy:
            if isinstance(element, dict) and element.get("bayt"):
                juft.append((str(element.get("nom") or "echo.dcm"), bytes(element["bayt"])))
            elif isinstance(element, (tuple, list)) and len(element) >= 2:
                juft.append((str(element[0]), bytes(element[1])))
    if bemor.get("echo_bayt"):
        juft.append((str(bemor.get("echo_fayl_nomi") or "echo.dcm"), bytes(bemor["echo_bayt"])))
    yol = _fayl_yol(bemor)
    if yol is not None and yol.exists() and yol.is_file():
        juft.append((yol.name, yol.read_bytes()))
    # Takrorlarni olib tashlash (nom+hajm)
    unikal: List[Tuple[str, bytes]] = []
    korilgan = set()
    for nom, bayt in juft:
        kalit = (nom, len(bayt))
        if kalit in korilgan:
            continue
        korilgan.add(kalit)
        unikal.append((nom, bayt))
    return unikal


def _bir_yozuvni_tasnif(nom: str, kadrlar: Sequence[Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Bitta DICOM/video yozuvini 11 ko‘rinishdan biriga bog‘laydi.

    Args:
        nom: Fayl nomi (faqat xabar).
        kadrlar: Kulrang kadrlar.
        meta: DICOM teglari.

    Returns:
        asosiy, ehtimol, manba, korinishlar. Klinik tasnif emas.
    """
    teg = tegdan_korinish(meta)
    eht = kadrlar_tasnif(list(kadrlar)) if kadrlar else {n: 0.0 for n in ECHO_KORINISHLAR}
    model_yorliq = max(eht, key=eht.get) if eht else "ANIQLANMAGAN"
    model_p = float(eht.get(model_yorliq, 0.0))

    if teg:
        manba = "dicom_teg"
        asosiy = teg
        # Teg va geometriya mos kelsa ishonch oshadi
        if teg == model_yorliq:
            manba = "dicom_teg+geometrik_model"
            p = max(model_p, 0.75)
        else:
            p = 0.62
            eht = dict(eht)
            eht[teg] = max(eht.get(teg, 0.0), 0.62)
    elif model_p >= 0.18:
        manba = "geometrik_model"
        asosiy = model_yorliq
        p = model_p
    else:
        manba = "aniqlanmadi"
        asosiy = "ANIQLANMAGAN"
        p = model_p

    tartib = sorted(eht.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return {
        "fayl": nom,
        "asosiy": asosiy,
        "ehtimol": round(float(p), 3),
        "manba": manba,
        "yuqori3": [{"nom": n, "ehtimol": round(float(v), 3)} for n, v in tartib],
        "kadrlar_soni": len(kadrlar),
        "meta": {k: meta.get(k) for k in ("series_description", "protocol_name", "view_name", "modality")},
    }


def classify_echo_views(bemor: Dict[str, Any]) -> Dict[str, Any]:
    """Echo DICOM/video/rasmdan 11 standart ko‘rinish yorlig‘ini beradi.

    Args:
        bemor: echo_bayt / echo_fayllar / echo_fayl yo‘li.

    Returns:
        ok, korinishlar, yozuvlar, xabar, model. Tashxis emas.
    """
    yuklama = _yuklamalarni_yig(bemor)
    if not yuklama:
        return {
            "ok": False,
            "xabar": "Echokardiogramma fayli yo‘q — tasnif o‘tkazib yuborildi.",
            "korinishlar": [],
            "yozuvlar": [],
            "model": None,
        }

    oqish = echo_yollardan_oqish(yuklama)
    if not oqish.get("ok"):
        return {
            "ok": False,
            "xabar": oqish.get("xabar") or "Echo o‘qilmadi.",
            "korinishlar": [],
            "yozuvlar": [],
            "model": None,
        }

    natijalar: List[Dict[str, Any]] = []
    preview: List[Any] = []
    for yoz in oqish.get("yozuvlar") or []:
        tasnif = _bir_yozuvni_tasnif(yoz.get("nom") or "", yoz.get("kadrlar") or [], yoz.get("meta") or {})
        natijalar.append(tasnif)
        kadrlar = yoz.get("kadrlar") or []
        if kadrlar:
            preview.append(kadrlar[0])

    asosiy_roy = [n["asosiy"] for n in natijalar if n.get("asosiy") and n["asosiy"] != "ANIQLANMAGAN"]
    unikal = []
    for n in asosiy_roy:
        if n not in unikal:
            unikal.append(n)

    manbalar: List[str] = []
    for n in natijalar:
        for m in str(n.get("manba") or "").split("+"):
            if m and m not in manbalar:
                manbalar.append(m)
    xabar = (
        f"Echo technician: {len(natijalar)} yozuv, ko‘rinishlar={unikal or ['ANIQLANMAGAN']}. "
        f"Manba={'+'.join(manbalar)}. 11 ta standart yorliq taxminiy, tashxis emas."
    )
    return {
        "ok": True,
        "xabar": xabar,
        "korinishlar": unikal or ["ANIQLANMAGAN"],
        "yozuvlar": natijalar,
        "kadrlar_soni": sum(n.get("kadrlar_soni") or 0 for n in natijalar),
        "model": "+".join(manbalar),
        "preview_kadrlar": preview[:4],
        "yetishmagan_standart": [k for k in ECHO_KORINISHLAR if k not in unikal],
    }


def echo_bormi(bemor: Dict[str, Any]) -> bool:
    """Echo fayl/bayt/kadr bor-yo‘qligini tekshiradi.

    Args:
        bemor: Agent holati.

    Returns:
        True — classify_echo_views chaqirish ma’noli.
    """
    if bemor.get("echo_bayt") or bemor.get("echo_fayllar") or bemor.get("echo_kadrlar"):
        return True
    yol = _fayl_yol(bemor)
    return yol is not None
