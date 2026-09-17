"""Ko‘p tarmoqli munozara (MDT): MedGemma (tasvir) va Qwen2.5-VL (video).

Har raundda xom I va oraliq Z qayta kiritiladi (gallyutsinatsiya cheklovi).
VLM/API yo‘qida shablon. Konsensus yoki max_raund da to‘xtaydi. Tashxis emas.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llm.client import llm_chat, llm_json, model_qisqacha, vlm_sozlama
from tools.ecg_tool import ekg_qisqa_png

MAX_RAUND = 3
MAX_VIDEO_KADR = 6
VLM_TIMEOUT = 180.0


def _data_url(png: bytes) -> str:
    """PNG ni OpenAI-mos image_url data URI ga aylantiradi.

    Args:
        png: Rasm baytlari.

    Returns:
        data:image/png;base64,...
    """
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _kadr_png(kadr: Any, max_tomon: int = 384) -> Optional[bytes]:
    """Kulrang/RGB kaderni kichik PNG qiladi (VLM yukini cheklash).

    Args:
        kadr: 2D yoki 3D massiv.
        max_tomon: Eng uzun tomon (piksel).

    Returns:
        PNG yoki None.
    """
    try:
        import numpy as np

        try:
            import cv2
        except ImportError:
            return None
        arr = np.asarray(kadr)
        if arr.ndim == 2:
            rgb = np.stack([arr, arr, arr], axis=-1)
        else:
            rgb = arr[..., :3] if arr.shape[-1] >= 3 else arr
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        if max(h, w) > max_tomon and h > 0 and w > 0:
            scale = max_tomon / float(max(h, w))
            rgb = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))))
        ok, buf = cv2.imencode(".png", rgb)
        return bytes(buf) if ok else None
    except Exception:
        return None


def _kadr_tanla(kadrlar: Sequence[Any], max_n: int = MAX_VIDEO_KADR) -> List[Any]:
    """Cine/video kadrlarini teng oraliqda tanlaydi.

    Args:
        kadrlar: To‘liq ketma-ketlik.
        max_n: Yuboriladigan yuqori chegara.

    Returns:
        Tanlangan kadrlar.
    """
    if not kadrlar:
        return []
    n = len(kadrlar)
    if n <= max_n:
        return list(kadrlar)
    return [kadrlar[int(round(i * (n - 1) / (max_n - 1)))] for i in range(max_n)]


def _echo_yozuvlar(bemor: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Bemor echo fayllaridan kadr yozuvlarini o‘qiydi.

    Args:
        bemor: echo_fayllar / echo_bayt.

    Returns:
        [{nom, kadrlar, video}].
    """
    from tools.echo_yuklash import echo_yollardan_oqish

    juft = []
    roy = bemor.get("echo_fayllar") or []
    if isinstance(roy, (list, tuple)):
        for element in roy:
            if isinstance(element, dict) and element.get("bayt"):
                juft.append((str(element.get("nom") or "echo.dcm"), bytes(element["bayt"])))
    if bemor.get("echo_bayt"):
        juft.append((str(bemor.get("echo_fayl_nomi") or "echo.dcm"), bytes(bemor["echo_bayt"])))
    if not juft:
        return []
    natija: List[Dict[str, Any]] = []
    for yoz in (echo_yollardan_oqish(juft).get("yozuvlar") or []):
        nom = str(yoz.get("nom") or "")
        kadrlar = yoz.get("kadrlar") or []
        if not kadrlar:
            continue
        past = nom.lower()
        video = past.endswith((".mp4", ".avi", ".mov", ".mpg", ".mpeg", ".mkv")) or len(kadrlar) >= 2
        natija.append({"nom": nom, "kadrlar": kadrlar, "video": video})
    return natija


def mdt_vizual_yig(
    bemor: Optional[Dict[str, Any]] = None,
    ecg: Optional[Dict[str, Any]] = None,
    segment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """MedGemma uchun still rasmlar va Qwen uchun video kadrlar.

    Args:
        bemor: EKG signal va echo fayllar.
        ecg: EP o‘lchovlari (tasma).
        segment: LV overlay PNG.

    Returns:
        still [{id,png}], video [{id,png}], yoq_dalillar. Tashxis emas.
    """
    bemor = bemor or {}
    still: List[Dict[str, Any]] = []
    video: List[Dict[str, Any]] = []
    yoq: List[str] = []

    sig = None
    for kalit in ("ecg_signal", "ekg", "signal"):
        if bemor.get(kalit) is not None:
            sig = bemor[kalit]
            break
    if sig is not None:
        tasma = (ecg or {}).get("tasma") or "II"
        sr = float(bemor.get("sampling_rate") or 500.0)
        png = ekg_qisqa_png(sig, sampling_rate=sr, tasma=str(tasma))
        if png:
            still.append({"id": "ecg_grafik", "png": png})
    else:
        yoq.append("ecg_grafik")

    if segment and segment.get("overlay_png"):
        still.append({"id": "lv_overlay", "png": bytes(segment["overlay_png"])})
    else:
        yoq.append("lv_overlay")

    yozuvlar = _echo_yozuvlar(bemor)
    if not yozuvlar:
        yoq.append("echo")
        yoq.append("echo_video")
    else:
        cine_bor = False
        for i, yoz in enumerate(yozuvlar):
            kadrlar = yoz.get("kadrlar") or []
            bir = _kadr_png(kadrlar[0]) if kadrlar else None
            if bir:
                still.append({"id": f"echo_still_{i}", "png": bir})
            if yoz.get("video"):
                cine_bor = True
                for j, kadr in enumerate(_kadr_tanla(kadrlar)):
                    png = _kadr_png(kadr)
                    if png:
                        video.append({"id": f"echo_video_{i}_{j}", "png": png})
        if not cine_bor:
            yoq.append("echo_video")
            if still:
                # still kadrni Qwen ham ko‘rsin, lekin cine emasligini bilsin
                for r in still:
                    if str(r.get("id") or "").startswith("echo_still_"):
                        video.append({"id": r["id"] + "_qwen", "png": r["png"]})

    return {
        "still": still,
        "video": video,
        "yoq_dalillar": yoq,
        "still_id": [r["id"] for r in still],
        "video_id": [r["id"] for r in video],
    }


def _multimodal_qism(matn: str, rasmlar: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Matn + rasmlarni chat content ro‘yxatiga yig‘adi.

    Args:
        matn: I/Z va ko‘rsatma.
        rasmlar: [{png}].

    Returns:
        OpenAI-mos content qismlari.
    """
    qismlar: List[Dict[str, Any]] = [{"type": "text", "text": matn}]
    for r in rasmlar:
        png = r.get("png")
        if png:
            qismlar.append({"type": "image_url", "image_url": {"url": _data_url(bytes(png))}})
    return qismlar


def _foydalanuvchi_matn(
    xom_i: str,
    oraliq_z: str,
    oldingi: str,
    rasmlar_id: Sequence[str],
    yoq: Sequence[str],
    video: bool,
) -> str:
    """Har raundda I va Z ni qayta yozadi (o‘ylab topish taqiqlanadi).

    Args:
        xom_i: Xom kirish.
        oraliq_z: Vosita natijalari.
        oldingi: Hamkasb fikri.
        rasmlar_id: Yuborilgan rasm identifikatorlari.
        yoq: Yo‘q modalitetlar.
        video: Qwen cine/kadr ketma-ketligi.

    Returns:
        Prompt matni.
    """
    tur = "video/kadr ketma-ketligi" if video else "statik tasvirlar (EKG/echo/LV)"
    return (
        f"I (xom, qayta):\n{xom_i}\n\n"
        f"Z (oraliq, qayta):\n{oraliq_z}\n\n"
        f"Yuborilgan {tur}: {', '.join(rasmlar_id) or 'yo‘q'}.\n"
        f"Yo‘q dalillar (o‘ylab topmang): {', '.join(yoq) or '—'}.\n\n"
        f"Hamkasb oxirgi fikri:\n{oldingi or '(birinchi so‘z)'}\n\n"
        "Faqat I, Z va yuborilgan rasmlarga tayaning. Tashxis qo‘ymang. "
        "Yakuniy qaror shifokorga tegishli."
    )


def _shablon(rol: str, xom_i: str, oraliq_z: str, rasmlar_id: Sequence[str], yoq: Sequence[str]) -> str:
    """VLM yo‘qida I/Z ni qayta ko‘rsatadigan ehtiyotkor matn.

    Args:
        rol: MedGemma yoki Qwen tavsifi.
        xom_i, oraliq_z: Qayta kiritilgan kontekst.
        rasmlar_id, yoq: Vizual dalil holati.

    Returns:
        Shablon fikr. Tashxis emas.
    """
    return (
        f"{rol}: VLM/API ulanmagan, shablon rejimida. "
        f"I qayta: {xom_i[:400]}. Z qayta: {oraliq_z[:500]}. "
        f"Rasmlar: {', '.join(rasmlar_id) or 'yo‘q'}; yo‘q: {', '.join(yoq) or '—'}. "
        "Yetarli vizual model javobi yo‘q — konsensus ehtiyotkor. "
        "Tashxis emas; shifokor ko‘rigi shart."
    )


def _rol_javobi(
    rol_nom: str,
    tizim: str,
    xom_i: str,
    oraliq_z: str,
    oldingi: str,
    rasmlar: Sequence[Dict[str, Any]],
    yoq: Sequence[str],
    video: bool,
) -> Tuple[str, str]:
    """Bitta VLM/LLM rolida munozara matnini oladi.

    Args:
        rol_nom: medgemma yoki qwen_vl (ulanish).
        tizim: System prompt.
        xom_i, oraliq_z, oldingi: Kontekst.
        rasmlar: PNG lar.
        yoq: Yo‘q dalillar.
        video: True — Qwen cine.

    Returns:
        (matn, manba=vlm|llm|shablon).
    """
    ids = [str(r.get("id") or "") for r in rasmlar]
    user = _foydalanuvchi_matn(xom_i, oraliq_z, oldingi, ids, yoq, video)
    soz = vlm_sozlama(rol_nom)
    if soz and rasmlar:
        javob = llm_chat(
            [
                {"role": "system", "content": tizim},
                {"role": "user", "content": _multimodal_qism(user, rasmlar)},
            ],
            model=soz["model"],
            baza_url=soz["base_url"],
            api_kalit=soz["api_key"],
            timeout=VLM_TIMEOUT,
            max_token=900,
        )
        if javob:
            return javob.strip(), "vlm"
    if soz:
        javob = llm_chat(
            [
                {"role": "system", "content": tizim},
                {"role": "user", "content": user},
            ],
            model=soz["model"],
            baza_url=soz["base_url"],
            api_kalit=soz["api_key"],
            timeout=VLM_TIMEOUT,
            max_token=900,
        )
        if javob:
            return javob.strip(), "llm"
    rol_matn = "MedGemma (tasvir)" if rol_nom == "medgemma" else "Qwen2.5-VL (video)"
    return _shablon(rol_matn, xom_i, oraliq_z, ids, yoq), "shablon"


def _konsensus_qoida(med: str, qwen: str) -> bool:
    """API yo‘qida oddiy matn belgilaridan konsensus.

    Args:
        med, qwen: Ikki rol matni.

    Returns:
        True — to‘xtash mumkin.
    """
    m, q = med.lower(), qwen.lower()
    if "konsensus" in m or "konsensus" in q or "agree" in m or "agree" in q:
        return True
    ehtiyot = ("yetarsiz", "insufficient", "shifokor", "ehtiyotkor", "tashxis emas")
    return any(b in m for b in ehtiyot) and any(b in q for b in ehtiyot)


def _konsensus_bormi(xom_i: str, oraliq_z: str, med: str, qwen: str) -> Tuple[bool, str]:
    """Bosh kardiolog (yoki qoida) konsensusni baholaydi.

    Args:
        xom_i, oraliq_z: Qayta kiritilgan I/Z.
        med, qwen: Joriy raund fikrlari.

    Returns:
        (konsensus, sabab). Tashxis emas.
    """
    zaxira = {
        "konsensus": _konsensus_qoida(med, qwen),
        "sabab": "Qoida: ikkala rol I/Z asosida ehtiyotkor yoki ochiq kelishuv.",
    }
    natija = llm_json(
        tizim=(
            "Bosh kardiolog MDT raundini baholaydi. JSON: "
            '{"konsensus":true|false,"sabab":"qisqa"}. '
            "Tashxis qo‘ymang. Faqat I, Z va ikki fikrga tayaning."
        ),
        foydalanuvchi=f"I:\n{xom_i}\nZ:\n{oraliq_z}\nMedGemma:\n{med}\nQwen2.5-VL:\n{qwen}",
        zaxira=zaxira,
    )
    return bool(natija.get("konsensus")), str(natija.get("sabab") or zaxira["sabab"])


def mdt_munozara(
    xom_i: str,
    oraliq_z: str,
    max_raund: int = MAX_RAUND,
    bemor: Optional[Dict[str, Any]] = None,
    lab: Optional[Dict[str, Any]] = None,
    ecg: Optional[Dict[str, Any]] = None,
    echo: Optional[Dict[str, Any]] = None,
    segment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Ikkita VLM roli mustaqil yozadi, keyin almashadi; bosh umumlashtiradi.

    Args:
        xom_i: Bemor va xom signallar qisqasi.
        oraliq_z: Lab, EKG, echo, fellow.
        max_raund: Maksimal munozara raundlari.
        bemor: Vizual fayllar (EKG/echo).
        lab, ecg, echo, segment: Oraliq vosita chiqishlari (Z ga qo‘shimcha).

    Returns:
        raundlar, umumlashtirish, konsensus, manba. Tashxis emas.
    """
    viz = mdt_vizual_yig(bemor, ecg, segment)
    z_qosh = oraliq_z or ""
    if lab and (lab.get("rag_satr") or lab.get("xabar")):
        if "LAB:" not in z_qosh:
            z_qosh += "\nLAB: " + str(lab.get("rag_satr") or lab.get("xabar"))
    if echo and echo.get("xabar") and "ECHO:" not in z_qosh:
        z_qosh += "\nECHO: " + str(echo.get("xabar"))
    yoq = list(viz.get("yoq_dalillar") or [])

    med_tizim = (
        "Siz MedGemma — tibbiy tasvirlar (EKG grafik, echo still, LV overlay) eksperti. "
        "Har raundda berilgan I va Z ni qayta o‘qing. Yo‘q dalilni o‘ylab topmang. "
        "Tashxis qo‘ymang. 5–8 jumla, o‘zbek yoki ingliz."
    )
    qwen_tizim = (
        "Siz Qwen2.5-VL — echo video/cine ketma-ketligi eksperti. "
        "Kadrlar ketma-ketligini harakat sifatida ko‘ring; still bo‘lsa shunday yozing. "
        "Har raundda I va Z ni qayta o‘qing. Yo‘q dalilni o‘ylab topmang. "
        "Tashxis qo‘ymang. 5–8 jumla, o‘zbek yoki ingliz."
    )

    raundlar: List[Dict[str, Any]] = []
    med = ""
    qwen = ""
    med_manba = "shablon"
    qwen_manba = "shablon"
    konsensus = False
    konsensus_sabab = ""
    chegara = max(1, max_raund)

    for raund in range(1, chegara + 1):
        med, med_manba = _rol_javobi(
            "medgemma",
            med_tizim,
            xom_i,
            z_qosh,
            qwen,
            viz.get("still") or [],
            yoq,
            video=False,
        )
        qwen, qwen_manba = _rol_javobi(
            "qwen_vl",
            qwen_tizim,
            xom_i,
            z_qosh,
            med,
            viz.get("video") or [],
            yoq,
            video=True,
        )
        if med_manba == "shablon" and qwen_manba == "shablon":
            konsensus = _konsensus_qoida(med, qwen)
            konsensus_sabab = (
                "VLM/API yo‘q: I va Z qayta kiritildi, qoida bilan ehtiyotkor to‘xtash."
            )
        else:
            konsensus, konsensus_sabab = _konsensus_bormi(xom_i, z_qosh, med, qwen)
        raundlar.append(
            {
                "raund": str(raund),
                "medgemma": med,
                "qwen": qwen,
                "medgemma_manba": med_manba,
                "qwen_manba": qwen_manba,
                "konsensus": konsensus,
                "konsensus_sabab": konsensus_sabab,
            }
        )
        if konsensus:
            break
        if med_manba == "shablon" and qwen_manba == "shablon":
            break

    umumiy = None
    if not (med_manba == "shablon" and qwen_manba == "shablon"):
        umumiy = llm_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Bosh kardiolog MDT ni qisqa umumlashtiradi. Tashxis qo‘ymang. "
                        "I va Z ni qayta inobatga oling. Ikkala rol fikrini yonma-yon solishtiring."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"I (qayta):\n{xom_i}\nZ (qayta):\n{z_qosh}\n"
                        f"Konsensus={konsensus} ({konsensus_sabab})\nRaundlar:\n{raundlar}"
                    ),
                },
            ]
        )
    if not umumiy:
        umumiy = (
            f"MDT: MedGemma manba={med_manba}, Qwen2.5-VL manba={qwen_manba}. "
            f"I qayta kiritildi. Z qayta kiritildi. "
            f"Konsensus={konsensus} ({konsensus_sabab}). "
            "Qo‘shimcha klinik tasdiq kerak. Tashxis emas."
        )
    model = model_qisqacha()
    return {
        "ok": True,
        "raundlar": raundlar,
        "umumlashtirish": umumiy.strip(),
        "raund_soni": len(raundlar),
        "konsensus": konsensus,
        "konsensus_sabab": konsensus_sabab,
        "medgemma_manba": med_manba,
        "qwen_manba": qwen_manba,
        "medgemma_model": model.get("medgemma") if med_manba != "shablon" else "shablon",
        "qwen_model": model.get("qwen_vl") if qwen_manba != "shablon" else "shablon",
        "still_id": viz.get("still_id") or [],
        "video_id": viz.get("video_id") or [],
        "yoq_dalillar": yoq,
        "xom_i": xom_i,
        "oraliq_z": z_qosh[:2000],
        "xabar": "MDT yakunlandi. Shifokor o‘rnini bosmaydi.",
    }
