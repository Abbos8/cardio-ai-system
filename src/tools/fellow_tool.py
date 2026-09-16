"""Cardiology fellow: lab token + EKG grafik + echo kadr/maska dan dastlabki xulosa.

Faqat mavjud dalillar; yo‘q modalitet o‘ylab topilmaydi.
LLM/VLM bo‘lmasa shablon. Tashxis o‘rnini bosmaydi.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional, Tuple

from llm.client import llm_chat, llm_vision_model
from tools.ecg_tool import ekg_qisqa_png


def _ekg_signal_ol(bemor: Dict[str, Any]) -> Any:
    """Bemor lug‘atidan EKG massivini oladi.

    Args:
        bemor: Agent bemor maydonlari.

    Returns:
        Signal yoki None.
    """
    for kalit in ("ecg_signal", "ekg", "signal"):
        if bemor.get(kalit) is not None:
            return bemor[kalit]
    return None


def _echo_kadr_png(bemor: Dict[str, Any], segment: Optional[Dict[str, Any]]) -> Optional[bytes]:
    """LV overlay yoki birinchi echo kaderni PNG qiladi.

    Args:
        bemor: echo_fayllar.
        segment: echo_mask (overlay_png).

    Returns:
        PNG yoki None.
    """
    if segment and segment.get("overlay_png"):
        return bytes(segment["overlay_png"])
    fayllar = bemor.get("echo_fayllar") or []
    if not fayllar:
        return None
    try:
        from tools.echo_yuklash import echo_fayldan_kadrlar
        import numpy as np

        try:
            import cv2
        except ImportError:
            cv2 = None
        bir = fayllar[0]
        oq = echo_fayldan_kadrlar(str(bir.get("nom") or "echo"), bytes(bir.get("bayt") or b""))
        kadrlar = oq.get("kadrlar") or []
        if not kadrlar:
            return None
        kadr = np.asarray(kadrlar[0])
        if cv2 is None:
            return None
        if kadr.ndim == 2:
            rgb = np.stack([kadr, kadr, kadr], axis=-1)
        else:
            rgb = kadr
        ok, buf = cv2.imencode(".png", rgb)
        return bytes(buf) if ok else None
    except Exception:
        return None


def _data_url(png: bytes) -> str:
    """PNG ni OpenAI-mos image_url data URI ga aylantiradi.

    Args:
        png: Rasm baytlari.

    Returns:
        data:image/png;base64,...
    """
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _dalil_matn(
    bemor: Dict[str, Any],
    lab: Optional[Dict[str, Any]],
    ecg: Optional[Dict[str, Any]],
    echo: Optional[Dict[str, Any]],
    segment: Optional[Dict[str, Any]],
    rag_dalillar: Optional[List[str]],
) -> Tuple[str, List[str], List[str]]:
    """Faqat mavjud modalitetlardan matn yig‘adi.

    Args:
        bemor, lab, ecg, echo, segment, rag_dalillar: Vosita chiqishlari.

    Returns:
        matn, bor_dalillar, yoq_dalillar.
    """
    bor: List[str] = []
    yoq: List[str] = []
    qismlar: List[str] = [
        "Cardiology fellow — dastlabki ko‘rik (tashxis emas).",
        f"Yosh={bemor.get('yosh')}, jins={bemor.get('jins') or bemor.get('sex')}, "
        f"shikoyat={bemor.get('shikoyatlar') or bemor.get('symptoms')}.",
        "Faqat pastdagi mavjud dalillardan yozing; yo‘q narsani o‘ylab topmang.",
    ]
    if lab and (lab.get("ok") or lab.get("rag_satr") or lab.get("tokenlar")):
        bor.append("lab")
        qismlar.append("LAB: " + str(lab.get("rag_satr") or lab.get("matn") or lab.get("xabar") or ""))
        if lab.get("tokenlar"):
            qismlar.append("LAB tokenlar: " + ", ".join(str(t) for t in lab.get("tokenlar")[:24]))
    else:
        yoq.append("lab")
        qismlar.append("LAB: yo‘q.")

    if ecg and ecg.get("ok"):
        bor.append("ecg")
        qismlar.append(
            f"EKG (sonlar): HR={ecg.get('yurak_chastotasi_bpm')} bpm, "
            f"QRS={ecg.get('qrs_ms')} ms, PR={ecg.get('pr_ms')} ms, "
            f"QT={ecg.get('qt_ms')} ms, QTc={ecg.get('qtc_bazett_ms')} ms, "
            f"SDNN={ecg.get('hrv_sdnn_ms')}, tasma={ecg.get('tasma')}."
        )
    elif ecg:
        yoq.append("ecg_olchov")
        qismlar.append("EKG o‘lchov: " + str(ecg.get("xabar") or "muvaffaqiyatsiz"))
    else:
        yoq.append("ecg")
        qismlar.append("EKG: yo‘q.")

    if echo and echo.get("ok"):
        bor.append("echo")
        qismlar.append(
            f"ECHO ko‘rinishlar={echo.get('korinishlar')} model={echo.get('model')}."
        )
    else:
        yoq.append("echo")
        qismlar.append("ECHO: yo‘q yoki tasnif qilinmagan.")

    if segment and segment.get("ok"):
        bor.append("lv_maska")
        qismlar.append(
            f"LV maska: {segment.get('maydon_px')} px, ulush={segment.get('ulush')}, "
            f"ko‘rinish={segment.get('korinish')}."
        )
    else:
        yoq.append("lv_maska")
        qismlar.append("LV maska: yo‘q.")

    if rag_dalillar:
        bor.append("rag")
        qismlar.append("Adabiyot (tekshirish): " + " | ".join(rag_dalillar[:3]))
    qismlar.append("Yakuniy qaror shifokorga tegishli.")
    return "\n".join(qismlar), bor, yoq


def _rasmlarni_yig(
    bemor: Dict[str, Any],
    ecg: Optional[Dict[str, Any]],
    segment: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """EKG grafik va echo overlay PNG larini yig‘adi.

    Args:
        bemor: EKG signal, echo fayllar.
        ecg: Intervallar (tasma nomi).
        segment: Overlay PNG.

    Returns:
        [{id, png}]. Yo‘q bo‘lsa bo‘sh.
    """
    rasmlar: List[Dict[str, Any]] = []
    sig = _ekg_signal_ol(bemor)
    if sig is not None:
        tasma = (ecg or {}).get("tasma") or "II"
        sr = float(bemor.get("sampling_rate") or 500.0)
        png = ekg_qisqa_png(sig, sampling_rate=sr, tasma=str(tasma))
        if png:
            rasmlar.append({"id": "ecg_grafik", "png": png})
    echo_png = _echo_kadr_png(bemor, segment)
    if echo_png:
        kalit = "lv_overlay" if (segment and segment.get("overlay_png")) else "echo_kadr"
        rasmlar.append({"id": kalit, "png": echo_png})
    return rasmlar


def dastlabki_tashxis(
    bemor: Dict[str, Any],
    lab: Optional[Dict[str, Any]] = None,
    ecg: Optional[Dict[str, Any]] = None,
    echo: Optional[Dict[str, Any]] = None,
    segment: Optional[Dict[str, Any]] = None,
    rag_dalillar: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Lab + EKG grafik + echo kadr/maskani bitta fellow xulosasiga birlashtiradi.

    Args:
        bemor: Demografiya, EKG signal, echo fayllar.
        lab, ecg, echo, segment: Ekspert javoblari.
        rag_dalillar: CardiacRAG C.

    Returns:
        matn, manba=llm_multimodal|llm|shablon, dalillar. Tashxis emas.
    """
    matn, bor, yoq = _dalil_matn(bemor, lab, ecg, echo, segment, rag_dalillar)
    rasmlar = _rasmlarni_yig(bemor, ecg, segment)
    rasm_id = [r["id"] for r in rasmlar]
    if rasm_id:
        bor = list(bor) + [i for i in rasm_id if i not in bor]

    tizim = (
        "Siz kardiologiya fellowisiz. Faqat foydalanuvchi yuborgan matn va rasmlardan "
        "ehtiyotkor dastlabki ko‘rik yozing. Yo‘q modalitet (lab/EKG/echo/LV) haqida "
        "taxmin qilmang, tashxis qo‘ymang, EF aytmang. Rasmlar bo‘lsa ularni matndagi "
        "sonlar bilan solishtiring, lekin yangi o‘lchov o‘ylab topmang. "
        "Oxirida shifokor o‘rnini bosmasligini yozing. Qisqa o‘zbek yoki ingliz matn."
    )
    shablon = matn + "\nDalillar: " + ", ".join(bor) + ("\nYo‘q: " + ", ".join(yoq) if yoq else "")

    javob = None
    manba = "shablon"
    if rasmlar:
        qismlar: List[Dict[str, Any]] = [{"type": "text", "text": matn}]
        for r in rasmlar:
            qismlar.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url(r["png"])},
                }
            )
        javob = llm_chat(
            [
                {"role": "system", "content": tizim},
                {"role": "user", "content": qismlar},
            ],
            model=llm_vision_model(),
        )
        if javob:
            manba = "llm_multimodal"
    if javob is None:
        javob = llm_chat(
            [
                {"role": "system", "content": tizim},
                {"role": "user", "content": matn},
            ]
        )
        if javob:
            manba = "llm"

    return {
        "ok": True,
        "matn": javob.strip() if javob else shablon,
        "manba": manba,
        "dalillar": bor,
        "yoq_dalillar": yoq,
        "rasmlar": rasm_id,
        "xabar": f"Fellow xulosa (manba={manba}). Tashxis emas.",
    }
