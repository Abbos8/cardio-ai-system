"""Cardiology fellow: yig‘ilgan unimodal natijalardan dastlabki ehtiyotkor xulosa.

Multimodal VLM bo‘lmasa, tuzilgan matn + ixtiyoriy LLM. Tashxis o‘rnini bosmaydi.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from llm.client import llm_chat


def dastlabki_tashxis(
    bemor: Dict[str, Any],
    lab: Optional[Dict[str, Any]] = None,
    ecg: Optional[Dict[str, Any]] = None,
    echo: Optional[Dict[str, Any]] = None,
    segment: Optional[Dict[str, Any]] = None,
    rag_dalillar: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Mavjud vosita natijalarini fellow xulosasiga yig‘adi.

    Args:
        bemor: Demografiya va shikoyat.
        lab, ecg, echo, segment: Ekspert vosita javoblari.
        rag_dalillar: CardiacRAG konteksti.

    Returns:
        matn, manba=llm|shablon. Klinik tashxis emas.
    """
    qismlar: List[str] = [
        "Cardiology fellow (dastlabki ko‘rik, tashxis emas).",
        f"Yosh={bemor.get('yosh')}, jins={bemor.get('jins') or bemor.get('sex')}, "
        f"shikoyat={bemor.get('shikoyatlar') or bemor.get('symptoms')}.",
    ]
    if lab and lab.get("ok"):
        qismlar.append(lab.get("rag_satr") or lab.get("matn") or "")
    elif lab:
        qismlar.append(f"Lab: {lab.get('xabar')}")
    if ecg and ecg.get("ok"):
        qismlar.append(
            f"EKG EP: HR={ecg.get('yurak_chastotasi_bpm')} bpm, QRS={ecg.get('qrs_ms')} ms, "
            f"PR={ecg.get('pr_ms')} ms, QT={ecg.get('qt_ms')} ms, "
            f"QTc_Bazett={ecg.get('qtc_bazett_ms')} ms, "
            f"HRV_SDNN={ecg.get('hrv_sdnn_ms')}, HRV_RMSSD={ecg.get('hrv_rmssd_ms')}, "
            f"tasma={ecg.get('tasma')}."
        )
        tech = ecg.get("technician")
        if isinstance(tech, dict):
            qismlar.append(f"EKG technician sifat={tech.get('sifat')}: {tech.get('xabar')}")
    elif ecg:
        qismlar.append(f"EKG: {ecg.get('xabar')}")
    if echo:
        qismlar.append(f"Echo tasnif: {echo.get('xabar')} | ko‘rinishlar={echo.get('korinishlar')}")
    if segment:
        qismlar.append(f"LV segmentatsiya: {segment.get('xabar')}")
    if rag_dalillar:
        qismlar.append("Adabiyot (tekshirish uchun): " + " | ".join(rag_dalillar[:3]))
    qismlar.append("Yakuniy qaror shifokorga tegishli.")
    shablon = "\n".join(p for p in qismlar if p)

    javob = llm_chat(
        [
            {
                "role": "system",
                "content": (
                    "Siz kardiologiya fellowisiz. Faqat ehtiyotkor dastlabki ko‘rik yozing. "
                    "Tashxis qo‘ymang. JSON emas, qisqa o‘zbek yoki ingliz matn. "
                    "Shifokor o‘rnini bosmasligini yozing."
                ),
            },
            {"role": "user", "content": shablon},
        ]
    )
    return {
        "ok": True,
        "matn": javob.strip() if javob else shablon,
        "manba": "llm" if javob else "shablon",
        "xabar": "Dastlabki fellow xulosasi. Tashxis emas.",
    }
