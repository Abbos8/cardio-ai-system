"""LangGraph asosidagi bosh kardiolog: 6 bosqichli klinik workflow.

1) Murakkablik bahosi  2) CardiacRAG reja  3) ekspert vositalar
4) CONTINUE/STOP yangilanish  5) MDT  6) yakuniy xulosa.
Xulosa shifokor o‘rnini bosmaydi.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agents.mdt import mdt_munozara
from llm.client import llm_json
from rag.medical_rag import MedicalRAG
from tools.ecg_tool import ecg_technician_tahlil, electrophysiologist_tahlil
from tools.echo_segmenter import segment_lv
from tools.echo_tool import classify_echo_views, echo_bormi
from tools.fellow_tool import dastlabki_tashxis
from tools.lab_tool import process_lab

Amal = Literal["CONTINUE", "STOP"]
MAX_QADAM = 16

QRS_KENG_MS = 120.0
PR_UZOQ_MS = 200.0
QT_UZOQ_MS = 460.0
HR_PAST = 50.0
HR_YUQori = 100.0


class ChiefHolat(TypedDict, total=False):
    """Graf holati: reja, vositalar, MDT va CONTINUE/STOP."""

    bemor: Dict[str, Any]
    murakkablik: str
    murakkablik_sababi: str
    rag_dalillar: List[str]
    reja: List[Dict[str, Any]]
    reja_indeks: int
    lab_natija: Optional[Dict[str, Any]]
    ecg_natija: Optional[Dict[str, Any]]
    ecg_technician_natija: Optional[Dict[str, Any]]
    ep_natija: Optional[Dict[str, Any]]
    echo_natija: Optional[Dict[str, Any]]
    echo_mask: Optional[Dict[str, Any]]
    fellow_natija: Optional[Dict[str, Any]]
    mdt_natija: Optional[Dict[str, Any]]
    mdt_kerak: bool
    xom_i: str
    oraliq_z: str
    umumlashtirish: str
    vizual: Dict[str, Any]
    qadam: int
    amal: Amal
    baho_sababi: str
    rejali_tugun: str
    qadam_tarixi: List[str]
    xulosa: str


def _bemorni_normallashtir(malumot: Any) -> Dict[str, Any]:
    """Kirishni agent kutadigan bemor lug‘atiga keltiradi.

    Args:
        malumot: To‘g‘ridan-to‘g‘ri bemor dict yoki {\"bemor\": {...}}.

    Returns:
        Yosh, shikoyat, EKG, lab va boshqa klinik maydonlar.
    """
    if not isinstance(malumot, dict):
        return {}
    if isinstance(malumot.get("bemor"), dict):
        asos = dict(malumot["bemor"])
        for kalit in ("ecg_signal", "sampling_rate", "klinik_savol"):
            if kalit in malumot and kalit not in asos:
                asos[kalit] = malumot[kalit]
        return asos
    return dict(malumot)


def _son(qiymat: Any) -> Optional[float]:
    """O‘lchovni float ga o‘giradi.

    Args:
        qiymat: HR, QRS, lab qiymati.

    Returns:
        Son yoki None.
    """
    if qiymat is None:
        return None
    try:
        return float(qiymat)
    except (TypeError, ValueError):
        return None


def _echo_qisqacha(natija: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Graf holatida katta kadr massivlarini saqlamaslik.

    Args:
        natija: classify_echo_views javobi.

    Returns:
        Preview-siz qisqa lug‘at.
    """
    if not natija:
        return {}
    return {
        "ok": natija.get("ok"),
        "xabar": natija.get("xabar"),
        "korinishlar": natija.get("korinishlar") or [],
        "yozuvlar": natija.get("yozuvlar") or [],
        "kadrlar_soni": natija.get("kadrlar_soni"),
        "model": natija.get("model"),
        "yetishmagan_standart": natija.get("yetishmagan_standart") or [],
    }


def _lv_qisqacha(natija: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Graf holatida to‘liq piksel maskani tashlamaydi (faqat overlay PNG).

    Args:
        natija: segment_lv javobi.

    Returns:
        Signallarsiz qisqa lug‘at. Overlay kichik PNG.
    """
    if not natija:
        return {}
    return {
        "ok": natija.get("ok"),
        "xabar": natija.get("xabar"),
        "maydon_px": natija.get("maydon_px"),
        "ulush": natija.get("ulush"),
        "shakl": natija.get("shakl"),
        "korinish": natija.get("korinish"),
        "fayl": natija.get("fayl"),
        "model": natija.get("model"),
        "overlay_png": natija.get("overlay_png"),
        "kadrlar_soni": natija.get("kadrlar_soni"),
    }


def _ekg_qisqacha(natija: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Holatda saqlash uchun EKG dan o‘lchovlarni qoldiradi (katta signal emas).

    Args:
        natija: process_ecg_signal / technician / EP javobi.

    Returns:
        Signallarsiz qisqa lug‘at (to‘lqin indekslari saqlanadi).
    """
    if not natija:
        return {}
    tech = natija.get("technician")
    ep = natija.get("ep")
    return {
        "ok": natija.get("ok"),
        "xabar": natija.get("xabar"),
        "yurak_chastotasi_bpm": natija.get("yurak_chastotasi_bpm"),
        "qrs_ms": natija.get("qrs_ms"),
        "pr_ms": natija.get("pr_ms"),
        "qt_ms": natija.get("qt_ms"),
        "qtc_bazett_ms": natija.get("qtc_bazett_ms"),
        "tasma": natija.get("tasma"),
        "urishlar_soni": natija.get("urishlar_soni"),
        "hrv_sdnn_ms": natija.get("hrv_sdnn_ms"),
        "hrv_rmssd_ms": natija.get("hrv_rmssd_ms"),
        "tolqinlar": natija.get("tolqinlar") or {},
        "yetishmagan": natija.get("yetishmagan") or [],
        "technician": tech,
        "ep": ep,
        "sifat": (tech or {}).get("sifat") if isinstance(tech, dict) else None,
    }


def _ekg_signal_ol(bemor: Dict[str, Any]) -> Any:
    """Bemor lug‘atidan EKG massivini oladi.

    Args:
        bemor: Agent holatidagi bemor.

    Returns:
        Signal yoki None.
    """
    for kalit in ("ecg_signal", "ekg", "signal"):
        if kalit in bemor and bemor[kalit] is not None:
            return bemor[kalit]
    return None


def _ekg_signal_bormi(bemor: Dict[str, Any]) -> bool:
    """12 tasmali EKG bor-yo‘qligini tekshiradi.

    Args:
        bemor: Bemor lug‘ati.

    Returns:
        True — process_ecg_signal chaqirish mumkin.
    """
    signal = _ekg_signal_ol(bemor)
    if signal is None:
        return False
    if isinstance(signal, str):
        return bool(signal.strip())
    return True


def _rag_sorov_tuz(bemor: Dict[str, Any], ecg: Optional[Dict[str, Any]] = None) -> str:
    """Shikoyat, lab va EKG dan CardiacRAG so‘rovini yig‘adi.

    Args:
        bemor: Klinik maydonlar.
        ecg: Ixtiyoriy EKG o‘lchovi.

    Returns:
        Inglizcha-klinik so‘rov matni.
    """
    qismlar: List[str] = []
    savol = bemor.get("klinik_savol") or bemor.get("savol")
    if savol:
        qismlar.append(str(savol).strip())
    if bemor.get("yosh") is not None:
        qismlar.append(f"age {bemor.get('yosh')}")
    jins = bemor.get("jins") or bemor.get("sex")
    if jins:
        qismlar.append(str(jins))
    shikoyat = bemor.get("shikoyatlar") or bemor.get("symptoms") or bemor.get("shikoyat")
    if shikoyat:
        qismlar.append(f"symptoms: {shikoyat}")
    anamnez = bemor.get("anamnez") or bemor.get("history")
    if anamnez:
        qismlar.append(f"history: {anamnez}")
    lab_nat = process_lab(bemor)
    if lab_nat.get("rag_satr"):
        qismlar.append(lab_nat["rag_satr"])
    elif isinstance(bemor.get("laboratoriya"), dict) and bemor.get("laboratoriya"):
        qismlar.append("laboratory " + ", ".join(f"{k}={v}" for k, v in bemor["laboratoriya"].items()))
    if ecg and ecg.get("ok"):
        hr = _son(ecg.get("yurak_chastotasi_bpm"))
        qrs = _son(ecg.get("qrs_ms"))
        pr = _son(ecg.get("pr_ms"))
        qt = _son(ecg.get("qt_ms"))
        if hr is not None:
            qismlar.append(f"heart rate {hr:.0f} bpm")
            if hr < HR_PAST:
                qismlar.append("bradycardia ECG intervals")
            elif hr > HR_YUQori:
                qismlar.append("tachycardia ECG intervals")
        if qrs is not None:
            qismlar.append(f"QRS {qrs:.0f} ms")
            if qrs >= QRS_KENG_MS:
                qismlar.append("wide QRS complex")
        if pr is not None:
            qismlar.append(f"PR {pr:.0f} ms")
            if pr > PR_UZOQ_MS:
                qismlar.append("prolonged PR interval AV conduction")
        if qt is not None:
            qismlar.append(f"QT {qt:.0f} ms")
            if qt >= QT_UZOQ_MS:
                qismlar.append("prolonged QT interval")
    if not qismlar:
        return "cardiology ECG interpretation clinical guidelines"
    return ". ".join(qismlar)


def _oddiy_reja(bemor: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Murakkab bo‘lmagan holat uchun qisqa vosita ketma-ketligi.

    Args:
        bemor: EKG/lab mavjudligi.

    Returns:
        pending qadamlar.
    """
    qadamlar = [
        {"id": "p1", "vosita": "lab_technician", "tavsif": "Laboratoriya va dorilarni tokenlash", "holat": "pending"},
    ]
    if _ekg_signal_bormi(bemor):
        qadamlar.append(
            {
                "id": "p2",
                "vosita": "ecg_technician",
                "tavsif": "EKG tozalash va sifat",
                "holat": "pending",
            }
        )
        qadamlar.append(
            {
                "id": "p3",
                "vosita": "electrophysiologist",
                "tavsif": "P/QRS/T, PR/QRS/QT/QTc, HRV",
                "holat": "pending",
            }
        )
    if echo_bormi(bemor):
        qadamlar.append(
            {
                "id": "p-echo",
                "vosita": "echo_technician",
                "tavsif": "DICOM/video 11 ko‘rinish",
                "holat": "pending",
            }
        )
        qadamlar.append(
            {
                "id": "p-lv",
                "vosita": "echo_segmenter",
                "tavsif": "LV kontur maska",
                "holat": "pending",
            }
        )
    qadamlar.append(
        {"id": "p4", "vosita": "cardiology_fellow", "tavsif": "Dastlabki xulosa", "holat": "pending"}
    )
    return qadamlar


def _lugat_z_qisqa(obyekt: Any, max_len: int = 700) -> str:
    """Z matnida bayt/maska maydonlarini qisqartiradi (MDT gallyutsinatsiya cheklovi).

    Args:
        obyekt: Vosita natijasi.
        max_len: Qisqartirish.

    Returns:
        O‘qiladigan qisqa satr.
    """
    if not isinstance(obyekt, dict):
        return str(obyekt)[:max_len]
    otkaz = {"overlay_png", "maska", "kadrlar", "preview", "signal"}
    qism: List[str] = []
    for kalit, qiymat in obyekt.items():
        if kalit in otkaz or isinstance(qiymat, (bytes, bytearray)):
            continue
        if isinstance(qiymat, dict):
            qism.append(f"{kalit}={_lugat_z_qisqa(qiymat, 240)}")
        else:
            qism.append(f"{kalit}={qiymat}")
    return "; ".join(qism)[:max_len]


def _oraliq_matn(holat: ChiefHolat) -> str:
    """Z — bajarilgan vositalar qisqasi (MDT va yangilanish uchun).

    Args:
        holat: Joriy graf holati.

    Returns:
        Matn. Overlay PNG kiritilmaydi.
    """
    qism: List[str] = []
    if holat.get("lab_natija"):
        qism.append("LAB: " + _lugat_z_qisqa(holat["lab_natija"]))
    if holat.get("ecg_technician_natija"):
        qism.append("ECG_TECH: " + _lugat_z_qisqa(holat["ecg_technician_natija"]))
    if holat.get("ep_natija"):
        qism.append("ECG_EP: " + _lugat_z_qisqa(holat["ep_natija"]))
    elif holat.get("ecg_natija"):
        qism.append("ECG: " + _lugat_z_qisqa(holat["ecg_natija"]))
    if holat.get("echo_natija"):
        qism.append("ECHO: " + _lugat_z_qisqa(holat["echo_natija"]))
    if holat.get("echo_mask"):
        qism.append("LV: " + _lugat_z_qisqa(holat["echo_mask"]))
    if holat.get("fellow_natija"):
        qism.append("FELLOW: " + str((holat["fellow_natija"] or {}).get("matn", "")[:800]))
    if holat.get("rag_dalillar"):
        qism.append("C: " + " | ".join(holat["rag_dalillar"][:3]))
    return "\n".join(qism) if qism else "(oraliq natija yo‘q)"


class ChiefCardiologist:
    """6 bosqichli workflow ni LangGraph da boshqaradigan bosh kardiolog."""

    def __init__(self, rag: Optional[MedicalRAG] = None, max_qadam: int = MAX_QADAM) -> None:
        """Vositalarni bog‘laydi va grafni kompilyatsiya qiladi.

        Args:
            rag: Indekslangan MedicalRAG (CardiacRAG). None — seed uriniladi.
            max_qadam: Cheksiz sikl chegarasi.

        Returns:
            None. Graf `self.graf` da.
        """
        self.rag = rag
        self.max_qadam = max_qadam
        self.graf = self._graf_qur()

    def _qabul_qilish(self, holat: ChiefHolat) -> Dict[str, Any]:
        """Bemor ma’lumotlarini qabul qiladi (workflow 1-qism).

        Args:
            holat: LangGraph boshlang‘ich holati.

        Returns:
            Normallashtirilgan bemor va bo‘sh maydonlar.
        """
        bemor = _bemorni_normallashtir(holat.get("bemor") or holat)
        xom = (
            f"yosh={bemor.get('yosh')} jins={bemor.get('jins')} "
            f"shikoyat={bemor.get('shikoyatlar')} anamnez={bemor.get('anamnez')} "
            f"lab={bemor.get('laboratoriya')} ekg_bor={_ekg_signal_bormi(bemor)} "
            f"echo_bor={echo_bormi(bemor)}"
        )
        tarix = list(holat.get("qadam_tarixi") or [])
        tarix.append("qabul_qilish: bemor yozuvi o‘qildi")
        return {
            "bemor": bemor,
            "murakkablik": "",
            "murakkablik_sababi": "",
            "rag_dalillar": [],
            "reja": [],
            "reja_indeks": 0,
            "lab_natija": None,
            "ecg_natija": None,
            "ecg_technician_natija": None,
            "ep_natija": None,
            "echo_natija": None,
            "echo_mask": None,
            "fellow_natija": None,
            "mdt_natija": None,
            "mdt_kerak": False,
            "xom_i": xom,
            "oraliq_z": "",
            "umumlashtirish": "",
            "vizual": {},
            "qadam": 1,
            "amal": "CONTINUE",
            "baho_sababi": "",
            "rejali_tugun": "murakkablik_baholash",
            "qadam_tarixi": tarix,
            "xulosa": "",
        }

    def _murakkablik_baholash(self, holat: ChiefHolat) -> Dict[str, Any]:
        """DeepSeek-R1 (yoki qoida) bilan vazifa murakkabligini belgilaydi.

        Args:
            holat: Qabul qilingan bemor.

        Returns:
            murakkablik=oddiy|murakkab. Murakkab → CardiacRAG.
        """
        bemor = holat.get("bemor") or {}
        lab = bemor.get("laboratoriya") or {}
        trop = _son(lab.get("troponin_i") or lab.get("troponin"))
        ntp = _son(lab.get("nt_probnp"))
        echo_bor = echo_bormi(bemor)
        belgilar = []
        if trop is not None and trop >= 34:
            belgilar.append("yuqori troponin")
        if ntp is not None and ntp >= 300:
            belgilar.append("yuqori NT-proBNP")
        if echo_bor:
            belgilar.append("echo/DICOM bor")
        if _ekg_signal_bormi(bemor) and (trop is not None or echo_bor):
            belgilar.append("ko‘p modalitet")
        zaxira = {
            "murakkablik": "murakkab" if belgilar else "oddiy",
            "sabab": "; ".join(belgilar) if belgilar else "Bir modalitet, ogohlantiruvchi biomarker yo‘q.",
        }
        natija = llm_json(
            tizim=(
                "Siz DeepSeek-R1 asosidagi bosh kardiologsiz. JSON: "
                '{"murakkablik":"oddiy"|"murakkab","sabab":"..."}. '
                "Murakkab: ko‘p modalitet, noaniq EKG, yuqori troponin, echo. Tashxis qo‘ymang."
            ),
            foydalanuvchi=holat.get("xom_i") or "",
            zaxira=zaxira,
        )
        daraja = str(natija.get("murakkablik") or zaxira["murakkablik"]).lower()
        if daraja not in ("oddiy", "murakkab"):
            daraja = zaxira["murakkablik"]
        sabab = str(natija.get("sabab") or zaxira["sabab"])
        tarix = list(holat.get("qadam_tarixi") or [])
        tarix.append(f"murakkablik_baholash: {daraja} — {sabab}")
        return {
            "murakkablik": daraja,
            "murakkablik_sababi": sabab,
            "qadam": int(holat.get("qadam") or 0) + 1,
            "qadam_tarixi": tarix,
            "amal": "CONTINUE",
        }

    def _cardiac_rag_reja(self, holat: ChiefHolat) -> Dict[str, Any]:
        """Gibrid retrieval (C) va dastlabki klinik reja P ni tuzadi.

        Args:
            holat: Murakkablik va bemor.

        Returns:
            reja, rag_dalillar. Oddiy holatda qisqa reja.
        """
        bemor = holat.get("bemor") or {}
        tarix = list(holat.get("qadam_tarixi") or [])
        if holat.get("murakkablik") != "murakkab" or self.rag is None:
            reja = _oddiy_reja(bemor)
            tarix.append(
                "cardiac_rag_reja: oddiy reja (RAG o‘tkazildi)"
                if holat.get("murakkablik") != "murakkab"
                else "cardiac_rag_reja: RAG yo‘q — oddiy reja"
            )
            return {
                "reja": reja,
                "reja_indeks": 0,
                "rag_dalillar": [],
                "qadam": int(holat.get("qadam") or 0) + 1,
                "qadam_tarixi": tarix,
            }
        sorov = _rag_sorov_tuz(bemor)
        try:
            dalillar = self.rag.retrieve(sorov, n=3)
            reja = self.rag.reja_tuz(sorov, dalillar)
        except Exception as exc:
            dalillar = []
            reja = _oddiy_reja(bemor)
            tarix.append(f"cardiac_rag_reja: xato {exc}")
            return {
                "reja": reja,
                "reja_indeks": 0,
                "rag_dalillar": dalillar,
                "qadam": int(holat.get("qadam") or 0) + 1,
                "qadam_tarixi": tarix,
            }
        tarix.append(f"cardiac_rag_reja: {len(dalillar)} ta C, {len(reja)} ta qadam")
        return {
            "reja": reja,
            "reja_indeks": 0,
            "rag_dalillar": dalillar,
            "qadam": int(holat.get("qadam") or 0) + 1,
            "qadam_tarixi": tarix,
        }

    def _qadam_bajarish(self, holat: ChiefHolat) -> Dict[str, Any]:
        """Rejadagi navbatdagi ekspert vositasini chaqiradi.

        Args:
            holat: reja va indeks.

        Returns:
            Yangilangan vosita natijalari va reja holati.
        """
        bemor = holat.get("bemor") or {}
        reja = [dict(q) for q in (holat.get("reja") or [])]
        indeks = int(holat.get("reja_indeks") or 0)
        tarix = list(holat.get("qadam_tarixi") or [])
        yangi: Dict[str, Any] = {
            "qadam": int(holat.get("qadam") or 0) + 1,
            "qadam_tarixi": tarix,
        }
        if indeks >= len(reja):
            tarix.append("qadam_bajarish: reja tugadi")
            yangi["reja"] = reja
            return yangi

        qadam = reja[indeks]
        vosita = qadam.get("vosita")
        try:
            if vosita == "lab_technician":
                yangi["lab_natija"] = process_lab(bemor)
            elif vosita == "ecg_technician":
                signal = _ekg_signal_ol(bemor)
                if signal is None:
                    tech_q = {"ok": False, "xabar": "EKG signali yo‘q."}
                    yangi["ecg_technician_natija"] = tech_q
                    yangi["ecg_natija"] = tech_q
                else:
                    toliq = ecg_technician_tahlil(signal, sampling_rate=bemor.get("sampling_rate", 500.0))
                    tech_q = _ekg_qisqacha(toliq)
                    yangi["ecg_technician_natija"] = tech_q.get("technician") or tech_q
                    yangi["ecg_natija"] = tech_q
                    viz = dict(holat.get("vizual") or {})
                    viz["ekg_tozalangan"] = bool(toliq.get("tozalangan_signallar"))
                    viz["ekg_xabar"] = toliq.get("xabar")
                    viz["ekg_sifat"] = (toliq.get("technician") or {}).get("sifat")
                    yangi["vizual"] = viz
            elif vosita == "electrophysiologist":
                signal = _ekg_signal_ol(bemor)
                if signal is None:
                    ep_q = {"ok": False, "xabar": "EKG signali yo‘q."}
                    yangi["ep_natija"] = ep_q
                    yangi["ecg_natija"] = holat.get("ecg_natija") or ep_q
                    tarix.append("electrophysiologist: signal yo‘q")
                else:
                    toliq = electrophysiologist_tahlil(
                        signal, sampling_rate=bemor.get("sampling_rate", 500.0)
                    )
                    ep_q = _ekg_qisqacha(toliq)
                    yangi["ep_natija"] = ep_q.get("ep") or ep_q
                    birlash = dict(holat.get("ecg_natija") or {})
                    birlash.update(ep_q)
                    birlash["technician"] = (holat.get("ecg_technician_natija") or birlash.get("technician"))
                    yangi["ecg_natija"] = birlash
                    viz = dict(holat.get("vizual") or {})
                    viz["ekg_ep"] = ep_q.get("xabar")
                    viz["ekg_tasma"] = ep_q.get("tasma")
                    yangi["vizual"] = viz
                    tarix.append("electrophysiologist: P/QRS/T va intervallar")
            elif vosita == "echo_technician":
                toliq = classify_echo_views(bemor)
                yangi["echo_natija"] = _echo_qisqacha(toliq)
                viz = dict(holat.get("vizual") or {})
                viz["echo_ok"] = bool(toliq.get("ok"))
                viz["echo_korinishlar"] = toliq.get("korinishlar")
                viz["echo_model"] = toliq.get("model")
                viz["echo_kadrlar"] = toliq.get("kadrlar_soni")
                viz["echo_yetishmagan"] = toliq.get("yetishmagan_standart") or []
                yangi["vizual"] = viz
            elif vosita == "echo_segmenter":
                toliq = segment_lv(
                    bemor=bemor,
                    echo_natija=holat.get("echo_natija") or yangi.get("echo_natija"),
                )
                yangi["echo_mask"] = _lv_qisqacha(toliq)
                viz = dict(holat.get("vizual") or {})
                viz["lv_maska"] = bool(toliq.get("ok"))
                viz["lv_xabar"] = toliq.get("xabar")
                viz["lv_ulush"] = toliq.get("ulush")
                yangi["vizual"] = viz
            elif vosita == "cardiology_fellow":
                yangi["fellow_natija"] = dastlabki_tashxis(
                    bemor,
                    lab=holat.get("lab_natija") or yangi.get("lab_natija"),
                    ecg=holat.get("ecg_natija") or yangi.get("ecg_natija"),
                    echo=holat.get("echo_natija") or yangi.get("echo_natija"),
                    segment=holat.get("echo_mask") or yangi.get("echo_mask"),
                    rag_dalillar=holat.get("rag_dalillar"),
                )
                viz = dict(holat.get("vizual") or {})
                viz["fellow_manba"] = (yangi["fellow_natija"] or {}).get("manba")
                viz["fellow_dalillar"] = (yangi["fellow_natija"] or {}).get("dalillar")
                yangi["vizual"] = viz
            else:
                tarix.append(f"qadam_bajarish: noma’lum vosita {vosita}")
        except Exception as exc:
            qadam["holat"] = "xato"
            tarix.append(f"qadam_bajarish: {vosita} xato — {exc}")
            reja[indeks] = qadam
            yangi["reja"] = reja
            yangi["reja_indeks"] = indeks + 1
            return yangi

        qadam["holat"] = "done"
        reja[indeks] = qadam
        yangi["reja"] = reja
        yangi["reja_indeks"] = indeks + 1
        if vosita != "electrophysiologist":
            natija_qisqa = (
                yangi.get("lab_natija")
                or yangi.get("ecg_natija")
                or yangi.get("echo_natija")
                or yangi.get("echo_mask")
                or yangi.get("fellow_natija")
                or {}
            )
            xabar = ""
            if isinstance(natija_qisqa, dict):
                xabar = str(natija_qisqa.get("xabar") or natija_qisqa.get("matn") or "")[:160]
            tarix.append(f"qadam_bajarish: {vosita} — {xabar}")
        return yangi

    def _stepwise_yangilash(self, holat: ChiefHolat) -> Dict[str, Any]:
        """S ni umumlashtiradi, A=STOP/CONTINUE, kerak bo‘lsa P_{s+1}.

        Args:
            holat: Oxirgi vosita natijasi.

        Returns:
            amal, mdt_kerak, umumlashtirish.
        """
        reja = holat.get("reja") or []
        indeks = int(holat.get("reja_indeks") or 0)
        qadam_n = int(holat.get("qadam") or 0)
        z = _oraliq_matn(holat)
        tarix = list(holat.get("qadam_tarixi") or [])

        echo = holat.get("echo_natija") or {}
        ecg = holat.get("ecg_natija")
        noaniq = False
        if ecg is not None and ecg.get("ok") is False:
            noaniq = True
        if echo.get("ok") is False and holat.get("murakkablik") == "murakkab":
            noaniq = True
        if holat.get("echo_mask") and holat["echo_mask"].get("ok") is False and holat.get("murakkablik") == "murakkab":
            noaniq = True

        bemor = holat.get("bemor") or {}
        vizual = _ekg_signal_bormi(bemor) or echo_bormi(bemor)
        murakkab = holat.get("murakkablik") == "murakkab"
        zaxira = {
            "S": "Vosita natijasi qabul qilindi.",
            "A": "CONTINUE" if indeks < len(reja) else "STOP",
            "mdt": noaniq or vizual or murakkab,
            "P_next": None,
        }
        if qadam_n >= self.max_qadam:
            zaxira["A"] = "STOP"
            zaxira["S"] = "Qadam limiti."

        natija = llm_json(
            tizim=(
                "Bosh kardiolog stepwise update. JSON: "
                '{"S":"qisqa xulosa","A":"CONTINUE"|"STOP","mdt":true|false,'
                '"P_next":null yoki {"id":"pX","vosita":"...","tavsif":"..."}}. '
                "Tashxis qo‘ymang. Xato tarqalishini to‘xtating."
            ),
            foydalanuvchi=f"I:\n{holat.get('xom_i')}\nZ:\n{z}\nreja_indeks={indeks}/{len(reja)}",
            zaxira=zaxira,
        )
        amal: Amal = "STOP" if str(natija.get("A") or zaxira["A"]).upper() == "STOP" else "CONTINUE"
        if indeks >= len(reja):
            amal = "STOP"
        if qadam_n >= self.max_qadam:
            amal = "STOP"
        mdt_kerak = amal == "STOP" and (
            bool(natija.get("mdt", zaxira["mdt"])) or vizual or murakkab or noaniq
        )
        s_matn = str(natija.get("S") or zaxira["S"])
        p_next = natija.get("P_next")
        if amal == "CONTINUE" and isinstance(p_next, dict) and p_next.get("vosita"):
            yangi_reja = [dict(q) for q in reja]
            yangi_reja.append(
                {
                    "id": str(p_next.get("id") or f"p{len(yangi_reja)+1}"),
                    "vosita": p_next["vosita"],
                    "tavsif": str(p_next.get("tavsif") or ""),
                    "holat": "pending",
                }
            )
            reja = yangi_reja

        tarix.append(f"stepwise: {amal} — {s_matn[:120]} mdt={mdt_kerak}")
        return {
            "amal": amal,
            "baho_sababi": s_matn,
            "umumlashtirish": s_matn,
            "oraliq_z": z,
            "mdt_kerak": mdt_kerak,
            "reja": reja,
            "qadam": qadam_n + 1,
            "qadam_tarixi": tarix,
        }

    def _mdt(self, holat: ChiefHolat) -> Dict[str, Any]:
        """MedGemma va Qwen2.5-VL rollari bilan munozara (I va Z qayta kiritiladi).

        Args:
            holat: xom_i va oraliq_z.

        Returns:
            mdt_natija.
        """
        tarix = list(holat.get("qadam_tarixi") or [])
        natija = mdt_munozara(
            holat.get("xom_i") or "",
            holat.get("oraliq_z") or _oraliq_matn(holat),
            bemor=holat.get("bemor") or {},
            lab=holat.get("lab_natija"),
            ecg=holat.get("ecg_natija") or holat.get("ep_natija"),
            echo=holat.get("echo_natija"),
            segment=holat.get("echo_mask"),
        )
        tarix.append(
            f"mdt: {natija.get('raund_soni')} raund "
            f"konsensus={natija.get('konsensus')} "
            f"med={natija.get('medgemma_manba')} qwen={natija.get('qwen_manba')}"
        )
        viz = dict(holat.get("vizual") or {})
        viz["mdt_raund"] = natija.get("raund_soni")
        viz["mdt_konsensus"] = natija.get("konsensus")
        viz["mdt_med"] = natija.get("medgemma_manba")
        viz["mdt_qwen"] = natija.get("qwen_manba")
        return {
            "mdt_natija": natija,
            "vizual": viz,
            "qadam": int(holat.get("qadam") or 0) + 1,
            "qadam_tarixi": tarix,
            "amal": "STOP",
        }

    def _xulosa_tayyorlash(self, holat: ChiefHolat) -> Dict[str, Any]:
        """Barcha qadam va MDT ni yakuniy ehtiyotkor xulosaga yig‘adi.

        Args:
            holat: To‘liq graf holati.

        Returns:
            xulosa, amal=STOP.
        """
        bemor = holat.get("bemor") or {}
        qatorlar: List[str] = [
            "Bosh kardiolog yakuniy xulosasi (klinik yordam, tashxis emas).",
            f"Murakkablik: {holat.get('murakkablik')} ({holat.get('murakkablik_sababi')}).",
        ]
        yosh = bemor.get("yosh")
        jins = bemor.get("jins") or bemor.get("sex")
        shikoyat = bemor.get("shikoyatlar") or bemor.get("symptoms")
        qatorlar.append(f"Bemor: yosh={yosh}, jins={jins}, shikoyat={shikoyat}.")

        lab = holat.get("lab_natija")
        if lab:
            qatorlar.append(lab.get("matn") or lab.get("xabar") or "")
        ecg = holat.get("ecg_natija")
        if ecg:
            if ecg.get("ok"):
                qatorlar.append(
                    "EKG EP (taxminiy): "
                    f"tasma={ecg.get('tasma')}, HR={ecg.get('yurak_chastotasi_bpm')} bpm, "
                    f"QRS={ecg.get('qrs_ms')} ms, PR={ecg.get('pr_ms')} ms, "
                    f"QT={ecg.get('qt_ms')} ms, QTc_Bazett={ecg.get('qtc_bazett_ms')} ms, "
                    f"SDNN={ecg.get('hrv_sdnn_ms')}, RMSSD={ecg.get('hrv_rmssd_ms')}."
                )
                tech = holat.get("ecg_technician_natija") or ecg.get("technician")
                if isinstance(tech, dict) and tech.get("sifat"):
                    qatorlar.append(
                        f"EKG technician: sifat={tech.get('sifat')}, "
                        f"tozalangan_tasma={tech.get('tozalangan_tasma_soni')}, "
                        f"{tech.get('xabar') or ''}"
                    )
                ogoh: List[str] = []
                hr, qrs, pr, qt = (
                    _son(ecg.get("yurak_chastotasi_bpm")),
                    _son(ecg.get("qrs_ms")),
                    _son(ecg.get("pr_ms")),
                    _son(ecg.get("qt_ms")),
                )
                if hr is not None and (hr < HR_PAST or hr > HR_YUQori):
                    ogoh.append("yurak chastotasi odatdagi oraliqdan tashqarida")
                if qrs is not None and qrs >= QRS_KENG_MS:
                    ogoh.append("QRS ≥120 ms")
                if pr is not None and pr > PR_UZOQ_MS:
                    ogoh.append("PR >200 ms")
                if qt is not None and qt >= QT_UZOQ_MS:
                    ogoh.append("QT uzayishi ehtimoli")
                if ogoh:
                    qatorlar.append("Shifokor e’tibori: " + "; ".join(ogoh) + ".")
            else:
                qatorlar.append(f"EKG: {ecg.get('xabar')}")
        echo = holat.get("echo_natija")
        if echo:
            qatorlar.append(
                f"Echo technician: ko‘rinishlar={echo.get('korinishlar')} "
                f"model={echo.get('model')} kadr={echo.get('kadrlar_soni')}. "
                f"{echo.get('xabar')}"
            )
        mask = holat.get("echo_mask")
        if mask:
            qatorlar.append(f"LV segmentatsiya: {mask.get('xabar')}")
        fellow = holat.get("fellow_natija")
        if fellow:
            qatorlar.append(
                f"Fellow (manba={fellow.get('manba')} dalillar={fellow.get('dalillar')}): "
                + str(fellow.get("matn") or "")[:1200]
            )
        if holat.get("rag_dalillar"):
            qatorlar.append("CardiacRAG C (tekshirish uchun):")
            for i, matn in enumerate(holat["rag_dalillar"], start=1):
                qatorlar.append(f"  {i}. {matn[:400]}")
        mdt = holat.get("mdt_natija")
        if mdt:
            qatorlar.append(
                "MDT: "
                + str(mdt.get("umumlashtirish") or "")[:800]
                + f" (konsensus={mdt.get('konsensus')}, "
                + f"med={mdt.get('medgemma_manba')}, qwen={mdt.get('qwen_manba')})"
            )
        if holat.get("umumlashtirish"):
            qatorlar.append("Stepwise S: " + str(holat.get("umumlashtirish")))
        qatorlar.append(
            "Bu tizim shifokor o‘rnini bosmaydi. Yakuniy qaror klinik ko‘rik, "
            "to‘liq EKG, laboratoriya va tasvir asosida shifokorga tegishli."
        )
        tarix = list(holat.get("qadam_tarixi") or [])
        tarix.append("xulosa_tayyorlash: STOP")
        return {
            "xulosa": "\n".join(q for q in qatorlar if q),
            "amal": "STOP",
            "qadam_tarixi": tarix,
        }

    def _keyingi_amal_reja(self, holat: ChiefHolat) -> str:
        """Murakkablikdan keyin reja tuguniga o‘tadi.

        Args:
            holat: murakkablik maydoni.

        Returns:
            cardiac_rag_reja.
        """
        return "cardiac_rag_reja"

    def _keyingi_stepwise(self, holat: ChiefHolat) -> str:
        """CONTINUE → vosita; STOP+mdt → MDT; STOP → xulosa.

        Args:
            holat: amal va mdt_kerak.

        Returns:
            Tugun nomi.
        """
        if holat.get("amal") == "CONTINUE":
            return "qadam_bajarish"
        if holat.get("mdt_kerak"):
            return "mdt"
        return "xulosa_tayyorlash"

    def _graf_qur(self) -> Any:
        """6 bosqichli LangGraph: qabul → murakkablik → RAG reja → vosita sikli → MDT → xulosa.

        Returns:
            compile() qilingan graf.
        """
        graf = StateGraph(ChiefHolat)
        graf.add_node("qabul_qilish", self._qabul_qilish)
        graf.add_node("murakkablik_baholash", self._murakkablik_baholash)
        graf.add_node("cardiac_rag_reja", self._cardiac_rag_reja)
        graf.add_node("qadam_bajarish", self._qadam_bajarish)
        graf.add_node("stepwise_yangilash", self._stepwise_yangilash)
        graf.add_node("mdt", self._mdt)
        graf.add_node("xulosa_tayyorlash", self._xulosa_tayyorlash)

        graf.add_edge(START, "qabul_qilish")
        graf.add_edge("qabul_qilish", "murakkablik_baholash")
        graf.add_conditional_edges(
            "murakkablik_baholash",
            self._keyingi_amal_reja,
            {"cardiac_rag_reja": "cardiac_rag_reja"},
        )
        graf.add_edge("cardiac_rag_reja", "qadam_bajarish")
        graf.add_edge("qadam_bajarish", "stepwise_yangilash")
        graf.add_conditional_edges(
            "stepwise_yangilash",
            self._keyingi_stepwise,
            {
                "qadam_bajarish": "qadam_bajarish",
                "mdt": "mdt",
                "xulosa_tayyorlash": "xulosa_tayyorlash",
            },
        )
        graf.add_edge("mdt", "xulosa_tayyorlash")
        graf.add_edge("xulosa_tayyorlash", END)
        return graf.compile()

    def run(self, bemor: Dict[str, Any]) -> Dict[str, Any]:
        """Bemor ma’lumotini grafga yuboradi.

        Args:
            bemor: Klinik maydonlar va ixtiyoriy EKG/echo.

        Returns:
            xulosa, reja, vositalar, MDT. Tashxis emas.
        """
        boshlangich: ChiefHolat = {"bemor": bemor}
        return self.graf.invoke(boshlangich)


def chief_cardiologist_yarat(rag: Optional[MedicalRAG] = None) -> ChiefCardiologist:
    """ChiefCardiologist eksemplarini quradi.

    Args:
        rag: Ixtiyoriy MedicalRAG. None bo‘lsa seed disk keshdan yoki birinchi marta indekslanadi.

    Returns:
        Ishga tayyor agent.
    """
    if rag is None:
        try:
            rag = MedicalRAG()
            rag.urug_indeks()
        except Exception:
            rag = None
    return ChiefCardiologist(rag=rag)
