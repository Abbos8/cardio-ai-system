"""Kardiologik AI agent uchun 3 ustunli Streamlit interfeysi.

1-ustun: bemor va laboratoriya. 2-ustun: 12 tasmali EKG grafigi.
3-ustun: ChiefCardiologist xulosasi. Vizual panel shifokor so‘raganda ochiladi.
Tashxis o‘rnini bosmaydi.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import html


def _conda_numpy_aralashmasin() -> None:
    """conda base Streamlit/NumPy ni ushlab, tushunarli xato beradi.

    Returns:
        None. Noto‘g‘ri muhitda ImportError.
    """
    exe = (sys.executable or "").replace("\\", "/")
    venv_ichida = "/cardio-ai-system/venv/" in exe or exe.endswith("/cardio-ai-system/venv/bin/python")
    np_mod = sys.modules.get("numpy")
    np_fayl = (getattr(np_mod, "__file__", "") or "").replace("\\", "/")
    conda_numpy = "miniconda" in np_fayl and "/cardio-ai-system/venv/" not in np_fayl
    conda_python = "miniconda" in exe and "/cardio-ai-system/venv/" not in exe
    if conda_python or conda_numpy:
        raise ImportError(
            "numpy.core.multiarray: Streamlit conda (base) orqali ishga tushgan. "
            "Shu terminalni to‘xtating, keyin loyiha ildizida: conda deactivate && ./ishga_tushir.sh"
        )


_conda_numpy_aralashmasin()

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agents.chief_agent import ChiefCardiologist
from llm.client import llm_mavjud, model_qisqacha, vlm_sozlama
from rag.medical_rag import MedicalRAG
from tools.ecg_tool import TASMA_NOMLARI, namuna_12_tasma_ekg, tozalangan_matritsa, tasmalarni_tozala
from tools.ekg_yuklash import ekg_fayllardan_oqish
from tools.echo_view_model import ECHO_KORINISHLAR
from tools.echo_yuklash import echo_fayldan_kadrlar
from tools.lab_tool import process_lab
from xavfsizlik.audit import KLINIK_OGOHLANTIRISH, KLINIK_PRODUCTION

# Namuna EKG: kamida 2 s (ecg_tool talabi)
NAMUNA_SONIYA = 8.0
ODATIY_HZ = 500.0

# Laboratoriya: ko‘rsatish uchun taxminiy oraliqlar, tashxis chegarasi emas
LAB_MAYDONLARI: List[Tuple[str, str, str, float, float, float]] = [
    ("troponin_i", "Troponin I (ng/L)", "0–34 (laboratoriya usuliga bog‘liq)", 0.0, 500.0, 8.0),
    ("nt_probnp", "NT-proBNP (pg/mL)", "yosh/jinsga bog‘liq", 0.0, 20000.0, 80.0),
    ("kreatinin", "Kreatinin (µmol/L)", "taxminan 60–110", 20.0, 800.0, 78.0),
    ("kaliy", "Kaliy (mmol/L)", "taxminan 3.5–5.1", 2.0, 8.0, 4.2),
    ("glyukoza", "Glyukoza (mmol/L)", "och holatda taxminan 3.9–6.1", 1.0, 30.0, 5.4),
    ("ldl", "LDL (mmol/L)", "xavf guruhiga bog‘liq", 0.5, 10.0, 2.8),
    ("hdl", "HDL (mmol/L)", "taxminan >1.0", 0.2, 4.0, 1.2),
    ("hba1c", "HbA1c (%)", "taxminan <5.7", 3.5, 15.0, 5.5),
]


@st.cache_resource(show_spinner="CardiacRAG: BioClinicalBERT va FAISS yuklanmoqda...")
def _rag_ol(kesh_versiya: int = 4) -> Dict[str, Any]:
    """BERT ni jarayonda bir marta, FAISS ni disk keshdan yuklaydi.

    Args:
        kesh_versiya: Streamlit cache kaliti; atributlar o‘zgaganda oshiriladi.

    Returns:
        {ok, rag, xato}. BERT bo‘lmasa hashing; xatoda rag=None.
    """
    try:
        rag = MedicalRAG()
        rag.urug_indeks()
        return {"ok": True, "rag": rag, "xato": None}
    except Exception as exc:
        return {"ok": False, "rag": None, "xato": str(exc)}


def _rag_obyekt(paket: Any) -> Optional[MedicalRAG]:
    """Eski kesh (to‘g‘ridan-to‘g‘ri RAG) va yangi paketdan obyekt oladi.

    Args:
        paket: _rag_ol natijasi.

    Returns:
        MedicalRAG yoki None.
    """
    if isinstance(paket, dict):
        return paket.get("rag")
    return paket if paket is not None else None


def _rag_holat_matn(rag: Optional[MedicalRAG], xato: Optional[str] = None) -> str:
    """UI caption uchun RAG holatini yozadi (eski kesh obyektiga chidamli).

    Args:
        rag: MedicalRAG yoki None.
        xato: Yuklash xatosi.

    Returns:
        Qisqa holat matni. Tashxis emas.
    """
    if xato:
        return f"CardiacRAG xato: {xato}"
    if rag is None:
        return "CardiacRAG yuklanmadi (hashing/indeks yo‘q)."
    bert = bool(getattr(rag, "bert_ishlatildi", getattr(rag, "model", None) is not None))
    bolak = len(getattr(rag, "bolaklar", []) or [])
    qurilma = getattr(rag, "qurilma", "?")
    katalog = getattr(rag, "indeks_katalogi", "")
    if bert:
        return f"CardiacRAG: BioClinicalBERT + FAISS ({bolak} bo‘lak, {qurilma}). Indeks: {katalog}"
    sabab = getattr(rag, "bert_xato", None) or "USE_BIOCLINICAL_BERT=0 yoki hashing"
    return f"CardiacRAG: hashing zaxirasi ({bolak} bo‘lak). {sabab}"


def _rag_banner(paket: Any) -> None:
    """RAG/BERT muvaffaqiyat yoki xatoni ko‘rsatadi.

    Args:
        paket: _rag_ol natijasi.

    Returns:
        None. Tashxis emas.
    """
    if isinstance(paket, dict) and not paket.get("ok"):
        st.error(
            "CardiacRAG/BERT yuklanmadi: "
            + str(paket.get("xato") or "noma’lum")
            + ". Agent hashing/zaxira bilan davom etishi mumkin. Tashxis emas."
        )
        if st.button("RAG ni qayta yuklash"):
            _rag_ol.clear()
            st.rerun()
        return
    rag = _rag_obyekt(paket)
    matn = _rag_holat_matn(rag, (paket or {}).get("xato") if isinstance(paket, dict) else None)
    if rag is None:
        st.warning(matn)
    elif not getattr(rag, "bert_ishlatildi", True):
        st.warning(matn)
    else:
        st.caption(matn)


def _rejim_banner() -> None:
    """LLM/VLM yo‘qida shablon rejimini ochiq yozadi.

    Returns:
        None. Kalitni ko‘rsatmaydi.
    """
    if llm_mavjud():
        vlm = vlm_sozlama("medgemma") or vlm_sozlama("qwen_vl")
        if vlm:
            st.caption("LLM API ulangan. VLM sozlangan. Yakuniy qaror shifokorga tegishli.")
        else:
            st.caption(
                "LLM API ulangan. MedGemma/Qwen VLM yo‘q — MDT/fellow tasvir **shablon** bo‘lishi mumkin. "
                "Tashxis emas."
            )
        return
    st.warning(
        "**Shablon rejimida.** DeepSeek/VLM kaliti yo‘q — murakkablik, reja, fellow va MDT "
        "qoida/shablon bilan ishlaydi, agent to‘xtamaydi. Tashxis emas."
    )


def _klinik_ogohlantirish() -> None:
    """Har sahifada yuqori ogohlantirish (sticky).

    Returns:
        None. Tashxis o‘rnini bosmaydi.
    """
    st.markdown(
        """
<style>
.klinik-ogoh {
  position: sticky; top: 0; z-index: 999;
  background: #fff3cd; border: 1px solid #ffc107; color: #4a3800;
  padding: 0.42rem 0.8rem; border-radius: 6px; margin-bottom: 0.4rem;
  font-size: 0.87rem; line-height: 1.3;
}
.bolim-sarlavha {
  font-size: 0.93rem; font-weight: 600; color: #07111f;
  border-bottom: 1px solid #dde3ec; padding-bottom: 0.15rem;
  margin-top: 0.8rem; margin-bottom: 0.3rem;
}
.bosh-holat-ogoh {
  background: #eaf1fb; border: 1px dashed #9ab4d4; color: #1a3657;
  border-radius: 6px; padding: 1rem 1.2rem; text-align: center;
  font-size: 0.95rem; margin-top: 1rem;
}
.oqim-qator {
  display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.2rem 0 0.8rem;
}
.oqim-chip {
  font-size: 0.82rem; line-height: 1.2; white-space: nowrap;
  padding: 0.28rem 0.55rem; border-radius: 999px; border: 1px solid #d5dbe3;
  background: #f4f6f8; color: #3d4a5c;
}
.oqim-chip.done { background: #e5f6ea; border-color: #8dcea0; color: #145c2e; }
.oqim-chip.current { background: #e7f1fb; border-color: #7eb0e0; color: #0b3a66; font-weight: 600; }
.oqim-chip.skip { background: #f7f7f7; color: #6b7280; }
/* ── Yangi komponentlar ─────────────────────────────────────────────────── */
.lab-card {
  display:flex; align-items:center; gap:0.5rem; background:#f9fafb;
  border-radius:6px; padding:0.35rem 0.6rem; margin:0.18rem 0;
  font-size:0.84rem; border-left:4px solid #9ca3af;
}
.lab-card.ok     { border-left-color:#16a34a; }
.lab-card.warning{ border-left-color:#d97706; }
.lab-card.danger { border-left-color:#dc2626; }
.lab-card .lab-val { font-weight:700; font-size:0.94rem; margin-right:0.2rem; }
.lab-dot-ok      { color:#16a34a; }
.lab-dot-warning { color:#d97706; }
.lab-dot-danger  { color:#dc2626; }
.ecg-badge {
  display:inline-block; padding:0.22rem 0.55rem; border-radius:999px;
  font-size:0.82rem; font-weight:600; margin:0.15rem 0.2rem;
  background:#f4f6f8; color:#3d4a5c; border:1px solid #d5dbe3;
}
.ecg-badge.ok      { background:#dcfce7; color:#14532d; border-color:#86efac; }
.ecg-badge.warning { background:#fef3c7; color:#78350f; border-color:#fbbf24; }
.ecg-badge.danger  { background:#fee2e2; color:#7f1d1d; border-color:#f87171; }
.risk-badge {
  display:inline-block; padding:0.2rem 0.6rem; border-radius:999px;
  font-size:0.82rem; font-weight:700; margin:0.1rem 0.2rem;
}
.risk-badge.oddiy    { background:#dcfce7; color:#14532d; }
.risk-badge.murakkab { background:#ffedd5; color:#7c2d12; }
.risk-badge.unknown  { background:#f3f4f6; color:#6b7280; }
.agent-badge {
  display:inline-block; padding:0.2rem 0.6rem; border-radius:999px;
  font-size:0.82rem; font-weight:700; margin:0.1rem 0.2rem;
}
.agent-badge.stop     { background:#ccfbf1; color:#134e4a; }
.agent-badge.continue { background:#dbeafe; color:#1e3a8a; }
.mdt-gemma   { border-left:3px solid #7c3aed; background:#f5f3ff; border-radius:4px; padding:0.4rem 0.6rem; margin:0.2rem 0; }
.mdt-qwen    { border-left:3px solid #0891b2; background:#ecfeff; border-radius:4px; padding:0.4rem 0.6rem; margin:0.2rem 0; }
.mdt-deepseek{ border-left:3px solid #b91c1c; background:#fff1f2; border-radius:4px; padding:0.4rem 0.6rem; margin:0.2rem 0; }
.step-card {
  border:1px solid #e5e7eb; border-radius:6px;
  padding:0.5rem 0.75rem; margin:0.3rem 0; background:#fafafa;
}
.step-card.done    { border-left:4px solid #16a34a; }
.step-card.pending { border-left:4px solid #3b82f6; }
.step-card.xato    { border-left:4px solid #dc2626; }
.replan-alert {
  background:#fffbeb; border:1px solid #f59e0b; border-radius:6px;
  padding:0.5rem 0.8rem; margin:0.4rem 0; color:#78350f; font-size:0.87rem;
}
.log-entry {
  font-family:monospace; font-size:0.82rem; background:#f8fafc;
  border:1px solid #e2e8f0; border-radius:4px;
  padding:0.4rem 0.6rem; margin:0.2rem 0; white-space:pre-wrap;
  word-break:break-all; color:#1e293b;
}
</style>
""",
        unsafe_allow_html=True,
    )
    holat = "demo — klinik production emas" if not KLINIK_PRODUCTION else "production"
    st.markdown(
        f'<div class="klinik-ogoh">{html.escape(KLINIK_OGOHLANTIRISH)} ({holat})</div>',
        unsafe_allow_html=True,
    )


def _audit_paneli(agent_holat: Optional[Dict[str, Any]]) -> None:
    """Qaysi model va qaysi C ishlatilganini ko‘rsatadi.

    Args:
        agent_holat: run() natijasi.

    Returns:
        None. Kalit va PII yo‘q.
    """
    st.subheader("Audit (model va C)")
    model = model_qisqacha()
    st.caption(
        f"chat={model.get('chat')} | fellow_vision={model.get('fellow_vision')} | "
        f"MedGemma={model.get('medgemma')} | Qwen={model.get('qwen_vl')} | "
        f"LLM ulangan={model.get('llm_ulangan')}"
    )
    yozuv = (agent_holat or {}).get("audit") or {}
    if not yozuv:
        st.info("Tahlildan keyin shu yerga model va CardiacRAG C manbalari yoziladi.")
        return
    st.write("C manbalari: " + (", ".join(yozuv.get("C") or []) or "yo‘q"))
    st.write(
        f"fellow={yozuv.get('fellow_manba')}/{yozuv.get('fellow_model')} | "
        f"MDT MedGemma={yozuv.get('mdt_medgemma_manba')}/{yozuv.get('mdt_medgemma_model')} | "
        f"Qwen={yozuv.get('mdt_qwen_manba')}/{yozuv.get('mdt_qwen_model')}"
    )
    st.caption("Audit JSON mahalliy `data/audit/` da (git emas). Kalit va EKG signal yozilmaydi.")


def _sidebar_paneli(rag_paket: Any, agent_holat: Optional[Dict[str, Any]]) -> None:
    """Texnik holat (LLM/RAG), audit va formani tozalash tugmasini sidebar ga yig‘adi.

    Args:
        rag_paket: _rag_ol natijasi (ok, rag, xato).
        agent_holat: ChiefCardiologist.run natijasi yoki None.

    Returns:
        None. Tashxis emas; kalit va PII ko‘rsatilmaydi.
    """
    with st.sidebar:
        st.markdown("### 🫀 Tizim holati")

        # LLM/VLM qisqa holati
        if llm_mavjud():
            vlm = vlm_sozlama("medgemma") or vlm_sozlama("qwen_vl")
            if vlm:
                st.caption("✅ LLM ulangan · VLM sozlangan")
            else:
                st.caption("⚠️ LLM ulangan · VLM yo‘q (MDT shablon)")
        else:
            st.warning("Shablon rejimi: LLM kaliti yo‘q. MDT/fellow qoida bilan ishlaydi.", icon="⚠️")

        # RAG holati (yig‘ilgan)
        with st.expander("CardiacRAG holati", expanded=False):
            _rag_banner(rag_paket)

        st.divider()

        # Audit
        _audit_paneli(agent_holat)

        st.divider()

        # Agent izlar (qisqa versiya)
        if agent_holat:
            with st.expander("Agent izlar (qisqa)", expanded=False):
                _agent_izlar(agent_holat)

        st.divider()

        # Formani tozalash: tahlil natijalarini o‘chiradi, widget state saqlanadi
        st.caption("Tahlil natijalarini o‘chiring (yuklangan fayllar saqlanadi).")
        if st.button("🗑️ Formani tozalash", use_container_width=True, key="formani_tozala"):
            st.session_state.agent_holat = None
            st.session_state.bemor_tahlil = None
            st.session_state.vizual_ochiq = False
            st.rerun()


def _oqim_bosqichlari(holat: Optional[Dict[str, Any]]) -> List[str]:
    """6 bosqich holatini tarixdan chiqaradi.

    Args:
        holat: Agent run natijasi.

    Returns:
        pending|current|done|skip ro‘yxati (6 ta).
    """
    nomlar = ["pending"] * 6
    if not holat:
        nomlar[0] = "current"
        return nomlar
    tarix = [str(t) for t in (holat.get("qadam_tarixi") or [])]
    belgilar = [
        any(t.startswith("qabul_qilish") for t in tarix),
        any(t.startswith("murakkablik") for t in tarix),
        any("cardiac_rag_reja" in t for t in tarix),
        any(t.startswith("qadam_bajarish") or t.startswith("stepwise") for t in tarix),
        bool(holat.get("mdt_natija")) or any(t.startswith("mdt:") for t in tarix),
        any("xulosa_tayyorlash" in t for t in tarix) or bool(holat.get("xulosa")),
    ]
    if belgilar[5] and not belgilar[4]:
        belgilar[4] = False  # skip later
    last = -1
    for i, b in enumerate(belgilar):
        if b:
            nomlar[i] = "done"
            last = i
    if belgilar[5] and not holat.get("mdt_natija"):
        nomlar[4] = "skip"
    elif last >= 0 and last < 5 and not belgilar[5]:
        nomlar[last + 1] = "current"
    if belgilar[5]:
        for i in range(6):
            if nomlar[i] == "current":
                nomlar[i] = "done"
            if nomlar[i] == "pending" and i != 4:
                nomlar[i] = "done"
    return nomlar


def _olti_bosqich_chiz(holat: Optional[Dict[str, Any]]) -> None:
    """LangGraph 6 bosqichini progress qatori sifatida chizadi.

    Args:
        holat: Joriy yoki yakuniy graf holati.

    Returns:
        None. Tashxis emas.
    """
    sarlavhalar = [
        "1 · Qabul",
        "2 · Murakkablik",
        "3 · RAG reja",
        "4 · Vositalar",
        "5 · MDT",
        "6 · Xulosa",
    ]
    holatlar = _oqim_bosqichlari(holat)
    st.markdown("**6 bosqichli oqim**")
    qismlar = []
    for nom, s in zip(sarlavhalar, holatlar):
        yozuv = nom + (" · o‘tkazildi" if s == "skip" else "")
        if s == "current":
            yozuv = "▶ " + yozuv
        qismlar.append(
            f'<span class="oqim-chip {html.escape(s)}">{html.escape(yozuv)}</span>'
        )
    st.markdown(
        '<div class="oqim-qator">' + "".join(qismlar) + "</div>",
        unsafe_allow_html=True,
    )


def _reja_jonli(holat: Optional[Dict[str, Any]]) -> None:
    """Klinik reja P ni qadam holati bilan ko‘rsatadi.

    Args:
        holat: reja, reja_indeks.

    Returns:
        None.
    """
    reja = (holat or {}).get("reja") or []
    indeks = int((holat or {}).get("reja_indeks") or 0)
    st.markdown("**Klinik reja P (jonli)**")
    if not reja:
        st.caption("Reja hali tuzilmagan.")
        return
    for i, qadam in enumerate(reja):
        hol = str(qadam.get("holat") or "pending")
        if hol == "done":
            belgi = "✓"
        elif hol == "xato":
            belgi = "✗"
        elif i == indeks:
            belgi = "▶"
        else:
            belgi = "○"
        st.write(
            f"{belgi} {qadam.get('id')} [{hol}] {qadam.get('vosita')}: {qadam.get('tavsif')}"
        )


def _agent_ol() -> ChiefCardiologist:
    """LangGraph bosh kardiologini CardiacRAG bilan quradi.

    Returns:
        6 bosqichli workflow agenti.
    """
    return ChiefCardiologist(rag=_rag_obyekt(_rag_ol()))


def namuna_ekg_signal(sampling_rate: float = ODATIY_HZ, davomiylik: float = NAMUNA_SONIYA) -> np.ndarray:
    """O‘qitish uchun 12 tasmali sintetik EKG (n, 12) yasaydi.

    Args:
        sampling_rate: Namuna olish tezligi (Hz).
        davomiylik: Yozuv uzunligi (soniya).

    Returns:
        (n, 12) massiv. Haqiqiy bemor EKG si emas.
    """
    return namuna_12_tasma_ekg(sampling_rate=sampling_rate, davomiylik=davomiylik)


def csv_dan_ekg(fayl_bayt: bytes, sampling_rate_hint: float = ODATIY_HZ) -> Tuple[Optional[np.ndarray], str, float]:
    """CSV fayldan 12 tasmali EKG o‘qiydi (sarlavha I..V6 yoki 12 ustun).

    Args:
        fayl_bayt: Yuklangan CSV baytlari.
        sampling_rate_hint: Hz izohi yo‘q bo‘lsa.

    Returns:
        (massiv, xabar, hz). Xatoda massiv None. Tashxis qo‘yilmaydi.
    """
    natija = ekg_fayllardan_oqish([("ekg.csv", fayl_bayt)], sampling_rate_hint=sampling_rate_hint)
    hz = float(natija.get("sampling_rate") or sampling_rate_hint)
    return natija.get("signal"), str(natija.get("xabar") or ""), hz


def ekg_grafik_chiz(
    signal: np.ndarray,
    sampling_rate: float,
    tasma_tanlov: str,
    tolqinlar: Optional[Dict[str, Any]] = None,
    ep_tasma: Optional[str] = None,
    sarlavha: str = "EKG (ko‘rish, tashxis emas)",
) -> plt.Figure:
    """EKG ni Streamlit uchun matplotlib figuraga chizadi (ixtiyoriy P/QRS/T).

    Args:
        signal: (n, 12) yoki (12, n) massiv.
        sampling_rate: Hz.
        tasma_tanlov: Bitta tasma nomi yoki \"12 tasma\".
        tolqinlar: EP indekslari (r/p/t cho‘qqilar).
        ep_tasma: Intervallar olingan tasma.
        sarlavha: Figura sarlavhasi.

    Returns:
        Figura. O‘lchov emas, faqat ko‘rish.
    """
    massiv = np.asarray(signal, dtype=float)
    if massiv.ndim != 2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "EKG shakli noto‘g‘ri", ha="center")
        return fig
    if massiv.shape[0] == 12 and massiv.shape[1] != 12:
        massiv = massiv.T
    elif massiv.shape[1] != 12:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "12 tasma kutilgan", ha="center")
        return fig

    n = massiv.shape[0]
    t = np.arange(n) / float(sampling_rate)
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    def _belgi(ax: Any, y: np.ndarray) -> None:
        if not tolqinlar:
            return
        hz = float(sampling_rate)

        def nuqta(indekslar: List[int], rang: str, nom: str) -> None:
            if not indekslar:
                return
            idx = [i for i in indekslar if 0 <= i < len(y)]
            if not idx:
                return
            ax.scatter(
                np.asarray(idx) / hz,
                y[idx],
                s=18,
                c=rang,
                label=nom,
                zorder=3,
            )

        nuqta(list(tolqinlar.get("p_choqqilar") or []), "#1565c0", "P")
        nuqta(list(tolqinlar.get("r_choqqilar") or []), "#c62828", "R")
        nuqta(list(tolqinlar.get("t_choqqilar") or []), "#2e7d32", "T")

    if tasma_tanlov == "12 tasma":
        fig, oqlar = plt.subplots(6, 2, figsize=(8.2, 9.2), sharex=True)
        for i, nom in enumerate(TASMA_NOMLARI):
            ax = oqlar[i // 2][i % 2]
            ax.plot(t, massiv[:, i], color="#b71c1c", linewidth=0.7)
            if ep_tasma == nom or (ep_tasma is None and nom == "II"):
                _belgi(ax, massiv[:, i])
            ax.set_ylabel(nom, rotation=0, labelpad=18, va="center", fontsize=9)
            ax.set_yticks([])
        oqlar[-1][0].set_xlabel("Vaqt (s)")
        oqlar[-1][1].set_xlabel("Vaqt (s)")
        fig.suptitle(sarlavha, fontsize=11)
        fig.tight_layout()
        return fig

    idx = TASMA_NOMLARI.index(tasma_tanlov) if tasma_tanlov in TASMA_NOMLARI else 1
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.plot(t, massiv[:, idx], color="#b71c1c", linewidth=0.9)
    _belgi(ax, massiv[:, idx])
    if tolqinlar and (tolqinlar.get("p_choqqilar") or tolqinlar.get("r_choqqilar")):
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Vaqt (s)")
    ax.set_ylabel("Amplitude (shartli)")
    ax.set_title(f"{sarlavha} — {tasma_tanlov}")
    fig.tight_layout()
    return fig


def _kadr_rgb(kadr: Any) -> np.ndarray:
    """Kulrang echo kaderni Streamlit image uchun RGB qiladi.

    Args:
        kadr: 2D yoki 3D massiv.

    Returns:
        uint8 RGB. Tashxis emas.
    """
    arr = np.asarray(kadr)
    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            a_min, a_max = float(arr.min()), float(arr.max())
            if a_max > a_min:
                arr = (255.0 * (arr - a_min) / (a_max - a_min)).clip(0, 255).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)
        return np.stack([arr, arr, arr], axis=-1)
    return arr


def _tozalangan_ekg(signal: Optional[np.ndarray], sampling_rate: float) -> Tuple[Optional[np.ndarray], str]:
    """Technician filtrini qo‘llab (n, 12) chizma massivini beradi.

    Args:
        signal: Xom EKG.
        sampling_rate: Hz.

    Returns:
        Massiv va sarlavha. Tozalash ishlamasa xom signal.
    """
    if signal is None:
        return None, "EKG yo‘q."
    massiv = np.asarray(signal, dtype=float)
    if massiv.ndim == 2 and massiv.shape[0] == 12 and massiv.shape[1] != 12:
        massiv = massiv.T
    if massiv.ndim != 2 or massiv.shape[1] != 12:
        return massiv, "Xom EKG (12 tasma emas, tashxis emas)"
    try:
        tasmalar = {nom: massiv[:, i] for i, nom in enumerate(TASMA_NOMLARI)}
        toz = tasmalarni_tozala(tasmalar, sampling_rate)["tozalangan"]
        mat = tozalangan_matritsa(toz)
        if mat is not None:
            return mat, "Tozalangan 12 tasma (ko‘rish, tashxis emas)"
    except Exception:
        pass
    return massiv, "Xom EKG (tozalash ishlamadi, tashxis emas)"


def _echo_kadr_xarita(bemor: Optional[Dict[str, Any]], echo: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Yuklangan echo fayllarni 11 ko‘rinish yorlig‘iga bog‘laydi.

    Args:
        bemor: echo_fayllar.
        echo: classify_echo_views qisqasi.

    Returns:
        yorliq → [{nom, kadrlar}]. Tashxis emas.
    """
    xarita: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ECHO_KORINISHLAR}
    xarita["ANIQLANMAGAN"] = []
    yozuv_yorliq = {z.get("fayl"): z for z in (echo.get("yozuvlar") or [])}
    for element in (bemor or {}).get("echo_fayllar") or []:
        nom = element.get("nom")
        bayt = element.get("bayt")
        if not bayt:
            continue
        oq = echo_fayldan_kadrlar(str(nom), bytes(bayt))
        if not oq.get("ok") or not oq.get("kadrlar"):
            continue
        yor = str((yozuv_yorliq.get(nom) or {}).get("asosiy") or "ANIQLANMAGAN")
        if yor not in xarita:
            yor = "ANIQLANMAGAN"
        xarita[yor].append(
            {
                "nom": nom,
                "kadrlar": oq["kadrlar"],
                "manba": (yozuv_yorliq.get(nom) or {}).get("manba"),
                "ehtimol": (yozuv_yorliq.get(nom) or {}).get("ehtimol"),
            }
        )
    return xarita


def _lab_kartalar(lab_natija: Optional[Dict[str, Any]], bemor: Optional[Dict[str, Any]] = None) -> None:
    """Lab analitlarini rangli kartochkalar bilan ko'rsatadi.

    Args:
        lab_natija: agent_holat['lab_natija'] — {qiymatlar, ...}.
        bemor: Ixtiyoriy; zaxira laboratoriya lug'ati.

    Returns:
        None. Orientir rangi; tashxis emas.
    """
    qiymatlar: Dict[str, float] = (lab_natija or {}).get("qiymatlar") or {}
    if not qiymatlar:
        qiymatlar = (bemor or {}).get("laboratoriya") or {}
    if not qiymatlar:
        return

    def _holat(k: str, v: float) -> str:
        """Bir analitning xavf darajasini qaytaradi (ok/warning/danger)."""
        if k == "troponin_i":
            return "ok" if v < 34 else ("warning" if v < 100 else "danger")
        if k == "nt_probnp":
            return "ok" if v < 300 else ("warning" if v < 900 else "danger")
        if k == "kreatinin":
            return "danger" if v < 30 or v > 300 else ("ok" if 60 <= v <= 110 else "warning")
        if k == "kaliy":
            return "danger" if v < 3.0 or v > 5.5 else ("ok" if 3.5 <= v <= 5.1 else "warning")
        if k == "glyukoza":
            return "ok" if v <= 6.1 else ("warning" if v <= 11 else "danger")
        if k == "ldl":
            return "ok" if v < 2.6 else ("warning" if v < 4.9 else "danger")
        if k == "hdl":
            return "danger" if v < 0.7 else ("warning" if v <= 1.0 else "ok")
        if k == "hba1c":
            return "ok" if v < 5.7 else ("warning" if v < 6.5 else "danger")
        return "ok"

    belgi_map = {"ok": "●", "warning": "⚠", "danger": "🔴"}
    sarlavha_map = {m[0]: m[1] for m in LAB_MAYDONLARI}
    html_qismlar = []
    for k, v in qiymatlar.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        h = _holat(k, fv)
        s = html.escape(sarlavha_map.get(k, k))
        belgi = belgi_map.get(h, "●")
        html_qismlar.append(
            f'<div class="lab-card {html.escape(h)}">'
            f'<span class="lab-dot-{html.escape(h)}">{belgi}</span>'
            f'<span class="lab-val">{html.escape(str(round(fv, 2)))}</span>'
            f'<span>{s}</span>'
            f'</div>'
        )
    if html_qismlar:
        st.markdown("".join(html_qismlar), unsafe_allow_html=True)
        st.caption("Orientir. Tashxis emas.")


def _ecg_metrik(ecg_natija: Optional[Dict[str, Any]], ep_natija: Optional[Dict[str, Any]] = None) -> None:
    """HR/QRS/QTc/PR ni rangli pill badge sifatida ko'rsatadi.

    Args:
        ecg_natija: agent_holat['ecg_natija'] — ECG o'lchovlar.
        ep_natija: agent_holat['ep_natija'] — zaxira EP o'lchovlar.

    Returns:
        None. Taxminiy o'lchov; tashxis emas.
    """
    ecg = ecg_natija or {}
    ep = ep_natija or {}

    def _val(kalit: str) -> Optional[float]:
        """Birinchi mavjud qiymatni float qilib qaytaradi."""
        v = ecg.get(kalit) or ep.get(kalit)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    hr = _val("yurak_chastotasi_bpm")
    qrs = _val("qrs_ms")
    qtc = _val("qtc_bazett_ms")
    pr = _val("pr_ms")

    badge_qismlar = []
    if hr is not None:
        h = "danger" if (hr < 50 or hr > 100) else "ok"
        badge_qismlar.append(f'<span class="ecg-badge {html.escape(h)}">HR {int(round(hr))} bpm</span>')
    if qrs is not None:
        h = "warning" if qrs >= 120 else "ok"
        badge_qismlar.append(f'<span class="ecg-badge {html.escape(h)}">QRS {int(round(qrs))} ms</span>')
    if qtc is not None:
        h = "warning" if qtc >= 450 else "ok"
        badge_qismlar.append(f'<span class="ecg-badge {html.escape(h)}">QTc {int(round(qtc))} ms</span>')
    if pr is not None:
        h = "warning" if pr >= 200 else "ok"
        badge_qismlar.append(f'<span class="ecg-badge {html.escape(h)}">PR {int(round(pr))} ms</span>')

    if badge_qismlar:
        st.markdown(
            '<div style="margin:0.3rem 0;">' + "".join(badge_qismlar) + "</div>",
            unsafe_allow_html=True,
        )
        st.caption("Taxminiy o'lchov, tashxis emas.")


def _qadam_panel(agent_holat: Optional[Dict[str, Any]]) -> None:
    """Klinik reja P ni qadam holati va oraliq natija xilosasi bilan ko'rsatadi.

    Args:
        agent_holat: ChiefCardiologist.run natijasi yoki None.

    Returns:
        None. Bu panel klinik reja P ni aks ettiradi; tashxis emas.
    """
    if not agent_holat:
        return
    murakkablik = str(agent_holat.get("murakkablik") or "")
    amal = str(agent_holat.get("amal") or "")
    sababi = str(agent_holat.get("murakkablik_sababi") or "")[:80]
    risk_cls = "oddiy" if murakkablik == "oddiy" else ("murakkab" if murakkablik == "murakkab" else "unknown")
    amal_cls = "stop" if amal == "STOP" else "continue"

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.3rem;">'
        f'<b>Qadam-ba-qadam bajarish paneli</b>'
        f'<span class="risk-badge {html.escape(risk_cls)}">{html.escape(murakkablik or "—")}</span>'
        f'<span class="agent-badge {html.escape(amal_cls)}">{html.escape(amal or "—")}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if sababi:
        st.caption(sababi)

    # REPLAN detection: check qadam_tarixi for replan keyword
    tarix = [str(t) for t in (agent_holat.get("qadam_tarixi") or [])]
    if any("replan" in t.lower() for t in tarix):
        st.markdown(
            '<div class="replan-alert">⚠️ REPLAN aniqlandi — reja qayta tartibga solindi. '
            'Klinik qarorni shifokor tasdiqlashi kerak.</div>',
            unsafe_allow_html=True,
        )

    reja = (agent_holat or {}).get("reja") or []
    if not reja:
        st.caption("Reja hali tuzilmagan.")

    lab_n = agent_holat.get("lab_natija") or {}
    ecg_tech = agent_holat.get("ecg_technician_natija") or {}
    ep_n = agent_holat.get("ep_natija") or {}
    fellow_n = agent_holat.get("fellow_natija") or {}

    for qadam in reja:
        hol = str(qadam.get("holat") or "pending")
        vosita = str(qadam.get("vosita") or "")
        tavsif = str(qadam.get("tavsif") or "")[:120]
        qid = str(qadam.get("id") or "")

        if hol == "done":
            belgi, klass = "✅", "done"
        elif hol == "xato":
            belgi, klass = "❌", "xato"
        else:
            belgi, klass = "🔷", "pending"

        natija_matn = ""
        if hol == "done":
            if "lab_technician" in vosita:
                tokenlar = lab_n.get("tokenlar") or []
                natija_matn = f"Tokenlar: {len(tokenlar)} | Top: {', '.join(tokenlar[:3])}"
            elif "ecg_technician" in vosita:
                natija_matn = (
                    f"Sifat: {ecg_tech.get('sifat') or '—'} | "
                    f"HR≈{ecg_tech.get('yurak_chastotasi_bpm') or '—'} | "
                    f"{ecg_tech.get('davomiylik_s') or '—'} s"
                )
            elif "electrophysiologist" in vosita or "ep_tool" in vosita:
                natija_matn = (
                    f"PR {ep_n.get('pr_ms') or '—'} | "
                    f"QRS {ep_n.get('qrs_ms') or '—'} | "
                    f"QTc {ep_n.get('qtc_bazett_ms') or '—'}"
                )
            elif "fellow" in vosita.lower():
                matn = str(fellow_n.get("matn") or "")[:80]
                natija_matn = matn if matn else "Bajarildi"
            else:
                natija_matn = f"{vosita} bajarildi"

        xato_matn = ""
        if hol == "xato":
            xato_matn = str(qadam.get("xabar") or "Xato yuz berdi")[:100]

        st.markdown(
            f'<div class="step-card {html.escape(klass)}">'
            f'{belgi} <b>{html.escape(qid)}</b> — <b>{html.escape(vosita)}</b><br>'
            f'<span style="color:#6b7280;font-size:0.85rem;">{html.escape(tavsif)}</span>'
            + (f'<br><span style="color:#374151;font-size:0.83rem;">{html.escape(natija_matn)}</span>' if natija_matn else "")
            + (f'<br><span style="color:#dc2626;font-size:0.83rem;">❌ {html.escape(xato_matn)}</span>' if xato_matn else "")
            + '</div>',
            unsafe_allow_html=True,
        )

    baho = str(agent_holat.get("baho_sababi") or "")
    if baho:
        st.caption(f"Baholash sababi: {baho}")
    st.caption("Bu panel klinik reja P ni aks ettiradi. Tashxis emas.")


def _agent_izlar(agent_holat: Optional[Dict[str, Any]]) -> None:
    """Agent mulohaza va iz jurnalin to'rt tabli ko'rinishda chiqaradi.

    Args:
        agent_holat: ChiefCardiologist.run natijasi yoki None.

    Returns:
        None. Tizim izlari; tashxis emas.
    """
    if not agent_holat:
        st.info("Tahlildan keyin izlar ko'rinadi.")
        return

    import json as _json

    tab_rag, tab_boss, tab_ekspert, tab_tizim = st.tabs([
        "CardiacRAG izlar",
        "Bosh kardiolog mulohazasi",
        "Ekspert vositalar",
        "Tizim holati",
    ])

    with tab_rag:
        dalillar = agent_holat.get("rag_dalillar") or []
        if not dalillar:
            st.info("RAG dalillari yo'q.")
        else:
            for i, d in enumerate(dalillar):
                st.markdown(
                    f'<div class="log-entry">[{i + 1}] {html.escape(str(d)[:300])}</div>',
                    unsafe_allow_html=True,
                )
        audit = agent_holat.get("audit") or {}
        c_manbalar = audit.get("C") or []
        if c_manbalar:
            st.caption("C manbalari: " + ", ".join(str(m) for m in c_manbalar))

    with tab_boss:
        murakkablik = str(agent_holat.get("murakkablik") or "—")
        sababi = str(agent_holat.get("murakkablik_sababi") or "")
        risk_cls = "oddiy" if murakkablik == "oddiy" else ("murakkab" if murakkablik == "murakkab" else "unknown")
        keng_nom = (
            "BASIC (oddiy)" if murakkablik == "oddiy"
            else ("COMPLEX (murakkab)" if murakkablik == "murakkab" else murakkablik)
        )
        st.markdown(
            f'<span class="risk-badge {html.escape(risk_cls)}">{html.escape(keng_nom)}</span>',
            unsafe_allow_html=True,
        )
        if sababi:
            st.caption(sababi)
        tarix = agent_holat.get("qadam_tarixi") or []
        st.markdown("**Qadam tarixi (chain-of-thought):**")
        for i, q in enumerate(tarix):
            st.markdown(
                f'<div class="log-entry">[{i + 1}] {html.escape(str(q))}</div>',
                unsafe_allow_html=True,
            )
        baho = str(agent_holat.get("baho_sababi") or "")
        if baho:
            st.markdown(
                f'<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:6px;'
                f'padding:0.5rem 0.8rem;margin-top:0.5rem;">'
                f'<b>CONTINUE/STOP sababi:</b> {html.escape(baho)}</div>',
                unsafe_allow_html=True,
            )

    with tab_ekspert:
        ecg_tech = agent_holat.get("ecg_technician_natija") or {}
        ep_n2 = agent_holat.get("ep_natija") or {}
        echo_n2 = agent_holat.get("echo_natija") or {}
        mask_n2 = agent_holat.get("echo_mask") or {}
        fellow_n2 = agent_holat.get("fellow_natija") or {}

        with st.expander("EKG technician", expanded=True):
            if ecg_tech:
                for k in ["sifat", "tasma", "urishlar_soni", "yurak_chastotasi_bpm",
                          "davomiylik_s", "sampling_rate"]:
                    v = ecg_tech.get(k)
                    if v is not None:
                        st.markdown(
                            f'<div class="log-entry">{html.escape(k)}: {html.escape(str(v))}</div>',
                            unsafe_allow_html=True,
                        )
                yet = ecg_tech.get("yetishmagan") or []
                if yet:
                    st.markdown(
                        f'<div class="log-entry">yetishmagan: {html.escape(", ".join(yet))}</div>',
                        unsafe_allow_html=True,
                    )
                if ecg_tech.get("xabar"):
                    st.caption(str(ecg_tech["xabar"]))
            else:
                st.info("EKG technician natijasi yo'q.")

        with st.expander("Elektrofiziolog", expanded=False):
            if ep_n2:
                for k in ["tasma", "yurak_chastotasi_bpm", "pr_ms", "qrs_ms", "qt_ms",
                          "qtc_bazett_ms", "hrv_sdnn_ms", "hrv_rmssd_ms", "p_soni", "t_soni"]:
                    v = ep_n2.get(k)
                    if v is not None:
                        st.markdown(
                            f'<div class="log-entry">{html.escape(k)}: {html.escape(str(v))}</div>',
                            unsafe_allow_html=True,
                        )
                eslatmalar = ep_n2.get("eslatmalar") or []
                if eslatmalar:
                    st.markdown("**Eslatmalar:**")
                    for e in eslatmalar:
                        st.markdown(f"- {e}")
                if ep_n2.get("xabar"):
                    st.caption(str(ep_n2["xabar"]))
            else:
                st.info("Elektrofiziolog natijasi yo'q.")

        with st.expander("Echo technician", expanded=False):
            if echo_n2:
                korinishlar = echo_n2.get("korinishlar") or []
                if korinishlar:
                    st.markdown(
                        f'<div class="log-entry">korinishlar: {html.escape(", ".join(korinishlar))}</div>',
                        unsafe_allow_html=True,
                    )
                yet = echo_n2.get("yetishmagan_standart") or []
                if yet:
                    st.markdown(
                        f'<div class="log-entry">yetishmagan_standart: {html.escape(", ".join(yet))}</div>',
                        unsafe_allow_html=True,
                    )
                if echo_n2.get("xabar"):
                    st.caption(str(echo_n2["xabar"]))
            else:
                st.info("Echo natijasi yo'q.")

        with st.expander("Echo segmenter (LV maska)", expanded=False):
            if mask_n2:
                for k in ["korinish", "maydon_px", "ulush", "model", "xabar"]:
                    v = mask_n2.get(k)
                    if v is not None:
                        st.markdown(
                            f'<div class="log-entry">{html.escape(k)}: {html.escape(str(v))}</div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.info("LV maska natijasi yo'q.")

        with st.expander("Cardiology fellow", expanded=False):
            if fellow_n2:
                st.markdown(
                    f'<div class="log-entry">'
                    f'manba: {html.escape(str(fellow_n2.get("manba") or ""))} | '
                    f'model: {html.escape(str(fellow_n2.get("model") or ""))}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                dalillar2 = fellow_n2.get("dalillar") or []
                if dalillar2:
                    st.markdown("**Dalillar:**")
                    for d in dalillar2:
                        st.markdown(f"- {d}")
                yoq = fellow_n2.get("yoq_dalillar") or []
                if yoq:
                    st.markdown("**Yo'q dalillar:**")
                    for d in yoq:
                        st.markdown(f"- {d}")
                matn2 = str(fellow_n2.get("matn") or "")
                if matn2:
                    with st.expander("To'liq matn", expanded=False):
                        st.markdown(
                            f'<div class="log-entry">{html.escape(matn2)}</div>',
                            unsafe_allow_html=True,
                        )
                if fellow_n2.get("xabar"):
                    st.caption(str(fellow_n2["xabar"]) + " Tashxis emas.")
            else:
                st.info("Fellow natijasi yo'q.")

    with tab_tizim:
        tarix2 = agent_holat.get("qadam_tarixi") or []
        st.markdown(f"**Qadam soni:** {len(tarix2)} | **Amal:** {agent_holat.get('amal') or '—'}")
        for i, q in enumerate(tarix2):
            st.markdown(
                f'<div class="log-entry">[{i}] {html.escape(str(q))}</div>',
                unsafe_allow_html=True,
            )
        ep_n3 = agent_holat.get("ep_natija") or {}
        eslatmalar3 = ep_n3.get("eslatmalar") or []
        ecg_tech3 = agent_holat.get("ecg_technician_natija") or {}
        yet3 = ecg_tech3.get("yetishmagan") or []
        if eslatmalar3 or yet3:
            st.markdown("**Ogohlantirishlar:**")
            for e in eslatmalar3:
                st.warning(e)
            if yet3:
                st.warning("EKG yetishmagan tasma: " + ", ".join(yet3))
        audit2 = agent_holat.get("audit") or {}
        if audit2:
            st.markdown("**Audit:**")
            st.markdown(
                f'<div class="log-entry">{html.escape(_json.dumps(audit2, ensure_ascii=False, indent=2))}</div>',
                unsafe_allow_html=True,
            )


def _mdt_panel(mdt: Dict[str, Any], agent_holat: Optional[Dict[str, Any]] = None) -> None:
    """MDT muhokara raundlarini uch ekspert ustunida vizual ko'rsatadi.

    Args:
        mdt: mdt_munozara natijasi — raundlar, konsensus, umumlashtirish.
        agent_holat: I/Z zaxira; ixtiyoriy.

    Returns:
        None. MDT muhokara T=4 raund. Tashxis emas.
    """
    holat = agent_holat or {}
    raund_soni = int(mdt.get("raund_soni") or 0)
    konsensus = mdt.get("konsensus")
    k_rang = "#16a34a" if konsensus else "#dc2626"
    k_matn = "✅ KELISHILDI" if konsensus else "❌ KELISHMADI"

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.8rem;flex-wrap:wrap;margin-bottom:0.4rem;">'
        f'<b>MDT muhokara</b>'
        f'<span style="background:#f3f4f6;border-radius:4px;padding:0.15rem 0.5rem;'
        f'font-size:0.85rem;">Raund {raund_soni}</span>'
        f'<span style="color:{k_rang};font-weight:700;">{k_matn}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"MedGemma={mdt.get('medgemma_manba')} | Qwen={mdt.get('qwen_manba')} | "
        f"DeepSeek={mdt.get('deepseek_manba')}"
    )

    for r in mdt.get("raundlar") or []:
        r_soni = int(r.get("raund") or 0)
        st.markdown(f"**Raund {r_soni}** — kirish: {r.get('kirish') or '—'}")
        col_g, col_q, col_d = st.columns(3)

        with col_g:
            manba_g = str(r.get("medgemma_manba") or mdt.get("medgemma_manba") or "")
            st.markdown(
                f'<div class="mdt-gemma"><b>MedGemma</b><br>'
                f'<span style="font-size:0.77rem;color:#6b7280;">{html.escape(manba_g)}</span></div>',
                unsafe_allow_html=True,
            )
            matn_g = str(r.get("medgemma") or "")
            if len(matn_g) > 300:
                st.write(matn_g[:300] + "…")
                with st.expander("Ko'proq (MedGemma)"):
                    st.write(matn_g)
            else:
                st.write(matn_g)

        with col_q:
            manba_q = str(r.get("qwen_manba") or mdt.get("qwen_manba") or "")
            st.markdown(
                f'<div class="mdt-qwen"><b>Qwen2.5-VL</b><br>'
                f'<span style="font-size:0.77rem;color:#6b7280;">{html.escape(manba_q)}</span></div>',
                unsafe_allow_html=True,
            )
            matn_q = str(r.get("qwen") or "")
            if len(matn_q) > 300:
                st.write(matn_q[:300] + "…")
                with st.expander("Ko'proq (Qwen)"):
                    st.write(matn_q)
            else:
                st.write(matn_q)

        with col_d:
            manba_d = str(r.get("deepseek_manba") or mdt.get("deepseek_manba") or "")
            st.markdown(
                f'<div class="mdt-deepseek"><b>DeepSeek-R1</b><br>'
                f'<span style="font-size:0.77rem;color:#6b7280;">{html.escape(manba_d)}</span></div>',
                unsafe_allow_html=True,
            )
            matn_d = str(r.get("deepseek") or "")
            if len(matn_d) > 300:
                st.write(matn_d[:300] + "…")
                with st.expander("Ko'proq (DeepSeek)"):
                    st.write(matn_d)
            else:
                st.write(matn_d)

        if r_soni >= 3:
            q_agree = r.get("qwen_agree")
            g_agree = r.get("medgemma_agree")
            both = bool(q_agree) and bool(g_agree)
            rang2 = "#16a34a" if both else "#dc2626"
            k2 = "KELISHILDI" if both else "KELISHMADI"
            st.markdown(
                f'<div style="margin:0.2rem 0;">'
                f'agree(Q,D_oldingi)=<b>{q_agree}</b> | '
                f'agree(P,D_oldingi)=<b>{g_agree}</b> | '
                f'<span style="color:{rang2};font-weight:700;">{k2}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.divider()

    umumlashtirish = mdt.get("umumlashtirish")
    if umumlashtirish:
        st.markdown(
            f'<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:6px;'
            f'padding:0.6rem 0.8rem;margin:0.5rem 0;">'
            f'<b>Yakuniy D (DeepSeek-R1) xulosasi:</b><br>{html.escape(str(umumlashtirish))}'
            f'</div>',
            unsafe_allow_html=True,
        )
    if mdt.get("konsensus_sabab"):
        st.caption("Konsensus: " + str(mdt.get("konsensus_sabab")))

    with st.expander("I va Z (qayta kiritilgan)"):
        st.markdown("**I (xom)**")
        st.write(mdt.get("xom_i") or holat.get("xom_i") or "")
        st.markdown("**Z (oraliq)**")
        st.write(mdt.get("oraliq_z") or holat.get("oraliq_z") or "")

    st.caption("MDT muhokara T=4 raund. Tashxis emas.")


def _mdt_yonma_yon(mdt: Dict[str, Any], agent_holat: Optional[Dict[str, Any]] = None) -> None:
    """MDT raundlarini MedGemma | Qwen yonma-yon chizadi.

    Args:
        mdt: mdt_munozara natijasi.
        agent_holat: I/Z zaxira.

    Returns:
        None. Streamlit. Tashxis emas.
    """
    _mdt_panel(mdt, agent_holat)


def laboratoriya_matn(lab: Dict[str, float]) -> str:
    """Laboratoriya qiymatlarini agent so‘roviga qo‘shiladigan matnga aylantiradi.

    Args:
        lab: Ko‘rsatkich nomi → son.

    Returns:
        Qisqa klinik matn. Tashxis emas.
    """
    qismlar = [f"{k}={v}" for k, v in lab.items()]
    return "Laboratory: " + ", ".join(qismlar) if qismlar else ""


def ehtiyotkor_tavsiyalar(bemor: Dict[str, Any], agent_holat: Dict[str, Any]) -> List[str]:
    """EKG o‘lchovi va lab qiymatlaridan ehtiyotkor, umumiy tavsiyalar.

    Args:
        bemor: Forma ma’lumotlari va laboratoriya.
        agent_holat: ChiefCardiologist.run natijasi.

    Returns:
        Shifokor o‘rnini bosmaydigan qisqa bandlar.
    """
    tavsiya = [
        KLINIK_OGOHLANTIRISH,
        "O‘tkir ko‘krak og‘rig‘i, nafas qisishi yoki hushdan ketishda zudlik bilan shifokorga murojaat.",
    ]
    lab = (agent_holat.get("lab_natija") or {}).get("qiymatlar") or bemor.get("laboratoriya") or {}
    troponin = lab.get("troponin_i")
    if troponin is not None and troponin >= 34:
        tavsiya.append(
            "Troponin I kiritilgan qiymati yuqori ko‘rinadi — shoshilinch klinik baholash kerak "
            "(qiymat usulga bog‘liq, tashxis emas)."
        )
    kaliy = lab.get("kaliy")
    if kaliy is not None and (kaliy < 3.5 or kaliy > 5.1):
        tavsiya.append("Kaliy odatiy oraliqdan tashqarida — EKG va elektrolitni shifokor tekshirsin.")
    ntp = lab.get("nt_probnp")
    if ntp is not None and ntp >= 300:
        tavsiya.append("NT-proBNP yuqori bo‘lishi mumkin — yurak yetishmovchiligi shubhasi klinik tasdiqlansin.")
    ecg = agent_holat.get("ecg_natija") or {}
    if ecg.get("ok"):
        tavsiya.append(
            f"EKG o‘lchovlari taxminiy: HR {ecg.get('yurak_chastotasi_bpm')} bpm, "
            f"QRS {ecg.get('qrs_ms')} ms, PR {ecg.get('pr_ms')} ms, "
            f"QT {ecg.get('qt_ms')} ms, QTc {ecg.get('qtc_bazett_ms')} ms."
        )
    elif bemor.get("ecg_signal") is not None:
        tavsiya.append("EKG o‘lchovi to‘liq chiqmadi; xom grafikni shifokor ko‘rsin.")
    return tavsiya


def _ustun1_forma() -> Tuple[Dict[str, Any], Any, float, bool]:
    """Bemor va laboratoriya maydonlarini 1-ustunda guruhlangan holda chizadi.

    Args:
        Yo‘q. Streamlit widgetlari.

    Returns:
        bemor lug‘ati (EKGsiz), yuklangan EKG fayllar ro‘yxati, sampling_rate,
        yagona tahlil tugmasi. Demografiya doimo ko‘rinadi; tekshiruvlar alohida
        bo‘lim; laboratoriya 2-ustunli grid; disk yo‘li va Hz kengaytirilganda.
    """
    # ── 1. Asosiy klinik ma‘lumotlar ─────────────────────────────────────────────────
    st.markdown('<div class="bolim-sarlavha">Bemor</div>', unsafe_allow_html=True)
    yosh = st.number_input("Yosh (yil)", min_value=0, max_value=120, value=58, step=1)
    jins = st.selectbox("Jins", ["erkak", "ayol", "ko‘rsatilmagan"])
    shikoyatlar = st.text_area(
        "Shikoyatlar",
        value="Ko‘krak og‘rig‘i, zo‘riqishda nafas qisishi",
        height=80,
    )
    anamnez = st.text_area(
        "Anamnez",
        value="Gipertenziya. Oldingi MI yo‘q.",
        height=70,
    )
    dorilar = st.text_area(
        "Dori-darmonlar (har qator: nom | doza | chastota)",
        value="aspirin | 75 mg | once daily\nbisoprolol | 5 mg | once daily",
        height=70,
        help="Yoki: aspirin 75 mg od",
    )

    # ── 2. Tekshiruvlar (fayl yuklash) ────────────────────────────────────────────────
    st.markdown('<div class="bolim-sarlavha">Tekshiruvlar</div>', unsafe_allow_html=True)
    echo_yuk = st.file_uploader(
        "Echo (DICOM/video/rasm, bir nechta)",
        type=["dcm", "dicom", "mp4", "avi", "mov", "mkv", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="11 ko‘rinish: A2C A4C A3C PLAX PSAX-* subcostal SSN. Tashxis emas.",
    )
    if echo_yuk:
        st.success(f"Echo yuklandi: {len(echo_yuk)} ta fayl")
    lab_fayl = st.file_uploader(
        "Lab hisoboti (CSV yoki PDF)",
        type=["csv", "pdf", "txt"],
        help="Yuklansa quyidagi raqamli maydonlar ustidan yoziladi.",
    )
    fayllar = st.file_uploader(
        "EKG fayl(lar) (CSV yoki WFDB .hea+.dat)",
        type=["csv", "hea", "dat", "txt"],
        accept_multiple_files=True,
        help="CSV: I..V6 yoki 12 ustun; ixtiyoriy `# sampling_rate=500`. WFDB: .hea va .dat ni birga.",
    )
    namuna = st.checkbox("Namuna (sintetik) EKG ishlatish", value=True)

    # ── 3. Laboratoriya raqamlari — 2-ustunli grid ──────────────────────────────────────────────
    with st.expander("Laboratoriya qiymatlari", expanded=True):
        st.caption("Orientir; CSV/PDF yuklansa forma ustidan yoziladi. Tashxis emas.")
        lab: Dict[str, float] = {}
        juftlar = [LAB_MAYDONLARI[i : i + 2] for i in range(0, len(LAB_MAYDONLARI), 2)]
        for juft in juftlar:
            kol_chap, kol_ong = st.columns(2)
            for kol, (kalit, sarlavha_lab, izoh, past, yuqori, odatiy) in zip(
                [kol_chap, kol_ong], juft
            ):
                with kol:
                    lab[kalit] = float(
                        st.number_input(
                            sarlavha_lab,
                            min_value=past,
                            max_value=yuqori,
                            value=odatiy,
                            step=0.1,
                            help=izoh,
                        )
                    )

    # ── 3.5 Lab rangli kartalar (tahlildan keyin) ─────────────────────────────────────────────────
    _lab_holat_s = st.session_state.get("agent_holat")
    if _lab_holat_s and (_lab_holat_s.get("lab_natija") or {}).get("qiymatlar"):
        _lab_kartalar(_lab_holat_s.get("lab_natija"), bemor=None)

    # ── 4. Kengaytirilgan sozlamalar (odatda yopiq) ────────────────────────────────────────────────
    with st.expander("Kengaytirilgan sozlamalar", expanded=False):
        echo_fayl = st.text_input("Echo/DICOM disk yo‘li (ixtiyoriy)", value="")
        sampling_rate = st.number_input(
            "EKG sampling_rate (Hz)",
            min_value=100.0,
            max_value=2000.0,
            value=ODATIY_HZ,
            help="CSV/WFDB da avtoaniqlansa shu qiymat bekor bo‘ladi.",
        )
        st.caption(
            "CSV: I..V6 yoki 12 ustun; ixtiyoriy `# sampling_rate=500` va `time` ustuni. "
            "WFDB: .hea va .dat ni birga."
        )

    # ── 5. Yagona tahlil tugmasi ──────────────────────────────────────────────────────────────────
    tahlil = st.button("Tahlil qilish", type="primary", use_container_width=True, key="tahlil_past")

    bemor = {
        "yosh": int(yosh),
        "jins": jins,
        "shikoyatlar": shikoyatlar,
        "anamnez": anamnez,
        "dorilar": dorilar,
        "echo_fayl": echo_fayl.strip() or None,
        "laboratoriya": lab,
        "klinik_savol": laboratoriya_matn(lab),
        "sampling_rate": float(sampling_rate),
        "namuna_ekg": namuna,
    }
    if lab_fayl is not None:
        bemor["lab_fayl_bayt"] = lab_fayl.getvalue()
        bemor["lab_fayl_nomi"] = lab_fayl.name
    if echo_yuk:
        bemor["echo_fayllar"] = [{"nom": f.name, "bayt": f.getvalue()} for f in echo_yuk]
    return bemor, fayllar, float(sampling_rate), tahlil


def _ustun2_ekg(
    signal: Optional[np.ndarray],
    sampling_rate: float,
    xabar: str,
    agent_holat: Optional[Dict[str, Any]] = None,
) -> None:
    """2-ustunda EKG grafigini chiqaradi (tozalangan + P/QRS/T).

    Args:
        signal: (n, 12) massiv yoki None.
        sampling_rate: Hz.
        xabar: Yuklash/namuna izohi.
        agent_holat: EP to‘lqin indekslari uchun.

    Returns:
        None. Streamlit ga chizadi.
    """
    st.subheader("EKG signali")
    if signal is None:
        st.info("CSV/WFDB yuklang yoki namuna EKG ni belgilang.")
        return
    st.caption(xabar)
    tasma = st.selectbox("Ko‘rsatish", ["12 tasma"] + TASMA_NOMLARI, index=2)
    tozalangan_kor = st.checkbox("Tozalangan signal", value=True)
    belgi = st.checkbox("P / QRS / T belgilari (EP)", value=True)
    if tozalangan_kor:
        chizma, sarlavha = _tozalangan_ekg(signal, sampling_rate)
        if chizma is None:
            chizma = np.asarray(signal, dtype=float)
            sarlavha = "Xom EKG (ko‘rish, tashxis emas)"
    else:
        chizma = np.asarray(signal, dtype=float)
        sarlavha = "Xom EKG (ko‘rish, tashxis emas)"
    holat = agent_holat or {}
    ecg = holat.get("ecg_natija") or {}
    ep = holat.get("ep_natija") if isinstance(holat.get("ep_natija"), dict) else {}
    tolqinlar = (ep or {}).get("tolqinlar") or ecg.get("tolqinlar")
    if not belgi:
        tolqinlar = None
    ep_tasma = (ep or {}).get("tasma") or ecg.get("tasma")
    fig = ekg_grafik_chiz(
        chizma,
        sampling_rate,
        tasma,
        tolqinlar=tolqinlar,
        ep_tasma=ep_tasma,
        sarlavha=sarlavha,
    )
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    # EKG metrik badge'lari (HR/QRS/QTc/PR)
    _ecg_metrik(
        (agent_holat or {}).get("ecg_natija"),
        (agent_holat or {}).get("ep_natija") if isinstance((agent_holat or {}).get("ep_natija"), dict) else None,
    )
    holat = agent_holat or {}
    echo_n = holat.get("echo_natija") or {}
    if echo_n.get("korinishlar"):
        st.caption("Echo: " + ", ".join(echo_n.get("korinishlar") or []) + " — 11 ko‘rinish vizual tekshiruvda.")
    if (holat.get("echo_mask") or {}).get("ok"):
        st.caption("LV overlay vizual tekshiruvda.")
    if holat.get("mdt_natija"):
        st.caption("MDT raundlari vizual tekshiruvda yonma-yon.")


def _vizual_tekshirish_paneli(
    agent_holat: Optional[Dict[str, Any]],
    bemor: Optional[Dict[str, Any]],
    signal: Optional[np.ndarray],
    sampling_rate: float,
) -> None:
    """Shifokor uchun oraliq vizual natijalar: EKG, 11 echo, LV, MDT.

    Args:
        agent_holat: run() natijasi.
        bemor: Yuklangan echo fayllar.
        signal: EKG massivi.
        sampling_rate: Hz.

    Returns:
        None. Tashxis o‘rnini bosmaydi.
    """
    holat = agent_holat or {}
    st.divider()
    st.subheader("Vizual tekshirish paneli")
    st.caption(
        "Shifokor so‘rovi bilan ochilgan oraliq natijalar "
        "(tozalangan EKG, 11 echo ko‘rinish, LV overlay, MDT). "
        "Tashxis emas; yakuniy qaror shifokorga tegishli."
    )
    viz = holat.get("vizual") or {}
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("EKG sifat", str(viz.get("ekg_sifat") or "—"))
    e2.metric(
        "Echo ko‘rinish",
        len(viz.get("echo_korinishlar") or (holat.get("echo_natija") or {}).get("korinishlar") or []),
    )
    e3.metric("LV maska", "bor" if (holat.get("echo_mask") or {}).get("ok") else "yo‘q")
    e4.metric("MDT raund", viz.get("mdt_raund") if viz.get("mdt_raund") is not None else ((holat.get("mdt_natija") or {}).get("raund_soni") or "—"))

    tab_ekg, tab_echo, tab_lv, tab_mdt = st.tabs(
        ["Tozalangan 12 tasma", "Echo 11 ko‘rinish", "LV overlay", "MDT raundlari"]
    )
    with tab_ekg:
        chizma, sarlavha = _tozalangan_ekg(signal, sampling_rate)
        if chizma is None:
            st.info("EKG yo‘q — CSV/WFDB yoki namuna belgilang.")
        else:
            ep = holat.get("ep_natija") if isinstance(holat.get("ep_natija"), dict) else {}
            ecg = holat.get("ecg_natija") or {}
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("HR", ecg.get("yurak_chastotasi_bpm") or ep.get("yurak_chastotasi_bpm") or "—")
            m2.metric("PR ms", ecg.get("pr_ms") or ep.get("pr_ms") or "—")
            m3.metric("QRS ms", ecg.get("qrs_ms") or ep.get("qrs_ms") or "—")
            m4.metric("QT ms", ecg.get("qt_ms") or ep.get("qt_ms") or "—")
            m5.metric("QTc", ecg.get("qtc_bazett_ms") or ep.get("qtc_bazett_ms") or "—")
            fig = ekg_grafik_chiz(
                chizma,
                sampling_rate,
                "12 tasma",
                tolqinlar=(ep or {}).get("tolqinlar") or ecg.get("tolqinlar"),
                ep_tasma=(ep or {}).get("tasma") or ecg.get("tasma"),
                sarlavha=sarlavha,
            )
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            st.caption(viz.get("ekg_xabar") or (holat.get("ecg_technician_natija") or {}).get("xabar") or "")

    with tab_echo:
        echo = holat.get("echo_natija") or {}
        if echo.get("xabar"):
            st.caption(str(echo.get("xabar")) + " Tashxis emas.")
        xarita = _echo_kadr_xarita(bemor, echo)
        yet = echo.get("yetishmagan_standart") or viz.get("echo_yetishmagan") or []
        qatorlar = [ECHO_KORINISHLAR[i : i + 4] for i in range(0, len(ECHO_KORINISHLAR), 4)]
        for qator in qatorlar:
            ustunlar = st.columns(len(qator))
            for i, yorliq in enumerate(qator):
                with ustunlar[i]:
                    st.markdown(f"**{yorliq}**")
                    yozuvlar = xarita.get(yorliq) or []
                    if not yozuvlar:
                        st.info("yo‘q")
                        continue
                    bir = yozuvlar[0]
                    kadrlar = bir.get("kadrlar") or []
                    st.image(_kadr_rgb(kadrlar[0]), caption=str(bir.get("nom") or ""), width="stretch")
                    if len(kadrlar) >= 2:
                        kichik = st.columns(min(3, len(kadrlar)))
                        idx = [int(round(j * (len(kadrlar) - 1) / (len(kichik) - 1))) for j in range(len(kichik))] if len(kichik) > 1 else [0]
                        for j, col in enumerate(kichik):
                            with col:
                                st.image(_kadr_rgb(kadrlar[idx[j]]), width="stretch")
                    st.caption(f"{bir.get('manba') or ''} p={bir.get('ehtimol')}")
        if yet:
            st.caption("11 dan hali yo‘q: " + ", ".join(yet))
        if xarita.get("ANIQLANMAGAN"):
            st.markdown("**ANIQLANMAGAN**")
            for yoz in xarita["ANIQLANMAGAN"]:
                st.image(_kadr_rgb(yoz["kadrlar"][0]), caption=str(yoz.get("nom")), width="stretch")

    with tab_lv:
        mask = holat.get("echo_mask") or {}
        if mask.get("overlay_png"):
            st.image(
                mask["overlay_png"],
                caption=mask.get("xabar") or "LV kontur (algoritmik, EF/tashxis emas)",
                width="stretch",
            )
            st.write(
                f"ko‘rinish={mask.get('korinish')} | {mask.get('maydon_px')} px | "
                f"ulush={mask.get('ulush')} | model={mask.get('model')}"
            )
        elif mask.get("xabar"):
            st.warning(mask.get("xabar"))
        else:
            st.info("LV overlay yo‘q — echo kadr tahlildan keyin chiqadi.")

    with tab_mdt:
        mdt = holat.get("mdt_natija")
        if mdt:
            _mdt_panel(mdt, holat)
        else:
            st.info("MDT hali yo‘q. EKG/echo yoki murakkab holatda tahlildan keyin chiqadi.")


def _ustun3_xulosa(agent_holat: Optional[Dict[str, Any]], bemor: Optional[Dict[str, Any]]) -> None:
    """3-ustunda agent xulosasi va ehtiyotkor tavsiyalarni ko'rinishli karta bilan chizadi.

    Args:
        agent_holat: run() natijasi yoki None.
        bemor: Tahlil qilingan bemor.

    Returns:
        None. Xulosani ajratilgan karta ichida ko'rsatadi; audit sidebar da.
        Klinik tashxis o'rnini bosmaydi.
    """
    st.subheader("AI tibbiy xulosa")

    # ── Bo'sh holat ───────────────────────────────────────────────────────────
    if not agent_holat:
        st.markdown(
            '<div class="bosh-holat-ogoh">'
            "Chapdan bemor ma’lumotlarini va tekshiruvlarni kiriting, so‘ng "
            "<strong>Tahlil qilish</strong> tugmasini bosing."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Holat satri — TASK 7: rangli badge qatori ────────────────────────────
    _murakkablik = str(agent_holat.get("murakkablik") or "")
    _amal = str(agent_holat.get("amal") or "")
    _risk_cls = "oddiy" if _murakkablik == "oddiy" else ("murakkab" if _murakkablik == "murakkab" else "unknown")
    _amal_cls = "stop" if _amal == "STOP" else "continue"
    _sababi_q = str(agent_holat.get("murakkablik_sababi") or "")[:80]
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.2rem;">'
        f'<span class="risk-badge {html.escape(_risk_cls)}">{html.escape(_murakkablik or "—")}</span>'
        f'<span class="agent-badge {html.escape(_amal_cls)}">{html.escape(_amal or "—")}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if _sababi_q:
        st.caption(_sababi_q)

    # ── Qadam panel — TASK 4 ──────────────────────────────────────────────────
    with st.expander("Qadam-ba-qadam bajarish", expanded=True):
        _qadam_panel(agent_holat)

    # ── Xulosa kartasi (asosiy fokus) ─────────────────────────────────────────
    # Streamlit har markdownni alohida element qiladi, shuning uchun
    # ochuvchi/yopuvchi div matn maydonini o‘ramaydi. border=True haqiqiy karta.
    xulosa_matn = agent_holat.get("xulosa") or ""
    with st.container(border=True):
        st.markdown("**Xulosa**")
        st.text_area(
            "xulosa_maydoni",
            value=xulosa_matn,
            height=220,
            disabled=True,
            label_visibility="collapsed",
        )

    # ── Tavsiyalar ────────────────────────────────────────────────────────────
    st.markdown("**Tavsiyalar**")
    for band in ehtiyotkor_tavsiyalar(bemor or {}, agent_holat):
        st.markdown(f"- {band}")

    # ── Agent izlar — TASK 5 ──────────────────────────────────────────────────
    with st.expander("Agent izlar va mulohaza", expanded=False):
        _agent_izlar(agent_holat)

    # ── Lab technician ────────────────────────────────────────────────────────
    lab_n = agent_holat.get("lab_natija") or {}
    if lab_n:
        with st.expander("Lab technician (token/RAG satr)"):
            st.write(lab_n.get("xabar") or "")
            st.code(lab_n.get("rag_satr") or lab_n.get("matn") or "")
            if lab_n.get("dorilar"):
                st.write(lab_n.get("dorilar"))
            st.caption("Tokenlar: " + ", ".join(lab_n.get("tokenlar") or []))

    # ── Klinik reja va qadamlar ───────────────────────────────────────────────
    with st.expander("Klinik reja P", expanded=True):
        _reja_jonli(agent_holat)
    with st.expander("Qadamlar (LangGraph)", expanded=True):
        for qator in agent_holat.get("qadam_tarixi") or []:
            st.write(qator)

    # ── CardiacRAG dalillari ──────────────────────────────────────────────────
    dalillar = agent_holat.get("rag_dalillar") or []
    if dalillar:
        with st.expander("CardiacRAG kontekst C"):
            for d in dalillar:
                st.write(d)

    # ── MDT munozara ──────────────────────────────────────────────────────────
    mdt = agent_holat.get("mdt_natija")
    if mdt:
        with st.expander("MDT munozara (qisqa)", expanded=False):
            _mdt_panel(mdt, agent_holat)

    # ── EKG, Echo, Fellow expanders ───────────────────────────────────────────
    tech_n = agent_holat.get("ecg_technician_natija") or {}
    if tech_n:
        with st.expander("EKG technician (tozalash / sifat)"):
            st.write(tech_n.get("xabar") or "")
            st.write(
                f"Sifat: {tech_n.get('sifat')} | tasma {tech_n.get('tasma')} | "
                f"R={tech_n.get('urishlar_soni')} | HR\u2248{tech_n.get('yurak_chastotasi_bpm')} | "
                f"{tech_n.get('davomiylik_s')} s @ {tech_n.get('sampling_rate')} Hz"
            )
            if tech_n.get("yetishmagan"):
                st.caption("Yetishmagan tasma: " + ", ".join(tech_n.get("yetishmagan") or []))
    ep_n = agent_holat.get("ep_natija") or {}
    if ep_n:
        with st.expander("Elektrofiziolog (P/QRS/T, intervallar)"):
            st.write(ep_n.get("xabar") or "")
            st.write(
                f"tasma {ep_n.get('tasma')} | HR {ep_n.get('yurak_chastotasi_bpm')} | "
                f"PR {ep_n.get('pr_ms')} | QRS {ep_n.get('qrs_ms')} | "
                f"QT {ep_n.get('qt_ms')} | QTc {ep_n.get('qtc_bazett_ms')} | "
                f"SDNN {ep_n.get('hrv_sdnn_ms')} | RMSSD {ep_n.get('hrv_rmssd_ms')} | "
                f"P={ep_n.get('p_soni')} T={ep_n.get('t_soni')}"
            )
            if ep_n.get("eslatmalar"):
                for e in ep_n.get("eslatmalar") or []:
                    st.markdown(f"- {e}")
            st.caption("Qiymatlar taxminiy; tashxis emas.")
    echo_n = agent_holat.get("echo_natija") or {}
    if echo_n:
        with st.expander("Echo technician (11 ko'rinish)"):
            st.write(echo_n.get("xabar") or "")
            st.write(echo_n.get("yozuvlar") or echo_n.get("korinishlar"))
            yet = echo_n.get("yetishmagan_standart") or []
            if yet:
                st.caption("Hali yo'q (11 dan): " + ", ".join(yet))
    mask_n = agent_holat.get("echo_mask") or {}
    if mask_n:
        with st.expander("Echo segmenter (LV maska)"):
            st.write(mask_n.get("xabar") or "")
            if mask_n.get("ok"):
                st.write(
                    f"ko'rinish={mask_n.get('korinish')} | {mask_n.get('maydon_px')} px | "
                    f"ulush={mask_n.get('ulush')} | model={mask_n.get('model')}"
                )
            st.caption("Algoritmik kavak; klinik EF emas. To'liq rasm vizual tekshiruvda.")
    fellow_n = agent_holat.get("fellow_natija") or {}
    if fellow_n:
        with st.expander("Cardiology fellow (multimodal)"):
            st.write(f"manba={fellow_n.get('manba')} model={fellow_n.get('model')} | dalillar={fellow_n.get('dalillar')}")
            if fellow_n.get("yoq_dalillar"):
                st.caption("Yo'q (o'ylab topilmagan): " + ", ".join(fellow_n.get("yoq_dalillar") or []))
            st.write(fellow_n.get("matn") or "")
            st.caption((fellow_n.get("xabar") or "") + " Tashxis emas.")
            if fellow_n.get("manba") == "shablon" or (agent_holat.get("mdt_natija") or {}).get("medgemma_manba") == "shablon":
                st.warning("Fellow/MDT **shablon rejimida** (LLM yoki VLM javobi yo'q).")

    # ── Vizual holat eslatmasi ─────────────────────────────────────────────────
    viz = agent_holat.get("vizual") or {}
    st.caption(
        "Vizual tekshiruv shifokor so'raganda ochiladi: tozalangan 12 tasma, echo, LV, MDT. "
        f"sifat={viz.get('ekg_sifat')} echo={viz.get('echo_korinishlar')} "
        f"lv={viz.get('lv_maska')} mdt_raund={viz.get('mdt_raund')}."
    )


def asosiy() -> None:
    """Sahifani 3 ustunda yig'adi va tahlil tugmasida agentni ishga tushiradi.

    Args:
        Yo'q.

    Returns:
        None. Streamlit ilovasi. Texnik holat (RAG/LLM/audit) sidebar da;
        asosiy kanvas klinik vazifadan boshlanadi.
    """
    st.set_page_config(
        page_title="Kardiologik AI agent",
        layout="wide",
        page_icon="\U0001fac0",
    )
    st.title("Kardiologik AI agent")

    # Klinik ogohlantirish sahifa yuqorida qoladi (sticky)
    _klinik_ogohlantirish()

    # RAG yuklanadi; kesh eskirsa yangilanadi
    rag_paket = _rag_ol()
    rag = _rag_obyekt(rag_paket)
    if rag is not None and not hasattr(rag, "bert_ishlatildi"):
        _rag_ol.clear()
        rag_paket = _rag_ol()
        rag = _rag_obyekt(rag_paket)

    if "ekg_signal" not in st.session_state:
        st.session_state.ekg_signal = namuna_ekg_signal()
        st.session_state.ekg_xabar = "Sintetik namuna EKG (haqiqiy yozuv emas, tahlil kutilmoqda)."
        st.session_state.sampling_rate = ODATIY_HZ
        st.session_state.agent_holat = None
        st.session_state.bemor_tahlil = None
        st.session_state.vizual_ochiq = False

    # Sidebar: texnik holat, audit, tozalash tugmasi
    _sidebar_paneli(rag_paket, st.session_state.agent_holat)

    # 6 bosqichli progress qatori asosiy kanvas yuqorida
    bosqich_joy = st.empty()
    with bosqich_joy.container():
        _olti_bosqich_chiz(st.session_state.agent_holat)
        if st.session_state.agent_holat:
            _reja_jonli(st.session_state.agent_holat)

    col1, col2, col3 = st.columns([1.05, 1.25, 1.15], gap="large")
    with col1:
        bemor, fayllar, sampling_rate, tahlil = _ustun1_forma()

    signal = st.session_state.ekg_signal
    xabar = st.session_state.ekg_xabar
    if tahlil:
        if fayllar:
            juft = [(f.name, f.getvalue()) for f in fayllar]
            natija = ekg_fayllardan_oqish(juft, sampling_rate_hint=sampling_rate)
            signal = natija.get("signal")
            xabar = str(natija.get("xabar") or "")
            if natija.get("sampling_rate"):
                sampling_rate = float(natija["sampling_rate"])
        elif bemor.get("namuna_ekg"):
            signal = namuna_ekg_signal(sampling_rate)
            xabar = "Sintetik namuna EKG (haqiqiy yozuv emas)."
        else:
            signal, xabar = None, "EKG yo'q."
        st.session_state.ekg_signal = signal
        st.session_state.ekg_xabar = xabar
        st.session_state.sampling_rate = sampling_rate
        paket = dict(bemor)
        if signal is not None:
            paket["ecg_signal"] = signal
        paket["sampling_rate"] = sampling_rate
        lab_oldindan = process_lab(paket)
        if lab_oldindan.get("qiymatlar"):
            paket["laboratoriya"] = lab_oldindan["qiymatlar"]
        if lab_oldindan.get("rag_satr"):
            paket["klinik_savol"] = lab_oldindan["rag_satr"]
        if lab_oldindan.get("dorilar"):
            paket["dorilar_tuzilgan"] = lab_oldindan["dorilar"]
        st.session_state.vizual_ochiq = False
        oxirgi: Optional[Dict[str, Any]] = None
        try:
            agent = _agent_ol()
            with st.status("Agent tahlil qilmoqda (6 bosqich)...", expanded=True) as holat_ui:
                for holat in agent.run_oqim(paket):
                    oxirgi = holat
                    holat_ui.update(
                        label="Oqim: "
                        + str((holat.get("qadam_tarixi") or ["..."])[-1])[:120],
                        state="running",
                    )
                    with bosqich_joy.container():
                        _olti_bosqich_chiz(holat)
                        _reja_jonli(holat)
                    st.session_state.agent_holat = holat
                if oxirgi is None:
                    oxirgi = agent.run(paket)
                holat_ui.update(label="Tahlil tugadi (tashxis emas).", state="complete")
            st.session_state.agent_holat = oxirgi
        except Exception as exc:
            st.session_state.agent_holat = {
                "amal": "STOP",
                "xulosa": (
                    f"Tahlil bajarilmadi: {exc}. "
                    "Bu tashxis emas; shifokor bilan ko'rib chiqing."
                ),
                "qadam_tarixi": [str(exc)],
                "ecg_natija": None,
                "rag_dalillar": [],
            }
            st.error("Tahlil xatosi: " + str(exc))
        st.session_state.bemor_tahlil = paket

    with col2:
        _ustun2_ekg(
            st.session_state.ekg_signal,
            float(st.session_state.sampling_rate),
            st.session_state.ekg_xabar,
            st.session_state.agent_holat,
        )
    with col3:
        # Audit sidebar da; bu yerda faqat klinik xulosa
        _ustun3_xulosa(st.session_state.agent_holat, st.session_state.bemor_tahlil)

    if st.session_state.agent_holat:
        yozuv = "Vizual panelni yopish" if st.session_state.get("vizual_ochiq") else "Vizual tekshiruvni ochish"
        if st.button(yozuv):
            st.session_state.vizual_ochiq = not bool(st.session_state.get("vizual_ochiq"))
            st.rerun()
    if st.session_state.get("vizual_ochiq") and st.session_state.agent_holat:
        _vizual_tekshirish_paneli(
            st.session_state.agent_holat,
            st.session_state.bemor_tahlil,
            st.session_state.ekg_signal,
            float(st.session_state.sampling_rate),
        )


asosiy()
