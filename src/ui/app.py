"""Kardiologik AI agent uchun 3 ustunli Streamlit interfeysi.

1-ustun: bemor va laboratoriya. 2-ustun: 12 tasmali EKG grafigi.
3-ustun: ChiefCardiologist xulosasi. Tashxis o‘rnini bosmaydi.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
from rag.medical_rag import MedicalRAG
from tools.ecg_tool import TASMA_NOMLARI
from tools.lab_tool import process_lab

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
def _rag_ol(kesh_versiya: int = 3) -> Optional[MedicalRAG]:
    """BERT ni jarayonda bir marta, FAISS ni disk keshdan yuklaydi.

    Args:
        kesh_versiya: Streamlit cache kaliti; atributlar o‘zgaganda oshiriladi.

    Returns:
        Indekslangan RAG yoki None. BERT bo‘lmasa hashing zaxirasi.
        Seed o‘zgarmasa embedding qayta hisoblanmaydi.
    """
    try:
        rag = MedicalRAG()
        rag.urug_indeks()
        return rag
    except Exception:
        return None


def _rag_holat_matn(rag: Optional[MedicalRAG]) -> str:
    """UI caption uchun RAG holatini yozadi (eski kesh obyektiga chidamli).

    Args:
        rag: MedicalRAG yoki None.

    Returns:
        Qisqa holat matni. Tashxis emas.
    """
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


def _agent_ol() -> ChiefCardiologist:
    """LangGraph bosh kardiologini CardiacRAG bilan quradi.

    Returns:
        6 bosqichli workflow agenti.
    """
    return ChiefCardiologist(rag=_rag_ol())


def namuna_ekg_signal(sampling_rate: float = ODATIY_HZ, davomiylik: float = NAMUNA_SONIYA) -> np.ndarray:
    """O‘qitish uchun 12 tasmali sintetik EKG (n, 12) yasaydi.

    Args:
        sampling_rate: Namuna olish tezligi (Hz).
        davomiylik: Yozuv uzunligi (soniya).

    Returns:
        (n, 12) massiv. Haqiqiy bemor EKG si emas.
    """
    n = int(sampling_rate * davomiylik)
    t = np.arange(n) / sampling_rate
    hr = 72.0
    rr = 60.0 / hr
    asos = np.zeros(n)
    for k in np.arange(0.3, davomiylik - 0.3, rr):
        asos += 0.12 * np.exp(-((t - (k - 0.16)) ** 2) / (2 * 0.012**2))
        asos += 1.15 * np.exp(-((t - k) ** 2) / (2 * 0.012**2))
        asos -= 0.18 * np.exp(-((t - (k + 0.04)) ** 2) / (2 * 0.016**2))
        asos += 0.28 * np.exp(-((t - (k + 0.22)) ** 2) / (2 * 0.05**2))
    asos += 0.02 * np.sin(2 * np.pi * 0.25 * t)
    shkala = np.array([1.0, 1.1, 0.35, -0.55, 0.45, 0.7, 0.5, 0.85, 1.05, 1.2, 1.1, 0.95])
    shovqin = 0.015 * np.random.default_rng(7).normal(size=(n, 12))
    return (asos[:, None] * shkala[None, :]) + shovqin


def csv_dan_ekg(fayl_bayt: bytes) -> Tuple[Optional[np.ndarray], str]:
    """CSV fayldan 12 tasmali EKG o‘qiydi (sarlavha I..V6 yoki 12 ustun).

    Args:
        fayl_bayt: Yuklangan CSV baytlari.

    Returns:
        (massiv, xabar). Xatoda massiv None. Tashxis qo‘yilmaydi.
    """
    try:
        matn = fayl_bayt.decode("utf-8-sig")
    except UnicodeDecodeError:
        matn = fayl_bayt.decode("latin-1")
    oquvchi = csv.reader(io.StringIO(matn))
    qatorlar = [[c.strip() for c in q] for q in oquvchi if any(x.strip() for x in q)]
    if not qatorlar:
        return None, "CSV bo‘sh."
    birinchi = [c.upper().replace("AVR", "aVR").replace("AVL", "aVL").replace("AVF", "aVF") for c in qatorlar[0]]
    if all(nom in birinchi for nom in TASMA_NOMLARI):
        indekslar = [birinchi.index(nom) for nom in TASMA_NOMLARI]
        sonlar: List[List[float]] = []
        for qator in qatorlar[1:]:
            try:
                sonlar.append([float(qator[i]) for i in indekslar])
            except (ValueError, IndexError):
                continue
        if not sonlar:
            return None, "Sarlavhali CSV da sonli qator topilmadi."
        return np.asarray(sonlar, dtype=float), "12 tasma sarlavha bo‘yicha o‘qildi."
    sonlar = []
    for qator in qatorlar:
        if len(qator) < 12:
            continue
        try:
            sonlar.append([float(x) for x in qator[:12]])
        except ValueError:
            continue
    if not sonlar:
        return None, "CSV da 12 ustunli sonlar yo‘q. Sarlavha: I,II,...,V6."
    return np.asarray(sonlar, dtype=float), "12 ustunli CSV o‘qildi."


def ekg_grafik_chiz(signal: np.ndarray, sampling_rate: float, tasma_tanlov: str) -> plt.Figure:
    """EKG ni Streamlit uchun matplotlib figuraga chizadi.

    Args:
        signal: (n, 12) yoki (12, n) massiv.
        sampling_rate: Hz.
        tasma_tanlov: Bitta tasma nomi yoki \"12 tasma\".

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
    if tasma_tanlov == "12 tasma":
        fig, oqlar = plt.subplots(6, 2, figsize=(8.2, 9.2), sharex=True)
        for i, nom in enumerate(TASMA_NOMLARI):
            ax = oqlar[i // 2][i % 2]
            ax.plot(t, massiv[:, i], color="#b71c1c", linewidth=0.7)
            ax.set_ylabel(nom, rotation=0, labelpad=18, va="center", fontsize=9)
            ax.set_yticks([])
        oqlar[-1][0].set_xlabel("Vaqt (s)")
        oqlar[-1][1].set_xlabel("Vaqt (s)")
        fig.suptitle("12 tasmali EKG (ko‘rish, tashxis emas)", fontsize=11)
        fig.tight_layout()
        return fig

    idx = TASMA_NOMLARI.index(tasma_tanlov) if tasma_tanlov in TASMA_NOMLARI else 1
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.plot(t, massiv[:, idx], color="#b71c1c", linewidth=0.9)
    ax.set_xlabel("Vaqt (s)")
    ax.set_ylabel("Amplitude (shartli)")
    ax.set_title(f"Tasma {tasma_tanlov} (ko‘rish, tashxis emas)")
    fig.tight_layout()
    return fig


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
        "Bu tizim shifokor o‘rnini bosmaydi; yakuniy qaror klinik ko‘rikka tegishli.",
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
            f"QRS {ecg.get('qrs_ms')} ms, PR {ecg.get('pr_ms')} ms, QT {ecg.get('qt_ms')} ms."
        )
    elif bemor.get("ecg_signal") is not None:
        tavsiya.append("EKG o‘lchovi to‘liq chiqmadi; xom grafikni shifokor ko‘rsin.")
    return tavsiya


def _ustun1_forma() -> Tuple[Dict[str, Any], Any, float, bool]:
    """Bemor va laboratoriya maydonlarini 1-ustunda chizadi.

    Args:
        Yo‘q. Streamlit widgetlari.

    Returns:
        bemor lug‘ati (EKGsiz), yuklangan fayl, sampling_rate, tahlil tugmasi.
    """
    st.subheader("Bemor ma’lumotlari")
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
    echo_fayl = st.text_input("Echo/DICOM yo‘li (ixtiyoriy)", value="")
    tahlil_yuqori = st.button("Tahlil qilish", type="primary", width="stretch")
    st.subheader("Laboratoriya")
    st.caption("Qiymatlar orientir; CSV/PDF yuklansa forma ustidan yoziladi. Tashxis emas.")
    lab_fayl = st.file_uploader("Lab hisoboti (CSV yoki PDF)", type=["csv", "pdf", "txt"])
    lab: Dict[str, float] = {}
    for kalit, sarlavha, izoh, past, yuqori, odatiy in LAB_MAYDONLARI:
        lab[kalit] = float(
            st.number_input(sarlavha, min_value=past, max_value=yuqori, value=odatiy, step=0.1, help=izoh)
        )
    sampling_rate = st.number_input("EKG sampling_rate (Hz)", min_value=100.0, max_value=2000.0, value=ODATIY_HZ)
    fayl = st.file_uploader("EKG CSV (12 tasma)", type=["csv"])
    namuna = st.checkbox("Namuna (sintetik) EKG ishlatish", value=True)
    tahlil_past = st.button("Tahlil qilish", type="primary", width="stretch", key="tahlil_past")
    tahlil = bool(tahlil_yuqori or tahlil_past)
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
    return bemor, fayl, float(sampling_rate), tahlil


def _ustun2_ekg(signal: Optional[np.ndarray], sampling_rate: float, xabar: str) -> None:
    """2-ustunda EKG grafigini chiqaradi.

    Args:
        signal: (n, 12) massiv yoki None.
        sampling_rate: Hz.
        xabar: Yuklash/namuna izohi.

    Returns:
        None. Streamlit ga chizadi.
    """
    st.subheader("EKG signali")
    if signal is None:
        st.info("CSV yuklang yoki namuna EKG ni belgilang.")
        return
    st.caption(xabar)
    tasma = st.selectbox("Ko‘rsatish", ["12 tasma"] + TASMA_NOMLARI, index=2)
    fig = ekg_grafik_chiz(signal, sampling_rate, tasma)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _ustun3_xulosa(agent_holat: Optional[Dict[str, Any]], bemor: Optional[Dict[str, Any]]) -> None:
    """3-ustunda agent xulosasi va ehtiyotkor tavsiyalarni ko‘rsatadi.

    Args:
        agent_holat: run() natijasi yoki None.
        bemor: Tahlil qilingan bemor.

    Returns:
        None. Klinik tashxis o‘rnini bosmaydi.
    """
    st.subheader("AI tibbiy xulosa")
    if not agent_holat:
        st.info("Chapda ma’lumotni to‘ldirib, «Tahlil qilish» ni bosing.")
        return
    amal = agent_holat.get("amal", "")
    st.metric("Agent amali", amal)
    st.caption(
        f"Murakkablik: {agent_holat.get('murakkablik') or '—'} — "
        f"{agent_holat.get('murakkablik_sababi') or ''}"
    )
    st.text_area("Xulosa", value=agent_holat.get("xulosa") or "", height=260, disabled=True)
    st.markdown("**Tavsiyalar**")
    for band in ehtiyotkor_tavsiyalar(bemor or {}, agent_holat):
        st.markdown(f"- {band}")
    lab_n = agent_holat.get("lab_natija") or {}
    if lab_n:
        with st.expander("Lab technician (token/RAG satr)"):
            st.write(lab_n.get("xabar") or "")
            st.code(lab_n.get("rag_satr") or lab_n.get("matn") or "")
            if lab_n.get("dorilar"):
                st.write(lab_n.get("dorilar"))
            st.caption("Tokenlar: " + ", ".join(lab_n.get("tokenlar") or []))
    with st.expander("Klinik reja P"):
        for qadam in agent_holat.get("reja") or []:
            st.write(f"{qadam.get('id')} [{qadam.get('holat')}] {qadam.get('vosita')}: {qadam.get('tavsif')}")
    with st.expander("Qadamlar (LangGraph)"):
        for qator in agent_holat.get("qadam_tarixi") or []:
            st.write(qator)
    dalillar = agent_holat.get("rag_dalillar") or []
    if dalillar:
        st.markdown("**CardiacRAG kontekst C**")
        for d in dalillar:
            st.write(d)
    mdt = agent_holat.get("mdt_natija")
    if mdt:
        with st.expander("MDT munozara"):
            st.write(mdt.get("umumlashtirish"))
            for r in mdt.get("raundlar") or []:
                st.markdown(f"**Raund {r.get('raund')}**")
                st.write("MedGemma: " + str(r.get("medgemma"))[:800])
                st.write("Qwen2.5-VL: " + str(r.get("qwen"))[:800])
    with st.expander("Vizual tekshirish paneli"):
        viz = agent_holat.get("vizual") or {}
        st.write(f"EKG tozalangan signal saqlangan: {viz.get('ekg_tozalangan')}")
        st.write(f"EKG izoh: {viz.get('ekg_xabar')}")
        st.write(f"LV maska tayyor: {viz.get('lv_maska')}")
        st.caption("Echo proyeksiyalari va LV maskalari model ulangach shu yerda ko‘rinadi.")
        ecg = agent_holat.get("ecg_natija") or {}
        if ecg:
            st.write(
                f"HR {ecg.get('yurak_chastotasi_bpm')} | QRS {ecg.get('qrs_ms')} | "
                f"PR {ecg.get('pr_ms')} | QT {ecg.get('qt_ms')} | "
                f"SDNN {ecg.get('hrv_sdnn_ms')} | RMSSD {ecg.get('hrv_rmssd_ms')}"
            )


def asosiy() -> None:
    """Sahifani 3 ustunda yig‘adi va tahlil tugmasida agentni ishga tushiradi.

    Args:
        Yo‘q.

    Returns:
        None. Streamlit ilovasi.
    """
    st.set_page_config(
        page_title="Kardiologik AI agent",
        layout="wide",
        page_icon="🫀",
    )
    st.title("Kardiologik AI agent")
    rag = _rag_ol()
    if rag is not None and not hasattr(rag, "bert_ishlatildi"):
        _rag_ol.clear()
        rag = _rag_ol()
    rag_holat = _rag_holat_matn(rag)
    st.caption(
        "Klinik qaror qo‘llab-quvvatlash. Tashxis va davolash faqat shifokor zimmasida. "
        + rag_holat
    )

    if "ekg_signal" not in st.session_state:
        st.session_state.ekg_signal = namuna_ekg_signal()
        st.session_state.ekg_xabar = "Sintetik namuna EKG (haqiqiy yozuv emas, tahlil kutilmoqda)."
        st.session_state.sampling_rate = ODATIY_HZ
        st.session_state.agent_holat = None
        st.session_state.bemor_tahlil = None

    col1, col2, col3 = st.columns([1.05, 1.25, 1.15], gap="large")
    with col1:
        bemor, fayl, sampling_rate, tahlil = _ustun1_forma()

    signal = st.session_state.ekg_signal
    xabar = st.session_state.ekg_xabar
    if tahlil:
        if fayl is not None:
            signal, xabar = csv_dan_ekg(fayl.getvalue())
        elif bemor.get("namuna_ekg"):
            signal = namuna_ekg_signal(sampling_rate)
            xabar = "Sintetik namuna EKG (haqiqiy yozuv emas)."
        else:
            signal, xabar = None, "EKG yo‘q."
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
        with st.spinner("Agent tahlil qilmoqda..."):
            try:
                agent = _agent_ol()
                st.session_state.agent_holat = agent.run(paket)
            except Exception as exc:
                st.session_state.agent_holat = {
                    "amal": "STOP",
                    "xulosa": (
                        f"Tahlil bajarilmadi: {exc}. "
                        "Bu tashxis emas; shifokor bilan ko‘rib chiqing."
                    ),
                    "qadam_tarixi": [str(exc)],
                    "ecg_natija": None,
                    "rag_dalillar": [],
                }
            st.session_state.bemor_tahlil = paket

    with col2:
        _ustun2_ekg(
            st.session_state.ekg_signal,
            float(st.session_state.sampling_rate),
            st.session_state.ekg_xabar,
        )
    with col3:
        _ustun3_xulosa(st.session_state.agent_holat, st.session_state.bemor_tahlil)


asosiy()
