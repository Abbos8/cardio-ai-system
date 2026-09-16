"""Kardiologik AI agent uchun 3 ustunli Streamlit interfeysi.

1-ustun: bemor va laboratoriya. 2-ustun: 12 tasmali EKG grafigi.
3-ustun: ChiefCardiologist xulosasi. Pastda vizual tekshirish paneli.
Tashxis o‘rnini bosmaydi.
"""

from __future__ import annotations

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
from tools.ecg_tool import TASMA_NOMLARI, namuna_12_tasma_ekg, tozalangan_matritsa, tasmalarni_tozala
from tools.ekg_yuklash import ekg_fayllardan_oqish
from tools.echo_view_model import ECHO_KORINISHLAR
from tools.echo_yuklash import echo_fayldan_kadrlar
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


def _mdt_yonma_yon(mdt: Dict[str, Any], agent_holat: Optional[Dict[str, Any]] = None) -> None:
    """MDT raundlarini MedGemma | Qwen yonma-yon chizadi.

    Args:
        mdt: mdt_munozara natijasi.
        agent_holat: I/Z zaxira.

    Returns:
        None. Streamlit. Tashxis emas.
    """
    holat = agent_holat or {}
    st.caption(
        f"Raund={mdt.get('raund_soni')} | konsensus={mdt.get('konsensus')} | "
        f"MedGemma={mdt.get('medgemma_manba')} | Qwen={mdt.get('qwen_manba')}"
    )
    st.write(mdt.get("umumlashtirish"))
    if mdt.get("konsensus_sabab"):
        st.caption("Konsensus: " + str(mdt.get("konsensus_sabab")))
    with st.expander("I va Z (qayta kiritilgan)"):
        st.markdown("**I (xom)**")
        st.write(mdt.get("xom_i") or holat.get("xom_i") or "")
        st.markdown("**Z (oraliq)**")
        st.write(mdt.get("oraliq_z") or holat.get("oraliq_z") or "")
    for r in mdt.get("raundlar") or []:
        st.markdown(f"**Raund {r.get('raund')}**")
        chap, ong = st.columns(2)
        with chap:
            st.markdown("MedGemma (tasvir)")
            st.caption("manba=" + str(r.get("medgemma_manba") or mdt.get("medgemma_manba")))
            st.write(str(r.get("medgemma") or "")[:1200])
        with ong:
            st.markdown("Qwen2.5-VL (video)")
            st.caption("manba=" + str(r.get("qwen_manba") or mdt.get("qwen_manba")))
            st.write(str(r.get("qwen") or "")[:1200])
        st.caption("Raund konsensus: " + str(r.get("konsensus")))


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
            f"QRS {ecg.get('qrs_ms')} ms, PR {ecg.get('pr_ms')} ms, "
            f"QT {ecg.get('qt_ms')} ms, QTc {ecg.get('qtc_bazett_ms')} ms."
        )
    elif bemor.get("ecg_signal") is not None:
        tavsiya.append("EKG o‘lchovi to‘liq chiqmadi; xom grafikni shifokor ko‘rsin.")
    return tavsiya


def _ustun1_forma() -> Tuple[Dict[str, Any], Any, float, bool]:
    """Bemor va laboratoriya maydonlarini 1-ustunda chizadi.

    Args:
        Yo‘q. Streamlit widgetlari.

    Returns:
        bemor lug‘ati (EKGsiz), yuklangan EKG fayllar ro‘yxati, sampling_rate, tahlil tugmasi.
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
    echo_fayl = st.text_input("Echo/DICOM yo‘li (ixtiyoriy, disk)", value="")
    echo_yuk = st.file_uploader(
        "Echo (DICOM/video/rasm, bir nechta)",
        type=["dcm", "dicom", "mp4", "avi", "mov", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="11 ko‘rinish: A2C A4C A3C PLAX PSAX-* subcostal SSN. Tashxis emas.",
    )
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
    st.caption("CSV: I..V6 yoki 12 ustun; ixtiyoriy `# sampling_rate=500` va `time` ustuni. WFDB: .hea va .dat ni birga.")
    fayllar = st.file_uploader(
        "EKG fayl(lar) (CSV yoki WFDB .hea+.dat)",
        type=["csv", "hea", "dat", "txt"],
        accept_multiple_files=True,
    )
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
    holat = agent_holat or {}
    echo_n = holat.get("echo_natija") or {}
    if echo_n.get("korinishlar"):
        st.caption("Echo: " + ", ".join(echo_n.get("korinishlar") or []) + " — 11 ko‘rinish pastdagi panelda.")
    if (holat.get("echo_mask") or {}).get("ok"):
        st.caption("LV overlay pastdagi vizual panelda.")
    if holat.get("mdt_natija"):
        st.caption("MDT raundlari pastdagi panelda yonma-yon.")


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
        "Oraliq natijalar (tozalangan EKG, 11 echo ko‘rinish, LV overlay, MDT). "
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
            _mdt_yonma_yon(mdt, holat)
        else:
            st.info("MDT hali yo‘q. EKG/echo yoki murakkab holatda tahlildan keyin chiqadi.")


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
        with st.expander("MDT munozara (qisqa)", expanded=False):
            _mdt_yonma_yon(mdt, agent_holat)
    tech_n = agent_holat.get("ecg_technician_natija") or {}
    if tech_n:
        with st.expander("EKG technician (tozalash / sifat)"):
            st.write(tech_n.get("xabar") or "")
            st.write(
                f"Sifat: {tech_n.get('sifat')} | tasma {tech_n.get('tasma')} | "
                f"R={tech_n.get('urishlar_soni')} | HR≈{tech_n.get('yurak_chastotasi_bpm')} | "
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
        with st.expander("Echo technician (11 ko‘rinish)"):
            st.write(echo_n.get("xabar") or "")
            st.write(echo_n.get("yozuvlar") or echo_n.get("korinishlar"))
            yet = echo_n.get("yetishmagan_standart") or []
            if yet:
                st.caption("Hali yo‘q (11 dan): " + ", ".join(yet))
    mask_n = agent_holat.get("echo_mask") or {}
    if mask_n:
        with st.expander("Echo segmenter (LV maska)"):
            st.write(mask_n.get("xabar") or "")
            if mask_n.get("ok"):
                st.write(
                    f"ko‘rinish={mask_n.get('korinish')} | {mask_n.get('maydon_px')} px | "
                    f"ulush={mask_n.get('ulush')} | model={mask_n.get('model')}"
                )
            st.caption("Algoritmik kavak; klinik EF emas. To‘liq rasm pastdagi vizual panelda.")
    fellow_n = agent_holat.get("fellow_natija") or {}
    if fellow_n:
        with st.expander("Cardiology fellow (multimodal)"):
            st.write(f"manba={fellow_n.get('manba')} | dalillar={fellow_n.get('dalillar')}")
            if fellow_n.get("yoq_dalillar"):
                st.caption("Yo‘q (o‘ylab topilmagan): " + ", ".join(fellow_n.get("yoq_dalillar") or []))
            st.write(fellow_n.get("matn") or "")
            st.caption((fellow_n.get("xabar") or "") + " Tashxis emas.")
    viz = agent_holat.get("vizual") or {}
    st.caption(
        "Vizual panel pastda: tozalangan 12 tasma, echo 11 ko‘rinish, LV overlay, MDT. "
        f"sifat={viz.get('ekg_sifat')} echo={viz.get('echo_korinishlar')} "
        f"lv={viz.get('lv_maska')} mdt_raund={viz.get('mdt_raund')}."
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
            st.session_state.agent_holat,
        )
    with col3:
        _ustun3_xulosa(st.session_state.agent_holat, st.session_state.bemor_tahlil)

    _vizual_tekshirish_paneli(
        st.session_state.agent_holat,
        st.session_state.bemor_tahlil,
        st.session_state.ekg_signal,
        float(st.session_state.sampling_rate),
    )


asosiy()
