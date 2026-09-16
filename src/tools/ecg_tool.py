"""12 tasmali EKG: technician tozalash va EP intervallar/to‘lqinlar (neurokit2).

Natija taxminiy o‘lchovdir, klinik tashxis o‘rnini bosmaydi.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

try:
    import neurokit2 as nk
except ImportError:  # Kutubxona o‘rnatilmagan bo‘lsa, chaqiriqda tushunarli xabar
    nk = None

try:
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None

# Standart 12 tasma tartibi (Einthoven, Goldberger, precordial)
TASMA_NOMLARI: List[str] = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

# Ritm va intervallar uchun odatda Lead II ishlatiladi; u ishlamasa zaxira
ASOSIY_TASMA_TARTIBI: List[str] = ["II", "I", "V5", "V2", "aVF"]

# Neurokit2 to‘g‘ri ishlashi uchun taxminiy pastki chegara
MIN_URINISH_SONIYASI = 2.0

QRS_KENG_MS = 120.0
PR_UZOQ_MS = 200.0
QTC_UZOQ_MS = 460.0


def _indekslarni_tozala(qiymat: Any) -> List[int]:
    """neurokit2 to‘lqin indekslarini JSON-ga yaroqli int ro‘yxatga aylantiradi.

    Args:
        qiymat: Series, massiv yoki None.

    Returns:
        Finite musbat indekslar. Chizma va interval uchun.
    """
    if qiymat is None:
        return []
    try:
        arr = np.asarray(qiymat, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return []
    chiq: List[int] = []
    for x in arr:
        if np.isfinite(x) and x >= 0:
            chiq.append(int(x))
    return chiq


def _xato_javob(xabar: str, **qoshimcha: Any) -> Dict[str, Any]:
    """Muvaffaqiyatsiz EKG tahlili uchun bir xil javob shaklini qaytaradi.

    Args:
        xabar: Foydalanuvchiga tushunarli xato matni.
        **qoshimcha: Qisman hisoblangan maydonlar (masalan, tozalangan signal).

    Returns:
        ok=False bo‘lgan lug‘at. Dastur to‘xtamaydi, faqat xabar qaytadi.
    """
    javob: Dict[str, Any] = {
        "ok": False,
        "xabar": xabar,
        "yurak_chastotasi_bpm": None,
        "qrs_ms": None,
        "pr_ms": None,
        "qt_ms": None,
        "qtc_bazett_ms": None,
        "tasma": None,
        "urishlar_soni": 0,
        "hrv_sdnn_ms": None,
        "hrv_rmssd_ms": None,
        "tozalangan_signallar": None,
        "tolqinlar": {},
        "technician": None,
        "ep": None,
        "yetishmagan": [],
    }
    javob.update(qoshimcha)
    return javob


def namuna_12_tasma_ekg(
    sampling_rate: float = 500.0,
    davomiylik: float = 8.0,
    yurak_chastotasi: float = 72.0,
) -> np.ndarray:
    """NeuroKit2 (yoki Gaussian zaxira) bilan o‘qitish uchun 12 tasma yasaydi.

    Args:
        sampling_rate: Hz.
        davomiylik: Soniyada.
        yurak_chastotasi: bpm. Sinus ritmini tasdiqlamaydi.

    Returns:
        (n, 12) massiv. Haqiqiy bemor yozuvi emas.
    """
    sampling_rate = float(sampling_rate)
    n = int(sampling_rate * davomiylik)
    shkala = np.array([1.0, 1.15, 0.38, -0.55, 0.48, 0.72, 0.55, 0.88, 1.08, 1.22, 1.12, 0.95])
    rng = np.random.default_rng(7)
    if nk is not None:
        try:
            asos = np.asarray(
                nk.ecg_simulate(
                    duration=float(davomiylik),
                    sampling_rate=int(sampling_rate),
                    heart_rate=float(yurak_chastotasi),
                    noise=0.01,
                ),
                dtype=float,
            ).reshape(-1)
            if asos.size >= n:
                asos = asos[:n]
            else:
                asos = np.pad(asos, (0, n - asos.size))
            shovqin = 0.012 * rng.normal(size=(n, 12))
            return (asos[:, None] * shkala[None, :]) + shovqin
        except Exception:
            pass
    t = np.arange(n) / sampling_rate
    hr = float(yurak_chastotasi)
    rr = 60.0 / hr
    asos = np.zeros(n)
    for k in np.arange(0.35, davomiylik - 0.35, rr):
        asos += 0.15 * np.exp(-((t - (k - 0.16)) ** 2) / (2 * 0.010**2))
        asos += 1.35 * np.exp(-((t - k) ** 2) / (2 * 0.009**2))
        asos -= 0.22 * np.exp(-((t - (k + 0.04)) ** 2) / (2 * 0.012**2))
        asos += 0.32 * np.exp(-((t - (k + 0.22)) ** 2) / (2 * 0.045**2))
    asos += 0.015 * np.sin(2 * np.pi * 0.25 * t)
    shovqin = 0.01 * rng.normal(size=(n, 12))
    return (asos[:, None] * shkala[None, :]) + shovqin


def _12_tasma_ajrat(
    signal: Union[np.ndarray, Sequence[Any], Dict[str, Any]],
) -> Dict[str, Any]:
    """Kirishni 12 tasma nomli lug‘atga keltiradi (yetishmagan = nol).

    Args:
        signal: (n, 12) yoki (12, n) massiv, 12 ta ro‘yxat, yoki tasma nomli dict.

    Returns:
        tasmalar, yetishmagan. Shakl umuman yaroqsiz bo‘lsa ValueError.
    """
    if isinstance(signal, dict):
        tasmalar: Dict[str, np.ndarray] = {}
        yetishmagan: List[str] = []
        uzunlik: Optional[int] = None
        for nom in TASMA_NOMLARI:
            if nom in signal and signal[nom] is not None:
                arr = np.asarray(signal[nom], dtype=float).reshape(-1)
                tasmalar[nom] = arr
                uzunlik = arr.size if uzunlik is None else uzunlik
            else:
                yetishmagan.append(nom)
        if uzunlik is None:
            raise ValueError("EKG dict da hech bir tasma yo‘q.")
        for nom in yetishmagan:
            tasmalar[nom] = np.zeros(uzunlik, dtype=float)
        uzunliklar = {len(v) for v in tasmalar.values()}
        if len(uzunliklar) != 1:
            raise ValueError("Barcha EKG tasmalari bir xil uzunlikda bo‘lishi kerak.")
        return {"tasmalar": tasmalar, "yetishmagan": yetishmagan}

    massiv = np.asarray(signal, dtype=float)
    if massiv.ndim == 1:
        raise ValueError(
            "Bitta tasma kiritildi. 12 tasmali EKG uchun shakl (n, 12) yoki (12, n) bo‘lsin."
        )
    if massiv.ndim != 2:
        raise ValueError("EKG signali 2 o‘lchamli (namuna × tasma) bo‘lishi kerak.")

    if massiv.shape[0] == 12 and massiv.shape[1] != 12:
        massiv = massiv.T
    elif massiv.shape[1] == 12:
        pass
    elif massiv.shape[0] == 12:
        massiv = massiv.T
    else:
        raise ValueError(
            f"12 tasma kutilgan, olingan shakl {tuple(massiv.shape)}. "
            "Massiv (n, 12) yoki (12, n) bo‘lsin."
        )

    return {
        "tasmalar": {nom: massiv[:, i] for i, nom in enumerate(TASMA_NOMLARI)},
        "yetishmagan": [],
    }


def tasmalarni_tozala(
    tasmalar: Dict[str, np.ndarray],
    sampling_rate: float,
) -> Dict[str, Any]:
    """Har bir tasmani neurokit2 `ecg_clean` bilan filtrlaydi.

    Args:
        tasmalar: Nom → 1D signal.
        sampling_rate: Hz.

    Returns:
        tozalangan (nom → massiv|None), xatolar. Technician bosqichi.
    """
    tozalangan: Dict[str, Optional[np.ndarray]] = {}
    xatolar: List[str] = []
    for nom, tasma in tasmalar.items():
        try:
            if not np.isfinite(tasma).any() or np.allclose(tasma, 0.0, atol=1e-12):
                tozalangan[nom] = None
                xatolar.append(f"{nom}: bo‘sh/nol")
                continue
            if nk is None:
                tozalangan[nom] = np.asarray(tasma, dtype=float)
                continue
            tozalangan[nom] = nk.ecg_clean(
                tasma,
                sampling_rate=sampling_rate,
                method="neurokit",
            )
        except Exception as exc:
            tozalangan[nom] = None
            xatolar.append(f"{nom}: {exc}")
    return {"tozalangan": tozalangan, "xatolar": xatolar}


def tozalangan_matritsa(tozalangan: Dict[str, Optional[np.ndarray]]) -> Optional[np.ndarray]:
    """Tozalangan tasmalarni (n, 12) chizma massiviga yig‘adi.

    Args:
        tozalangan: Technician chiqishi.

    Returns:
        Massiv yoki hammasi None bo‘lsa None. UI grafigi uchun.
    """
    n = None
    for v in tozalangan.values():
        if v is not None:
            n = len(v)
            break
    if n is None:
        return None
    mat = np.zeros((n, 12), dtype=float)
    for i, nom in enumerate(TASMA_NOMLARI):
        v = tozalangan.get(nom)
        if v is not None and len(v) == n:
            mat[:, i] = v
    return mat


def _indeks_juft_ms(
    boshlar: Optional[Sequence[Any]],
    oxirlar: Optional[Sequence[Any]],
    sampling_rate: float,
) -> Optional[float]:
    """Ikki to‘lqin indeksi orasidagi median intervalni millisekundda beradi.

    Args:
        boshlar: Interval boshlanish indekslari (P onset, QRS onset va hokazo).
        oxirlar: Interval tugash indekslari.
        sampling_rate: Namuna olish tezligi (Hz).

    Returns:
        Median davomiylik (ms) yoki juftlik topilmasa None.
        PR/QRS/QT klinik o‘lchoviga yaqinlashtirish uchun ishlatiladi.
    """
    if boshlar is None or oxirlar is None:
        return None
    b_list = _indekslarni_tozala(boshlar)
    o_list = _indekslarni_tozala(oxirlar)
    if not b_list or not o_list:
        return None
    davomiyliklar: List[float] = []
    for bosh, oxir in zip(b_list, o_list):
        if oxir <= bosh:
            continue
        davomiyliklar.append((oxir - bosh) / sampling_rate * 1000.0)
    if not davomiyliklar:
        return None
    return float(np.median(davomiyliklar))


def _r_zaxira(tozalangan: np.ndarray, sampling_rate: float) -> np.ndarray:
    """neurokit2 R topmasa, masofa+balandlik bilan cho‘qqi qidiradi.

    Args:
        tozalangan: Bitta tasma.
        sampling_rate: Hz.

    Returns:
        Indekslar. Yomon/sintetik signalda zaxira.
    """
    if find_peaks is None:
        return np.array([], dtype=int)
    x = np.asarray(tozalangan, dtype=float)
    if x.size < 10:
        return np.array([], dtype=int)
    masofa = max(int(0.35 * sampling_rate), 1)
    chegara = float(np.percentile(np.abs(x), 90) * 0.35)
    choq, _ = find_peaks(x, distance=masofa, height=chegara)
    return choq.astype(int)


def _r_choqqilarni_top(tozalangan: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Tozalangan tasmasida R cho‘qqilarini (QRS markazi) qidiradi.

    Args:
        tozalangan: neurokit2 bilan tozalangan bitta EKG tasmasi.
        sampling_rate: Namuna olish tezligi (Hz).

    Returns:
        R cho‘qqi indekslari. Yurak tezligi va to‘lqin ajratish uchun.
    """
    if nk is not None:
        try:
            _, info = nk.ecg_peaks(tozalangan, sampling_rate=sampling_rate)
            choqqilar = np.asarray(info.get("ECG_R_Peaks", []), dtype=float)
            choqqilar = choqqilar[np.isfinite(choqqilar)]
            if choqqilar.size >= 2:
                return choqqilar.astype(int)
        except Exception:
            pass
    return _r_zaxira(tozalangan, sampling_rate)


def _tolqinlarni_ajrat(
    tozalangan: np.ndarray,
    r_choqqilar: np.ndarray,
    sampling_rate: float,
) -> Dict[str, Any]:
    """P, QRS va T nuqtalarini ajratadi (avval DWT, bo‘lmasa peak usuli).

    Args:
        tozalangan: Tozalangan EKG tasmasi.
        r_choqqilar: R cho‘qqi indekslari.
        sampling_rate: Namuna olish tezligi (Hz).

    Returns:
        neurokit2 to‘lqin lug‘ati. PR, QRS, QT ni o‘lchash uchun.
    """
    if nk is None:
        return {}
    try:
        _, waves = nk.ecg_delineate(
            tozalangan,
            r_choqqilar,
            sampling_rate=sampling_rate,
            method="dwt",
            show=False,
        )
        return waves or {}
    except Exception:
        try:
            _, waves = nk.ecg_delineate(
                tozalangan,
                r_choqqilar,
                sampling_rate=sampling_rate,
                method="peak",
                show=False,
            )
            return waves or {}
        except Exception:
            return {}


def _yurak_chastotasi(r_choqqilar: np.ndarray, sampling_rate: float) -> Optional[float]:
    """R–R oraliqlaridan o‘rtacha yurak urish chastotasini hisoblaydi.

    Args:
        r_choqqilar: R cho‘qqi indekslari.
        sampling_rate: Namuna olish tezligi (Hz).

    Returns:
        O‘rtacha urish/daqiqa (bpm) yoki RR yetarli bo‘lmasa None.
        Sinus ritmini tasdiqlamaydi.
    """
    if r_choqqilar.size < 2:
        return None
    rr_soniya = np.diff(r_choqqilar.astype(float)) / sampling_rate
    rr_soniya = rr_soniya[rr_soniya > 0]
    if rr_soniya.size == 0:
        return None
    return float(60.0 / np.mean(rr_soniya))


def _qtc_bazett_ms(qt_ms: Optional[float], hr_bpm: Optional[float]) -> Optional[float]:
    """Bazett QTc = QT / sqrt(RR). Tashxis emas.

    Args:
        qt_ms: QT millisekundda.
        hr_bpm: O‘rtacha HR.

    Returns:
        QTc ms yoki hisoblab bo‘lmasa None.
    """
    if qt_ms is None or hr_bpm is None or hr_bpm <= 0 or qt_ms <= 0:
        return None
    rr_s = 60.0 / hr_bpm
    if rr_s <= 0:
        return None
    return round(float(qt_ms) / np.sqrt(rr_s), 1)


def _hrv_ol(r_choqqilar: np.ndarray, sampling_rate: float) -> Dict[str, Optional[float]]:
    """R cho‘qqilaridan SDNN va RMSSD (HRV) ni hisoblaydi.

    Args:
        r_choqqilar: R indekslari.
        sampling_rate: Hz.

    Returns:
        hrv_sdnn_ms, hrv_rmssd_ms. Klinik tashxis emas.
    """
    if r_choqqilar.size < 3:
        return {"hrv_sdnn_ms": None, "hrv_rmssd_ms": None}
    if nk is not None:
        try:
            hrv = nk.hrv_time({"ECG_R_Peaks": r_choqqilar}, sampling_rate=sampling_rate, show=False)
            sdnn = hrv["HRV_SDNN"].iloc[0] if "HRV_SDNN" in hrv.columns else None
            rmssd = hrv["HRV_RMSSD"].iloc[0] if "HRV_RMSSD" in hrv.columns else None
            return {
                "hrv_sdnn_ms": None
                if sdnn is None or (isinstance(sdnn, float) and np.isnan(sdnn))
                else round(float(sdnn), 2),
                "hrv_rmssd_ms": None
                if rmssd is None or (isinstance(rmssd, float) and np.isnan(rmssd))
                else round(float(rmssd), 2),
            }
        except Exception:
            pass
    rr = np.diff(r_choqqilar.astype(float)) / sampling_rate * 1000.0
    if rr.size < 2:
        return {"hrv_sdnn_ms": None, "hrv_rmssd_ms": None}
    diff = np.diff(rr)
    return {
        "hrv_sdnn_ms": round(float(np.std(rr, ddof=1)), 2),
        "hrv_rmssd_ms": round(float(np.sqrt(np.mean(diff**2))), 2),
    }


def _sifat_bahosi(
    xom: np.ndarray,
    toza: np.ndarray,
    r_soni: int,
    davomiylik_s: float,
) -> str:
    """Technician uchun qisqa signal sifati (tashxis emas).

    Args:
        xom: Filtrlanmagan tasma.
        toza: Tozalangan tasma.
        r_soni: Topilgan R.
        davomiylik_s: Yozuv uzunligi.

    Returns:
        qisqa | shovqinli | yetarli.
    """
    if davomiylik_s < MIN_URINISH_SONIYASI:
        return "qisqa"
    if r_soni < 2:
        return "R noaniq"
    qoldiq = np.std(np.asarray(xom, dtype=float) - np.asarray(toza, dtype=float))
    signal_std = float(np.std(toza))
    if signal_std > 0 and qoldiq / signal_std > 0.85:
        return "shovqinli"
    return "yetarli"


def ecg_technician_tahlil(
    signal: Union[np.ndarray, Sequence[Any], Dict[str, Any]],
    sampling_rate: float = 500.0,
) -> Dict[str, Any]:
    """EKG technician: 12 tasmani tozalaydi, sifat va taxminiy HR.

    Args:
        signal: 12 tasma dict yoki (n, 12).
        sampling_rate: Hz.

    Returns:
        ok, tozalangan_signallar, urishlar, HR, sifat. Intervallar EP ga qoldiriladi.
    """
    if nk is None:
        return _xato_javob(
            "neurokit2 o‘rnatilmagan. `pip install neurokit2` qiling va qayta urinib ko‘ring."
        )
    try:
        if sampling_rate is None or float(sampling_rate) <= 0:
            return _xato_javob("sampling_rate musbat Hz bo‘lishi kerak (masalan, 500 yoki 1000).")
        sampling_rate = float(sampling_rate)
        ajrat = _12_tasma_ajrat(signal)
        tasmalar: Dict[str, np.ndarray] = ajrat["tasmalar"]
        yetishmagan: List[str] = ajrat["yetishmagan"]
        n_namuna = len(next(iter(tasmalar.values())))
        davomiylik = n_namuna / sampling_rate
        if n_namuna < int(sampling_rate * MIN_URINISH_SONIYASI):
            return _xato_javob(
                f"Signal juda qisqa ({davomiylik:.2f} s). "
                f"Kamida {MIN_URINISH_SONIYASI:.0f} soniya yozuv kerak.",
                yetishmagan=yetishmagan,
            )

        toz = tasmalarni_tozala(tasmalar, sampling_rate)
        tozalangan = toz["tozalangan"]
        tozalash_xatolari = toz["xatolar"]
        if all(v is None for v in tozalangan.values()):
            sabab = "; ".join(tozalash_xatolari) if tozalash_xatolari else "noma’lum"
            return _xato_javob(
                f"12 tasma ham tozalanmadi. {sabab}",
                tozalangan_signallar=tozalangan,
                yetishmagan=yetishmagan,
            )

        tasma_nomi = None
        r_choqqilar = np.array([], dtype=int)
        for nom in ASOSIY_TASMA_TARTIBI + TASMA_NOMLARI:
            sig = tozalangan.get(nom)
            if sig is None:
                continue
            try:
                r_choqqilar = _r_choqqilarni_top(sig, sampling_rate)
            except Exception:
                continue
            tasma_nomi = nom
            if r_choqqilar.size:
                break

        hr = _yurak_chastotasi(r_choqqilar, sampling_rate) if tasma_nomi else None
        asos_xom = tasmalar.get(tasma_nomi) if tasma_nomi else None
        asos_toza = tozalangan.get(tasma_nomi) if tasma_nomi else None
        sifat = "noma’lum"
        if asos_xom is not None and asos_toza is not None:
            sifat = _sifat_bahosi(asos_xom, asos_toza, int(r_choqqilar.size), davomiylik)

        izoh = (
            f"Technician: {tasma_nomi or '—'} da {int(r_choqqilar.size)} R, "
            f"sifat={sifat}, {davomiylik:.1f} s, {sampling_rate:g} Hz. "
            "Tozalash filtr; tashxis emas."
        )
        if yetishmagan:
            izoh += " Yetishmagan tasma: " + ", ".join(yetishmagan) + "."
        if tozalash_xatolari:
            izoh += " Xato: " + "; ".join(tozalash_xatolari) + "."

        technician = {
            "ok": True,
            "xabar": izoh,
            "tasma": tasma_nomi,
            "sifat": sifat,
            "davomiylik_s": round(davomiylik, 2),
            "sampling_rate": sampling_rate,
            "tozalangan_tasma_soni": sum(1 for v in tozalangan.values() if v is not None),
            "tozalash_xatolari": tozalash_xatolari,
            "yetishmagan": yetishmagan,
            "urishlar_soni": int(r_choqqilar.size),
            "yurak_chastotasi_bpm": None if hr is None else round(hr, 1),
        }
        return {
            "ok": True,
            "xabar": izoh,
            "yurak_chastotasi_bpm": technician["yurak_chastotasi_bpm"],
            "qrs_ms": None,
            "pr_ms": None,
            "qt_ms": None,
            "qtc_bazett_ms": None,
            "tasma": tasma_nomi,
            "urishlar_soni": int(r_choqqilar.size),
            "hrv_sdnn_ms": None,
            "hrv_rmssd_ms": None,
            "tozalangan_signallar": tozalangan,
            "tolqinlar": {"r_choqqilar": _indekslarni_tozala(r_choqqilar)},
            "technician": technician,
            "ep": None,
            "yetishmagan": yetishmagan,
        }
    except ValueError as exc:
        return _xato_javob(str(exc))
    except Exception as exc:
        return _xato_javob(f"EKG technician xatolik: {exc}.")


def electrophysiologist_tahlil(
    signal: Union[np.ndarray, Sequence[Any], Dict[str, Any]],
    sampling_rate: float = 500.0,
    tozalangan_signallar: Optional[Dict[str, Optional[np.ndarray]]] = None,
) -> Dict[str, Any]:
    """Elektrofiziolog: P/QRS/T, PR/QRS/QT/QTc, HRV. Tashxis qo‘ymaydi.

    Args:
        signal: Xom 12 tasma (tozalangan berilmasa).
        sampling_rate: Hz.
        tozalangan_signallar: Technician chiqishi bo‘lsa qayta tozalamaslik.

    Returns:
        Intervallar, to‘lqin indekslari, ehtiyotkor eslatmalar.
    """
    if nk is None:
        return _xato_javob(
            "neurokit2 o‘rnatilmagan. `pip install neurokit2` qiling va qayta urinib ko‘ring."
        )
    try:
        sampling_rate = float(sampling_rate)
        if tozalangan_signallar is None:
            tech = ecg_technician_tahlil(signal, sampling_rate=sampling_rate)
            if not tech.get("ok"):
                return tech
            tozalangan_signallar = tech.get("tozalangan_signallar")
        tozalangan = tozalangan_signallar or {}

        oxirgi_xato = ""
        for tasma_nomi in ASOSIY_TASMA_TARTIBI + TASMA_NOMLARI:
            tasma_signal = tozalangan.get(tasma_nomi)
            if tasma_signal is None:
                continue
            try:
                r_choqqilar = _r_choqqilarni_top(tasma_signal, sampling_rate)
            except Exception as exc:
                oxirgi_xato = str(exc)
                continue
            if r_choqqilar.size == 0:
                oxirgi_xato = "R cho‘qqilari topilmadi"
                continue

            hr = _yurak_chastotasi(r_choqqilar, sampling_rate)
            hrv = _hrv_ol(r_choqqilar, sampling_rate)
            waves = _tolqinlarni_ajrat(tasma_signal, r_choqqilar, sampling_rate)

            p_on = waves.get("ECG_P_Onsets")
            r_on = waves.get("ECG_R_Onsets")
            r_off = waves.get("ECG_R_Offsets")
            t_off = waves.get("ECG_T_Offsets")
            p_pk = waves.get("ECG_P_Peaks")
            t_pk = waves.get("ECG_T_Peaks")
            q_pk = waves.get("ECG_Q_Peaks")
            s_pk = waves.get("ECG_S_Peaks")

            pr_ms = _indeks_juft_ms(p_on, r_on, sampling_rate)
            qrs_ms = _indeks_juft_ms(r_on, r_off, sampling_rate)
            qt_ms = _indeks_juft_ms(r_on, t_off, sampling_rate)
            if qrs_ms is None:
                qrs_ms = _indeks_juft_ms(q_pk, s_pk, sampling_rate)
            elif qrs_ms < 40 or qrs_ms > 200:
                qrs_alt = _indeks_juft_ms(q_pk, s_pk, sampling_rate)
                if qrs_alt is not None and 40 <= qrs_alt <= 200:
                    qrs_ms = qrs_alt
            if pr_ms is None:
                pr_ms = _indeks_juft_ms(p_pk, r_choqqilar, sampling_rate)
            if qt_ms is None:
                qt_ms = _indeks_juft_ms(q_pk, t_off, sampling_rate)
            if qt_ms is None:
                qt_ms = _indeks_juft_ms(q_pk, t_pk, sampling_rate)
            elif qt_ms < 250 or qt_ms > 700:
                qt_alt = _indeks_juft_ms(q_pk, t_pk, sampling_rate)
                if qt_alt is not None and 250 <= qt_alt <= 700:
                    qt_ms = qt_alt

            qtc = _qtc_bazett_ms(qt_ms, hr)

            p_list = _indekslarni_tozala(p_pk)
            t_list = _indekslarni_tozala(t_pk)
            r_list = _indekslarni_tozala(r_choqqilar)
            tolqinlar = {
                "r_choqqilar": r_list,
                "p_choqqilar": p_list,
                "t_choqqilar": t_list,
                "p_boshlanish": _indekslarni_tozala(p_on),
                "qrs_boshlanish": _indekslarni_tozala(r_on),
                "qrs_tugash": _indekslarni_tozala(r_off),
                "t_tugash": _indekslarni_tozala(t_off),
            }

            yetishmagan_olchov = [
                nom
                for nom, qiymat in (("PR", pr_ms), ("QRS", qrs_ms), ("QT", qt_ms), ("HR", hr))
                if qiymat is None
            ]
            eslatma: List[str] = []
            if yetishmagan_olchov:
                eslatma.append("Qisman: " + ", ".join(yetishmagan_olchov) + " aniqlanmadi")
            if qrs_ms is not None and qrs_ms >= QRS_KENG_MS:
                eslatma.append("QRS ≥120 ms (kengayish ehtimoli, tashxis emas)")
            if pr_ms is not None and pr_ms > PR_UZOQ_MS:
                eslatma.append("PR >200 ms (uzayish ehtimoli, tashxis emas)")
            if qtc is not None and qtc >= QTC_UZOQ_MS:
                eslatma.append("QTc Bazett ≥460 ms (tekshirish, tashxis emas)")
            if r_choqqilar.size >= 3 and len(p_list) < max(1, r_choqqilar.size // 3):
                eslatma.append("P to‘lqin kam topildi — ritmni shifokor o‘qisin")
            izoh = (
                f"EP: tasma {tasma_nomi}, P/QRS/T ajratildi. "
                + ("; ".join(eslatma) if eslatma else "Intervallar taxminiy.")
                + " Klinik tashxis o‘rnini bosmaydi."
            )
            ep = {
                "ok": True,
                "xabar": izoh,
                "tasma": tasma_nomi,
                "yurak_chastotasi_bpm": None if hr is None else round(hr, 1),
                "qrs_ms": None if qrs_ms is None else round(qrs_ms, 1),
                "pr_ms": None if pr_ms is None else round(pr_ms, 1),
                "qt_ms": None if qt_ms is None else round(qt_ms, 1),
                "qtc_bazett_ms": qtc,
                "urishlar_soni": int(r_choqqilar.size),
                "p_soni": len(p_list),
                "t_soni": len(t_list),
                "hrv_sdnn_ms": hrv.get("hrv_sdnn_ms"),
                "hrv_rmssd_ms": hrv.get("hrv_rmssd_ms"),
                "eslatmalar": eslatma,
                "tolqinlar": tolqinlar,
            }
            return {
                "ok": True,
                "xabar": izoh,
                "yurak_chastotasi_bpm": ep["yurak_chastotasi_bpm"],
                "qrs_ms": ep["qrs_ms"],
                "pr_ms": ep["pr_ms"],
                "qt_ms": ep["qt_ms"],
                "qtc_bazett_ms": qtc,
                "tasma": tasma_nomi,
                "urishlar_soni": ep["urishlar_soni"],
                "hrv_sdnn_ms": ep["hrv_sdnn_ms"],
                "hrv_rmssd_ms": ep["hrv_rmssd_ms"],
                "tozalangan_signallar": tozalangan,
                "tolqinlar": tolqinlar,
                "technician": None,
                "ep": ep,
                "yetishmagan": [],
            }

        return _xato_javob(
            "Tozalash bor, lekin EP intervallari hisoblanmadi. "
            + (oxirgi_xato or "Signalni va sampling_rate ni tekshiring."),
            tozalangan_signallar=tozalangan,
        )
    except ValueError as exc:
        return _xato_javob(str(exc))
    except Exception as exc:
        return _xato_javob(f"EKG EP tahlilida xatolik: {exc}.")


def process_ecg_signal(
    signal: Union[np.ndarray, Sequence[Any], Dict[str, Any]],
    sampling_rate: float = 500.0,
) -> Dict[str, Any]:
    """Technician tozalash + EP intervallarini bitta javobda birlashtiradi.

    Args:
        signal: 12 tasma: dict (I..V6), yoki (n, 12) / (12, n) massiv.
        sampling_rate: Namuna olish tezligi, Hz (ko‘p klinik EKG da 500).

    Returns:
        ok, technician, ep, intervallar, tozalangan_signallar, tolqinlar.
        Bu o‘lchovlar tashxis emas.
    """
    tech = ecg_technician_tahlil(signal, sampling_rate=sampling_rate)
    if not tech.get("ok"):
        return tech
    ep_toliq = electrophysiologist_tahlil(
        signal,
        sampling_rate=sampling_rate,
        tozalangan_signallar=tech.get("tozalangan_signallar"),
    )
    if not ep_toliq.get("ok"):
        ep_toliq["technician"] = tech.get("technician")
        ep_toliq["tozalangan_signallar"] = tech.get("tozalangan_signallar")
        ep_toliq["yetishmagan"] = tech.get("yetishmagan") or []
        return ep_toliq
    ep_toliq["technician"] = tech.get("technician")
    ep_toliq["yetishmagan"] = tech.get("yetishmagan") or []
    ep_toliq["xabar"] = (tech.get("xabar") or "") + " | " + (ep_toliq.get("xabar") or "")
    return ep_toliq
