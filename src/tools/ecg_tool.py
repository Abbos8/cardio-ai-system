"""12 tasmali EKG ni tozalash va asosiy intervallarni o‘lchash (neurokit2).

Natija taxminiy o‘lchovdir, klinik tashxis o‘rnini bosmaydi.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

try:
    import neurokit2 as nk
except ImportError:  # Kutubxona o‘rnatilmagan bo‘lsa, chaqiriqda tushunarli xabar
    nk = None

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
        "tasma": None,
        "urishlar_soni": 0,
        "hrv_sdnn_ms": None,
        "hrv_rmssd_ms": None,
        "tozalangan_signallar": None,
    }
    javob.update(qoshimcha)
    return javob


def _muvaffaqiyat_javob(
    xabar: str,
    yurak_chastotasi_bpm: Optional[float],
    qrs_ms: Optional[float],
    pr_ms: Optional[float],
    qt_ms: Optional[float],
    tasma: Optional[str],
    urishlar_soni: int,
    tozalangan_signallar: Dict[str, Optional[np.ndarray]],
    hrv_sdnn_ms: Optional[float] = None,
    hrv_rmssd_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Muvaffaqiyatli (yoki qisman) EKG o‘lchovini bir xil shaklda qaytaradi.

    Args:
        xabar: Qisqa izoh; qiymatlar tashxis emasligi eslatiladi.
        yurak_chastotasi_bpm: O‘rtacha yurak urish chastotasi.
        qrs_ms: QRS kengligi (ms).
        pr_ms: PR intervali (ms).
        qt_ms: QT intervali (ms).
        tasma: Intervallar olingan tasma nomi.
        urishlar_soni: Topilgan R cho‘qqilari soni.
        tozalangan_signallar: 12 tasmaning tozalangan qiymatlari.

    Returns:
        ok=True bo‘lgan natija lug‘ati.
    """
    return {
        "ok": True,
        "xabar": xabar,
        "yurak_chastotasi_bpm": yurak_chastotasi_bpm,
        "qrs_ms": qrs_ms,
        "pr_ms": pr_ms,
        "qt_ms": qt_ms,
        "tasma": tasma,
        "urishlar_soni": urishlar_soni,
        "hrv_sdnn_ms": hrv_sdnn_ms,
        "hrv_rmssd_ms": hrv_rmssd_ms,
        "tozalangan_signallar": tozalangan_signallar,
    }


def _12_tasma_ajrat(
    signal: Union[np.ndarray, Sequence[Any], Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    """Kirishni 12 tasma nomli lug‘atga keltiradi.

    Args:
        signal: (n, 12) yoki (12, n) massiv, 12 ta ro‘yxat, yoki tasma nomli dict.

    Returns:
        Har bir tasma uchun 1D float massiv.

    Raises:
        ValueError: Shakl 12 tasmaga mos kelmasa.
    """
    if isinstance(signal, dict):
        tasmalar: Dict[str, np.ndarray] = {}
        for nom in TASMA_NOMLARI:
            if nom not in signal:
                raise ValueError(f"12 tasmali EKG da '{nom}' tasmasi yo‘q.")
            tasmalar[nom] = np.asarray(signal[nom], dtype=float).reshape(-1)
        uzunliklar = {len(v) for v in tasmalar.values()}
        if len(uzunliklar) != 1:
            raise ValueError("Barcha EKG tasmalari bir xil uzunlikda bo‘lishi kerak.")
        return tasmalar

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

    return {nom: massiv[:, i] for i, nom in enumerate(TASMA_NOMLARI)}


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
    if not boshlar or not oxirlar:
        return None
    davomiyliklar: List[float] = []
    for bosh, oxir in zip(boshlar, oxirlar):
        if bosh is None or oxir is None:
            continue
        try:
            b_idx = float(bosh)
            o_idx = float(oxir)
        except (TypeError, ValueError):
            continue
        if np.isnan(b_idx) or np.isnan(o_idx) or o_idx <= b_idx:
            continue
        davomiyliklar.append((o_idx - b_idx) / sampling_rate * 1000.0)
    if not davomiyliklar:
        return None
    return float(np.median(davomiyliklar))


def _r_choqqilarni_top(tozalangan: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Tozalangan tasmasida R cho‘qqilarini (QRS markazi) qidiradi.

    Args:
        tozalangan: neurokit2 bilan tozalangan bitta EKG tasmasi.
        sampling_rate: Namuna olish tezligi (Hz).

    Returns:
        R cho‘qqi indekslari. Yurak tezligi va to‘lqin ajratish uchun kerak.
    """
    _, info = nk.ecg_peaks(tozalangan, sampling_rate=sampling_rate)
    choqqilar = np.asarray(info.get("ECG_R_Peaks", []), dtype=float)
    choqqilar = choqqilar[np.isfinite(choqqilar)]
    return choqqilar.astype(int)


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
        _, waves = nk.ecg_delineate(
            tozalangan,
            r_choqqilar,
            sampling_rate=sampling_rate,
            method="peak",
            show=False,
        )
        return waves or {}


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
    try:
        hrv = nk.hrv_time({"ECG_R_Peaks": r_choqqilar}, sampling_rate=sampling_rate, show=False)
        sdnn = hrv["HRV_SDNN"].iloc[0] if "HRV_SDNN" in hrv.columns else None
        rmssd = hrv["HRV_RMSSD"].iloc[0] if "HRV_RMSSD" in hrv.columns else None
        return {
            "hrv_sdnn_ms": None if sdnn is None or (isinstance(sdnn, float) and np.isnan(sdnn)) else round(float(sdnn), 2),
            "hrv_rmssd_ms": None if rmssd is None or (isinstance(rmssd, float) and np.isnan(rmssd)) else round(float(rmssd), 2),
        }
    except Exception:
        rr = np.diff(r_choqqilar.astype(float)) / sampling_rate * 1000.0
        if rr.size < 2:
            return {"hrv_sdnn_ms": None, "hrv_rmssd_ms": None}
        diff = np.diff(rr)
        return {
            "hrv_sdnn_ms": round(float(np.std(rr, ddof=1)), 2),
            "hrv_rmssd_ms": round(float(np.sqrt(np.mean(diff**2))), 2),
        }


def _intervallarni_hisobla(
    tozalangan: np.ndarray,
    sampling_rate: float,
) -> Dict[str, Any]:
    """Bitta tozalangan tasmasidan HR, QRS, PR va QT ni oladi.

    Args:
        tozalangan: Asosiy tasma (odatda Lead II).
        sampling_rate: Namuna olish tezligi (Hz).

    Returns:
        O‘lchovlar va qisqa izoh. Qiymatlar tashxis emas.
    """
    r_choqqilar = _r_choqqilarni_top(tozalangan, sampling_rate)
    if r_choqqilar.size == 0:
        return {
            "yurak_chastotasi_bpm": None,
            "qrs_ms": None,
            "pr_ms": None,
            "qt_ms": None,
            "urishlar_soni": 0,
            "hrv_sdnn_ms": None,
            "hrv_rmssd_ms": None,
            "izoh": "R cho‘qqilari topilmadi. Signal qisqa, shovqinli yoki tasma noto‘g‘ri bo‘lishi mumkin.",
        }

    hr = _yurak_chastotasi(r_choqqilar, sampling_rate)
    hrv = _hrv_ol(r_choqqilar, sampling_rate)
    waves = _tolqinlarni_ajrat(tozalangan, r_choqqilar, sampling_rate)

    # PR: P boshlanishi → QRS boshlanishi
    pr_ms = _indeks_juft_ms(waves.get("ECG_P_Onsets"), waves.get("ECG_R_Onsets"), sampling_rate)
    # QRS: QRS boshlanishi → QRS tugashi
    qrs_ms = _indeks_juft_ms(waves.get("ECG_R_Onsets"), waves.get("ECG_R_Offsets"), sampling_rate)
    # QT: QRS boshlanishi → T tugashi
    qt_ms = _indeks_juft_ms(waves.get("ECG_R_Onsets"), waves.get("ECG_T_Offsets"), sampling_rate)

    # Ba’zi usullarda R onset/offset bo‘lmaydi — Q/S va T peak bilan zaxira
    if qrs_ms is None:
        qrs_ms = _indeks_juft_ms(waves.get("ECG_Q_Peaks"), waves.get("ECG_S_Peaks"), sampling_rate)
    if pr_ms is None:
        pr_ms = _indeks_juft_ms(waves.get("ECG_P_Peaks"), r_choqqilar, sampling_rate)
    if qt_ms is None:
        qt_ms = _indeks_juft_ms(waves.get("ECG_Q_Peaks"), waves.get("ECG_T_Offsets"), sampling_rate)
    if qt_ms is None:
        qt_ms = _indeks_juft_ms(waves.get("ECG_Q_Peaks"), waves.get("ECG_T_Peaks"), sampling_rate)

    yetishmagan = [
        nom
        for nom, qiymat in (("PR", pr_ms), ("QRS", qrs_ms), ("QT", qt_ms), ("HR", hr))
        if qiymat is None
    ]
    if yetishmagan:
        izoh = (
            "Qisman o‘lchov: "
            + ", ".join(yetishmagan)
            + " aniqlanmadi. Qiymatlar taxminiy, tashxis o‘rnini bosmaydi."
        )
    else:
        izoh = "Intervallar Lead asosida taxminiy hisoblandi. Klinik tashxis o‘rnini bosmaydi."

    return {
        "yurak_chastotasi_bpm": None if hr is None else round(hr, 1),
        "qrs_ms": None if qrs_ms is None else round(qrs_ms, 1),
        "pr_ms": None if pr_ms is None else round(pr_ms, 1),
        "qt_ms": None if qt_ms is None else round(qt_ms, 1),
        "urishlar_soni": int(r_choqqilar.size),
        "hrv_sdnn_ms": hrv.get("hrv_sdnn_ms"),
        "hrv_rmssd_ms": hrv.get("hrv_rmssd_ms"),
        "izoh": izoh,
    }


def process_ecg_signal(
    signal: Union[np.ndarray, Sequence[Any], Dict[str, Any]],
    sampling_rate: float = 500.0,
) -> Dict[str, Any]:
    """12 tasmali EKG ni tozalaydi va QRS, PR, QT hamda yurak tezligini hisoblaydi.

    Har bir tasma neurokit2 `ecg_clean` bilan filtrlanadi. Intervallar odatda
    Lead II dagi R cho‘qqilari va P/QRS/T nuqtalaridan olinadi.

    Args:
        signal: 12 tasma: dict (I..V6), yoki (n, 12) / (12, n) massiv.
        sampling_rate: Namuna olish tezligi, Hz (ko‘p klinik EKG da 500).

    Returns:
        Lug‘at: ok, xabar, yurak_chastotasi_bpm, qrs_ms, pr_ms, qt_ms,
        tasma, urishlar_soni, tozalangan_signallar.
        Xatoda dastur to‘xtamaydi — ok=False va tushunarli xabar qaytadi.
        Bu o‘lchovlar tashxis emas.
    """
    if nk is None:
        return _xato_javob(
            "neurokit2 o‘rnatilmagan. `pip install neurokit2` qiling va qayta urinib ko‘ring."
        )

    try:
        if sampling_rate is None or float(sampling_rate) <= 0:
            return _xato_javob(
                "sampling_rate musbat Hz bo‘lishi kerak (masalan, 500 yoki 1000)."
            )
        sampling_rate = float(sampling_rate)

        tasmalar = _12_tasma_ajrat(signal)
        n_namuna = len(next(iter(tasmalar.values())))
        if n_namuna < int(sampling_rate * MIN_URINISH_SONIYASI):
            return _xato_javob(
                f"Signal juda qisqa ({n_namuna / sampling_rate:.2f} s). "
                f"Kamida {MIN_URINISH_SONIYASI:.0f} soniya yozuv kerak."
            )

        tozalangan: Dict[str, Optional[np.ndarray]] = {}
        tozalash_xatolari: List[str] = []
        for nom, tasma in tasmalar.items():
            try:
                if not np.isfinite(tasma).any():
                    raise ValueError("faqat bo‘sh yoki NaN qiymatlar")
                tozalangan[nom] = nk.ecg_clean(
                    tasma,
                    sampling_rate=sampling_rate,
                    method="neurokit",
                )
            except Exception as exc:
                tozalangan[nom] = None
                tozalash_xatolari.append(f"{nom}: {exc}")

        if all(v is None for v in tozalangan.values()):
            sabab = "; ".join(tozalash_xatolari) if tozalash_xatolari else "noma’lum"
            return _xato_javob(
                f"12 tasma ham tozalanmadi. {sabab}",
                tozalangan_signallar=tozalangan,
            )

        oxirgi_xato = ""
        for tasma_nomi in ASOSIY_TASMA_TARTIBI + TASMA_NOMLARI:
            tasma_signal = tozalangan.get(tasma_nomi)
            if tasma_signal is None:
                continue
            try:
                olchov = _intervallarni_hisobla(tasma_signal, sampling_rate)
            except Exception as exc:
                oxirgi_xato = str(exc)
                continue

            qoshimcha = ""
            if tozalash_xatolari:
                qoshimcha = " Ba’zi tasmalar tozalanmadi: " + "; ".join(tozalash_xatolari) + "."

            return _muvaffaqiyat_javob(
                xabar=olchov["izoh"] + qoshimcha,
                yurak_chastotasi_bpm=olchov["yurak_chastotasi_bpm"],
                qrs_ms=olchov["qrs_ms"],
                pr_ms=olchov["pr_ms"],
                qt_ms=olchov["qt_ms"],
                tasma=tasma_nomi,
                urishlar_soni=olchov["urishlar_soni"],
                tozalangan_signallar=tozalangan,
                hrv_sdnn_ms=olchov.get("hrv_sdnn_ms"),
                hrv_rmssd_ms=olchov.get("hrv_rmssd_ms"),
            )

        return _xato_javob(
            "Tozalash bo‘ldi, lekin hech bir tasmasida QRS/PR/QT hisoblanmadi. "
            + (oxirgi_xato or "Signalni va sampling_rate ni tekshiring."),
            tozalangan_signallar=tozalangan,
        )
    except ValueError as exc:
        return _xato_javob(str(exc))
    except Exception as exc:
        return _xato_javob(
            f"EKG tahlilida kutilmagan xatolik: {exc}. "
            "Signal shakli (n, 12) va sampling_rate ni tekshiring."
        )
