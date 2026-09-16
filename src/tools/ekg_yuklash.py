"""12 tasmali EKG ni CSV yoki WFDB (.hea/.dat) dan o‘qiydi.

Natija tashxis emas; faqat signal va sampling_rate.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ecg_tool.TASMA_NOMLARI bilan bir xil tartib (siklik import yo‘q)
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

# Fayl sarlavhalaridagi tasma nomlari
TASMA_SINONIM: Dict[str, str] = {
    "i": "I",
    "lead i": "I",
    "lead_i": "I",
    "ii": "II",
    "lead ii": "II",
    "lead_ii": "II",
    "mlii": "II",
    "ml2": "II",
    "iii": "III",
    "lead iii": "III",
    "avr": "aVR",
    "avl": "aVL",
    "avf": "aVF",
    "v1": "V1",
    "v2": "V2",
    "v3": "V3",
    "v4": "V4",
    "v5": "V5",
    "v6": "V6",
}

VAQT_SARLAVHA = {"time", "t", "sec", "second", "seconds", "ms", "timestamp", "vaqt"}


def _tasma_nomi(matn: str) -> Optional[str]:
    """Ustun sarlavhasini I..V6 ga moslaydi.

    Args:
        matn: CSV/WFDB ustun nomi.

    Returns:
        Standart tasma yoki None.
    """
    s = re.sub(r"\s+", " ", str(matn).strip().lower())
    s = s.replace("_", " ")
    s = s.replace("lead", "").strip()
    s = s.replace("ekg", "").replace("ecg", "").strip()
    kalit = s.replace(" ", "")
    if kalit in TASMA_SINONIM:
        return TASMA_SINONIM[kalit]
    if s in TASMA_SINONIM:
        return TASMA_SINONIM[s]
    return TASMA_SINONIM.get(kalit)


def _sharhdan_hz(qatorlar: Sequence[str]) -> Optional[float]:
    """CSV boshidagi # sampling_rate=500 kabi izohlardan Hz ni oladi.

    Args:
        qatorlar: Fayl satrlari.

    Returns:
        Hz yoki None.
    """
    for qator in qatorlar[:20]:
        if not qator.lstrip().startswith("#"):
            continue
        top = re.search(
            r"(?:sampling[_ ]?rate|fs|hz)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)",
            qator,
            flags=re.I,
        )
        if top:
            hz = float(top.group(1))
            if hz > 0:
                return hz
    return None


def csv_dan_ekg(
    fayl_bayt: bytes,
    sampling_rate_hint: Optional[float] = None,
) -> Dict[str, Any]:
    """CSV baytidan 12 tasma (yoki qisman) EKG o‘qiydi.

    Args:
        fayl_bayt: Yuklangan CSV.
        sampling_rate_hint: Formadagi Hz; vaqt ustuni bo‘lmasa ishlatiladi.

    Returns:
        ok, signal (n, 12), sampling_rate, yetishmagan, xabar.
    """
    try:
        matn = fayl_bayt.decode("utf-8-sig")
    except UnicodeDecodeError:
        matn = fayl_bayt.decode("latin-1")
    satrlar = matn.splitlines()
    hz_sharh = _sharhdan_hz(satrlar)
    toza = [s for s in satrlar if s.strip() and not s.lstrip().startswith("#")]
    if not toza:
        return {"ok": False, "signal": None, "sampling_rate": None, "xabar": "CSV bo‘sh.", "yetishmagan": []}

    oquvchi = csv.reader(io.StringIO("\n".join(toza)))
    qatorlar = [[c.strip() for c in q] for q in oquvchi if any(x.strip() for x in q)]
    if not qatorlar:
        return {"ok": False, "signal": None, "sampling_rate": None, "xabar": "CSV da qator yo‘q.", "yetishmagan": []}

    sarlavha = qatorlar[0]
    tasma_idx: Dict[str, int] = {}
    vaqt_idx: Optional[int] = None
    for i, nom in enumerate(sarlavha):
        past = nom.strip().lower()
        if past in VAQT_SARLAVHA:
            vaqt_idx = i
            continue
        tasma = _tasma_nomi(nom)
        if tasma and tasma not in tasma_idx:
            tasma_idx[tasma] = i

    son_qatorlar = qatorlar
    if tasma_idx:
        son_qatorlar = qatorlar[1:]
    elif vaqt_idx is not None:
        son_qatorlar = qatorlar[1:]

    if not tasma_idx:
        # Sarlavhasiz 12 ustun
        sonlar: List[List[float]] = []
        for qator in son_qatorlar:
            if len(qator) < 12:
                continue
            try:
                sonlar.append([float(x) for x in qator[:12]])
            except ValueError:
                continue
        if not sonlar:
            return {
                "ok": False,
                "signal": None,
                "sampling_rate": None,
                "yetishmagan": list(TASMA_NOMLARI),
                "xabar": "CSV da 12 ustunli sonlar yoki I..V6 sarlavha yo‘q.",
            }
        massiv = np.asarray(sonlar, dtype=float)
        hz = hz_sharh or sampling_rate_hint or 500.0
        return {
            "ok": True,
            "signal": massiv,
            "sampling_rate": float(hz),
            "yetishmagan": [],
            "xabar": f"12 ustunli CSV o‘qildi ({massiv.shape[0]} namuna, {hz:g} Hz).",
        }

    n = len(son_qatorlar)
    massiv = np.zeros((n, 12), dtype=float)
    yozildi = np.zeros(n, dtype=bool)
    vaqt: List[float] = []
    for r, qator in enumerate(son_qatorlar):
        qator_ok = False
        for tasma, idx in tasma_idx.items():
            if idx >= len(qator):
                continue
            try:
                massiv[r, TASMA_NOMLARI.index(tasma)] = float(qator[idx])
                qator_ok = True
            except ValueError:
                continue
        if vaqt_idx is not None and vaqt_idx < len(qator):
            try:
                vaqt.append(float(qator[vaqt_idx]))
                qator_ok = True
            except ValueError:
                vaqt.append(np.nan)
        yozildi[r] = qator_ok
    if not yozildi.any():
        return {
            "ok": False,
            "signal": None,
            "sampling_rate": None,
            "yetishmagan": [n for n in TASMA_NOMLARI if n not in tasma_idx],
            "xabar": "Sarlavhali CSV da sonli qator topilmadi.",
        }
    massiv = massiv[yozildi]
    if vaqt:
        vaqt_arr = np.asarray(vaqt, dtype=float)[yozildi]
        if np.isfinite(vaqt_arr).sum() >= 2:
            dt = np.diff(vaqt_arr[np.isfinite(vaqt_arr)])
            dt = dt[dt > 0]
            if dt.size:
                median_dt = float(np.median(dt))
                # ms da bo‘lsa
                if median_dt > 0.05:
                    median_dt = median_dt / 1000.0
                if median_dt > 0:
                    hz_sharh = hz_sharh or (1.0 / median_dt)

    hz = hz_sharh or sampling_rate_hint or 500.0
    yetishmagan = [n for n in TASMA_NOMLARI if n not in tasma_idx]
    izoh = f"CSV: {len(tasma_idx)} tasma, {massiv.shape[0]} namuna, {hz:g} Hz."
    if yetishmagan:
        izoh += " Yetishmaydi: " + ", ".join(yetishmagan) + " (nol bilan to‘ldirildi)."
    return {
        "ok": True,
        "signal": massiv,
        "sampling_rate": float(hz),
        "yetishmagan": yetishmagan,
        "xabar": izoh,
    }


def _hea_qatorini_ajrat(qator: str) -> List[str]:
    """WFDB .hea satrini bo‘laklarga ajratadi.

    Args:
        qator: Header satri.

    Returns:
        Tokenlar.
    """
    return qator.strip().split()


def wfdb_dan_ekg(hea_bayt: bytes, dat_bayt: bytes) -> Dict[str, Any]:
    """WFDB .hea + .dat (format 16 yoki 212) dan EKG o‘qiydi.

    Args:
        hea_bayt: Header matni.
        dat_bayt: Signal baytlari.

    Returns:
        csv_dan_ekg bilan bir xil shakl. 12 tasma bo‘lmasa qisman to‘ldiriladi.
    """
    try:
        matn = hea_bayt.decode("ascii", errors="replace")
    except Exception:
        matn = hea_bayt.decode("latin-1", errors="replace")
    qatorlar = [q for q in matn.splitlines() if q.strip() and not q.startswith("#")]
    if not qatorlar:
        return {"ok": False, "signal": None, "sampling_rate": None, "yetishmagan": [], "xabar": "WFDB .hea bo‘sh."}

    bosh = _hea_qatorini_ajrat(qatorlar[0])
    if len(bosh) < 4:
        return {
            "ok": False,
            "signal": None,
            "sampling_rate": None,
            "yetishmagan": [],
            "xabar": "WFDB .hea birinchi qatori: nom nsig fs nsamp.",
        }
    try:
        nsig = int(bosh[1])
        fs = float(bosh[2])
        nsamp = int(float(bosh[3]))
    except ValueError:
        return {
            "ok": False,
            "signal": None,
            "sampling_rate": None,
            "yetishmagan": [],
            "xabar": "WFDB .hea da nsig/fs/nsamp o‘qilmadi.",
        }
    if nsig <= 0 or fs <= 0 or nsamp <= 0:
        return {
            "ok": False,
            "signal": None,
            "sampling_rate": None,
            "yetishmagan": [],
            "xabar": "WFDB .hea qiymatlari musbat bo‘lsin.",
        }

    tasmalar: List[Dict[str, Any]] = []
    for qator in qatorlar[1 : 1 + nsig]:
        tok = _hea_qatorini_ajrat(qator)
        if len(tok) < 2:
            continue
        fmt = tok[1]
        gain = 200.0
        baseline = 0.0
        nom = tok[-1] if tok else f"s{len(tasmalar)}"
        try:
            if len(tok) >= 3:
                gain_qism = tok[2].split("/")[0]
                gain = float(gain_qism) if gain_qism not in {"", "0"} else 200.0
            if len(tok) >= 5:
                baseline = float(tok[4])
        except ValueError:
            pass
        tasma = _tasma_nomi(nom) or nom
        tasmalar.append({"fmt": fmt, "gain": gain if gain else 200.0, "baseline": baseline, "nom": tasma})

    if len(tasmalar) != nsig:
        return {
            "ok": False,
            "signal": None,
            "sampling_rate": None,
            "yetishmagan": [],
            "xabar": f"WFDB: {nsig} tasma kutilgan, header da {len(tasmalar)}.",
        }

    fmt0 = str(tasmalar[0]["fmt"])
    try:
        if fmt0 == "16":
            xom = np.frombuffer(dat_bayt, dtype="<i2")
            kerak = nsamp * nsig
            if xom.size < kerak:
                nsamp = xom.size // nsig
                xom = xom[: nsamp * nsig]
            else:
                xom = xom[:kerak]
            mat = xom.reshape(nsamp, nsig).astype(float)
        elif fmt0 == "212":
            mat = _format_212_oq(dat_bayt, nsamp, nsig)
        else:
            return {
                "ok": False,
                "signal": None,
                "sampling_rate": None,
                "yetishmagan": [],
                "xabar": f"WFDB format {fmt0} qo‘llab-quvvatlanmaydi (16 yoki 212).",
            }
    except Exception as exc:
        return {
            "ok": False,
            "signal": None,
            "sampling_rate": None,
            "yetishmagan": [],
            "xabar": f"WFDB .dat o‘qilmadi: {exc}",
        }

    for i, t in enumerate(tasmalar):
        mat[:, i] = (mat[:, i] - t["baseline"]) / t["gain"]

    chiq = np.zeros((mat.shape[0], 12), dtype=float)
    topilgan: List[str] = []
    for i, t in enumerate(tasmalar):
        nom = t["nom"]
        if nom in TASMA_NOMLARI:
            chiq[:, TASMA_NOMLARI.index(nom)] = mat[:, i]
            topilgan.append(nom)
        elif nsig == 12:
            chiq[:, i] = mat[:, i]
            topilgan.append(TASMA_NOMLARI[i])

    if nsig == 12 and not any(n in TASMA_NOMLARI for n in (t["nom"] for t in tasmalar)):
        topilgan = list(TASMA_NOMLARI)

    yetishmagan = [n for n in TASMA_NOMLARI if n not in topilgan]
    if not topilgan and nsig != 12:
        return {
            "ok": False,
            "signal": None,
            "sampling_rate": None,
            "yetishmagan": list(TASMA_NOMLARI),
            "xabar": "WFDB tasmalari I..V6 ga mos kelmadi.",
        }
    izoh = f"WFDB: {len(topilgan) or nsig} tasma, {chiq.shape[0]} namuna, {fs:g} Hz."
    if yetishmagan:
        izoh += " Yetishmaydi: " + ", ".join(yetishmagan) + "."
    return {
        "ok": True,
        "signal": chiq,
        "sampling_rate": float(fs),
        "yetishmagan": yetishmagan if topilgan else [],
        "xabar": izoh,
    }


def _format_212_oq(dat_bayt: bytes, nsamp: int, nsig: int) -> np.ndarray:
    """WFDB 212 (12-bit juft) formatini float matritsaga aylantiradi.

    Args:
        dat_bayt: Xom .dat.
        nsamp: Namuna soni.
        nsig: Tasma soni (odatda 2).

    Returns:
        (nsamp, nsig) massiv.
    """
    if nsig != 2:
        raise ValueError("Format 212 odatda 2 tasma uchun; 12 tasmali yozuvda format 16 ishlating.")
    # 3 bayt = 2 ta 12-bit namuna (ikki tasma bir vaqtda)
    n_juft = nsamp
    kerak_bayt = int(np.ceil(n_juft * 3 / 1))
    buf = np.frombuffer(dat_bayt[:kerak_bayt], dtype=np.uint8)
    if buf.size < 3:
        raise ValueError("Format 212 fayl juda qisqa.")
    n_triple = buf.size // 3
    buf = buf[: n_triple * 3].reshape(n_triple, 3)
    a = buf[:, 0].astype(np.int32) + ((buf[:, 1].astype(np.int32) & 0x0F) << 8)
    b = buf[:, 2].astype(np.int32) + ((buf[:, 1].astype(np.int32) & 0xF0) << 4)
    a = np.where(a >= 2048, a - 4096, a)
    b = np.where(b >= 2048, b - 4096, b)
    n = min(nsamp, a.size)
    mat = np.column_stack([a[:n], b[:n]]).astype(float)
    return mat


def ekg_fayllardan_oqish(
    fayllar: Sequence[Tuple[str, bytes]],
    sampling_rate_hint: Optional[float] = None,
) -> Dict[str, Any]:
    """Bir yoki bir nechta yuklangan fayldan EKG yig‘adi (CSV yoki WFDB).

    Args:
        fayllar: (fayl_nomi, bayt) juftliklari.
        sampling_rate_hint: CSV da Hz yo‘q bo‘lsa.

    Returns:
        ok, signal, sampling_rate, yetishmagan, xabar.
    """
    if not fayllar:
        return {"ok": False, "signal": None, "sampling_rate": None, "yetishmagan": [], "xabar": "Fayl yuklanmadi."}

    nomlar = {nom.lower(): bayt for nom, bayt in fayllar}
    hea = next((b for n, b in nomlar.items() if n.endswith(".hea")), None)
    dat = next((b for n, b in nomlar.items() if n.endswith(".dat")), None)
    if hea is not None and dat is not None:
        return wfdb_dan_ekg(hea, dat)
    if hea is not None and dat is None:
        return {
            "ok": False,
            "signal": None,
            "sampling_rate": None,
            "yetishmagan": [],
            "xabar": "WFDB uchun .hea bilan birga .dat ham yuklang.",
        }
    if dat is not None and hea is None:
        return {
            "ok": False,
            "signal": None,
            "sampling_rate": None,
            "yetishmagan": [],
            "xabar": "WFDB uchun .dat bilan birga .hea ham yuklang.",
        }

    csv_bayt = None
    for n, b in nomlar.items():
        if n.endswith(".csv") or n.endswith(".txt"):
            csv_bayt = b
            break
    if csv_bayt is None and len(fayllar) == 1:
        csv_bayt = fayllar[0][1]
    if csv_bayt is None:
        return {
            "ok": False,
            "signal": None,
            "sampling_rate": None,
            "yetishmagan": [],
            "xabar": "CSV (.csv) yoki WFDB (.hea + .dat) yuklang.",
        }
    return csv_dan_ekg(csv_bayt, sampling_rate_hint=sampling_rate_hint)
