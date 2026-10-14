"""Uchta sintetik bemor holati: sodda, murakkab, juda murakkab.

JSON fayl klinik matn va laboratoriyani beradi. EKG CSV va echo DICOM
shu fayldan yoziladi, shunda interfeys oynalari avtomatik to‘ladi.
Haqiqiy bemor yozuvi emas; tizimni tekshirish uchun.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from tools.ecg_tool import TASMA_NOMLARI, namuna_12_tasma_ekg
from tools.echo_view_model import ECHO_KORINISHLAR, yasama_echo_kadr

# data/sintetik/<id>.json — manba. CSV va DICOM shu modul yozadi.
SINTETIK_ILDIZ = Path(__file__).resolve().parents[2] / "data" / "sintetik"
HOLAT_IDLAR = ("sodda", "murakkab", "juda_murakkab")

# DICOM SeriesDescription — echo_view_model.tegdan_korinish kalitlariga mos
_TEG_MATN = {
    "A2C": "apical 2 chamber",
    "A4C": "apical 4 chamber",
    "A3C": "apical 3 chamber",
    "PLAX": "parasternal long axis",
    "PSAX-AV": "psax-av",
    "PSAX-MV": "psax mv",
    "PSAX-PM": "psax pm",
    "PSAX-APEX": "psax apex",
    "SUBCOSTAL-4C": "subcostal 4 chamber",
    "SUBCOSTAL-IVC": "subcostal ivc",
    "SSN": "suprasternal",
}

_ECHO_URUG = {"sodda": 11, "murakkab": 22, "juda_murakkab": 33}


def _uid(matn: str) -> str:
    """Barqaror DICOM UID (har safar o‘zgarmas).

    Args:
        matn: Holat va ko‘rinish kaliti.

    Returns:
        Raqam va nuqtadan iborat UID. Bemor identifikatori emas.
    """
    n = int(hashlib.sha256(matn.encode("utf-8")).hexdigest()[:15], 16)
    return f"1.2.826.0.1.3680043.10.511.{n}"


def holat_json_yoli(holat_id: str) -> Path:
    """Sintetik holat JSON faylining yo‘lini qaytaradi.

    Args:
        holat_id: sodda, murakkab yoki juda_murakkab.

    Returns:
        data/sintetik/<id>.json.
    """
    if holat_id not in HOLAT_IDLAR:
        raise ValueError(f"Noma’lum sintetik holat: {holat_id}. {', '.join(HOLAT_IDLAR)}")
    return SINTETIK_ILDIZ / f"{holat_id}.json"


def _json_oqish(holat_id: str) -> Dict[str, Any]:
    """JSON dan klinik maydonlarni o‘qiydi.

    Args:
        holat_id: Fayl nomi (kengaytmasiz).

    Returns:
        Tekshirilgan lug‘at. Haqiqiy bemor emas.
    """
    yol = holat_json_yoli(holat_id)
    if not yol.is_file():
        raise FileNotFoundError(f"Sintetik fayl yo‘q: {yol}")
    holat = json.loads(yol.read_text(encoding="utf-8"))
    if not isinstance(holat, dict):
        raise ValueError(f"{yol.name} obyekt bo‘lishi kerak.")
    kerak = ("yosh", "jins", "shikoyatlar", "anamnez", "dorilar", "laboratoriya", "echo_korinishlar")
    yetishmagan = [k for k in kerak if k not in holat]
    if yetishmagan:
        raise ValueError(f"{yol.name} da maydon yo‘q: {', '.join(yetishmagan)}")
    korinishlar = [str(k) for k in holat["echo_korinishlar"]]
    notanish = [k for k in korinishlar if k not in ECHO_KORINISHLAR]
    if notanish:
        raise ValueError(f"{yol.name} noma’lum echo ko‘rinish: {', '.join(notanish)}")
    if not korinishlar:
        raise ValueError(f"{yol.name} da kamida bitta echo ko‘rinish bo‘lsin.")
    lab = holat["laboratoriya"]
    if not isinstance(lab, dict) or not lab:
        raise ValueError(f"{yol.name} laboratoriya bo‘sh.")
    holat["id"] = holat_id
    holat["echo_korinishlar"] = korinishlar
    holat["sampling_rate"] = float(holat.get("sampling_rate") or 500)
    holat["ekg_davomiylik"] = float(holat.get("ekg_davomiylik") or 8)
    holat["yurak_chastotasi"] = float(holat.get("yurak_chastotasi") or 72)
    return holat


def _ekg_signal(holat: Dict[str, Any]) -> np.ndarray:
    """Holatdagi yurak chastotasiga mos 12 tasmali sintetik EKG yasaydi.

    Args:
        holat: sampling_rate, ekg_davomiylik, yurak_chastotasi, id.

    Returns:
        (n, 12) float massiv. Haqiqiy yozuv emas.
    """
    return namuna_12_tasma_ekg(
        sampling_rate=float(holat["sampling_rate"]),
        davomiylik=float(holat["ekg_davomiylik"]),
        yurak_chastotasi=float(holat["yurak_chastotasi"]),
    )


def _ekg_csv(signal: np.ndarray, sampling_rate: float, izoh: str) -> bytes:
    """12 tasma CSV (boshida sampling_rate izohi) yasaydi.

    Args:
        signal: (n, 12).
        sampling_rate: Hz.
        izoh: Fayl sharhi. Tashxis emas.

    Returns:
        UTF-8 bayt. ekg_yuklash.csv_dan_ekg o‘qiy oladi.
    """
    qatorlar = [
        f"# sampling_rate={float(sampling_rate):g}",
        f"# {izoh}",
        ",".join(TASMA_NOMLARI),
    ]
    for qator in np.asarray(signal, dtype=float):
        qatorlar.append(",".join(f"{float(x):.6f}" for x in qator))
    return ("\n".join(qatorlar) + "\n").encode("utf-8")


def _dicom_bayt(holat_id: str, korinish: str, kadr: np.ndarray) -> bytes:
    """Bitta sintetik kadrni ultrasound DICOM baytiga yozadi.

    Args:
        holat_id: Bemor holati (PatientName sintetik).
        korinish: 11 standart yorliqdan biri.
        kadr: (H, W) uint8.

    Returns:
        DICOM bayt. SeriesDescription ko‘rinishni beradi. Haqiqiy bemor emas.
    """
    try:
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian
    except ImportError as exc:
        raise ImportError("Sintetik echo uchun pydicom kerak.") from exc

    rasm = np.ascontiguousarray(kadr, dtype=np.uint8)
    if rasm.ndim != 2:
        raise ValueError(f"Echo kadr 2D bo‘lishi kerak, shakl {rasm.shape}")
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.6.1"
    file_meta.MediaStorageSOPInstanceUID = _uid(f"{holat_id}:{korinish}:sop")
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = _uid("cardio-ai-sintetik")

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientName = f"SINTETIK^{holat_id.upper()}"
    ds.PatientID = f"SYN-{holat_id}"
    ds.Modality = "US"
    ds.SeriesDescription = _TEG_MATN[korinish]
    ds.StudyDescription = "Sintetik echo (haqiqiy bemor emas)"
    ds.ImageComments = "Sintetik kadr. Klinik tashxis emas."
    ds.StudyDate = "20260101"
    ds.ContentDate = "20260101"
    ds.ContentTime = "000000"
    ds.Rows = int(rasm.shape[0])
    ds.Columns = int(rasm.shape[1])
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = rasm.tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, implicit_vr=False, little_endian=True, enforce_file_format=True)
    return buf.getvalue()


def _echo_elementlar(holat: Dict[str, Any]) -> List[Dict[str, Any]]:
    """JSON dagi ko‘rinishlar uchun sintetik DICOM ro‘yxatini yasaydi.

    Args:
        holat: id va echo_korinishlar.

    Returns:
        nom, bayt, korinish, kadr. Ko‘rinish oynasini to‘ldirish uchun.
    """
    rng = np.random.default_rng(_ECHO_URUG.get(str(holat["id"]), 7))
    chiq: List[Dict[str, Any]] = []
    for korinish in holat["echo_korinishlar"]:
        kadr = yasama_echo_kadr(korinish, rng)
        nom = f"{korinish}.dcm"
        chiq.append(
            {
                "nom": nom,
                "bayt": _dicom_bayt(str(holat["id"]), korinish, kadr),
                "korinish": korinish,
                "kadr": kadr,
            }
        )
    return chiq


def _fayllarga_yoz(holat_id: str, ekg_bayt: bytes, echo: List[Dict[str, Any]]) -> Dict[str, Any]:
    """EKG CSV va echo DICOM ni data/sintetik/<id>/ ostiga yozadi.

    Args:
        holat_id: Papka nomi.
        ekg_bayt: CSV.
        echo: nom va bayt.

    Returns:
        json, ekg va echo yo‘llari. Tekshirish uchun ochiq fayllar.
    """
    papka = SINTETIK_ILDIZ / holat_id
    echo_papka = papka / "echo"
    echo_papka.mkdir(parents=True, exist_ok=True)
    ekg_yol = papka / "ekg.csv"
    ekg_yol.write_bytes(ekg_bayt)
    echo_yollar: List[str] = []
    for element in echo:
        yol = echo_papka / str(element["nom"])
        yol.write_bytes(element["bayt"])
        echo_yollar.append(str(yol))
    return {
        "json": str(holat_json_yoli(holat_id)),
        "ekg": str(ekg_yol),
        "echo": echo_yollar,
    }


def sintetik_yukla(holat_id: str) -> Dict[str, Any]:
    """Bitta sintetik holatni JSON dan o‘qib, EKG va echo bilan qaytaradi.

    Args:
        holat_id: sodda, murakkab yoki juda_murakkab.

    Returns:
        Forma maydonlari, signal (n, 12), echo_fayllar va fayl yo‘llari.
        Klinik tashxis emas; loyihani solishtirib ko‘rish uchun.
    """
    holat = _json_oqish(holat_id)
    signal = _ekg_signal(holat)
    hz = float(holat["sampling_rate"])
    izoh = (
        f"Sintetik EKG ({holat_id}), HR sozlama {holat['yurak_chastotasi']:g} bpm. "
        "Haqiqiy yozuv emas."
    )
    echo = _echo_elementlar(holat)
    fayllar = _fayllarga_yoz(holat_id, _ekg_csv(signal, hz, izoh), echo)
    return {
        "id": holat_id,
        "sarlavha": str(holat.get("sarlavha") or holat_id),
        "izoh": str(holat.get("izoh") or ""),
        "kuzatish": str(holat.get("kuzatish") or ""),
        "yosh": int(holat["yosh"]),
        "jins": str(holat["jins"]),
        "shikoyatlar": str(holat["shikoyatlar"]),
        "anamnez": str(holat["anamnez"]),
        "dorilar": str(holat["dorilar"]),
        "laboratoriya": {str(k): float(v) for k, v in holat["laboratoriya"].items()},
        "sampling_rate": hz,
        "yurak_chastotasi": float(holat["yurak_chastotasi"]),
        "signal": signal,
        "ekg_xabar": izoh + f" Fayl: {fayllar['ekg']}",
        "echo_fayllar": echo,
        "fayllar": fayllar,
    }
