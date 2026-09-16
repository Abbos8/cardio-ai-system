"""Echo DICOM, video va rasm kadrlarini o‘qiydi.

Natija tashxis emas; faqat kadrlar va meta.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
except ImportError:
    pydicom = None
    apply_modality_lut = None
    apply_voi_lut = None

try:
    import cv2
except ImportError:
    cv2 = None


def _kulrang(kadr: np.ndarray) -> np.ndarray:
    """RGB/RGBA ni 2D kulrang kadrga aylantiradi.

    Args:
        kadr: 2D yoki 3D massiv.

    Returns:
        uint8 (H, W).
    """
    arr = np.asarray(kadr)
    if arr.ndim == 3:
        if arr.shape[-1] >= 3:
            arr = arr[..., :3].astype(float).mean(axis=-1)
        else:
            arr = arr[..., 0]
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Kadr 2D bo‘lishi kerak, shakl {arr.shape}")
    if arr.dtype != np.uint8:
        a_min, a_max = float(np.min(arr)), float(np.max(arr))
        if a_max > a_min:
            arr = (255.0 * (arr - a_min) / (a_max - a_min)).clip(0, 255)
        else:
            arr = np.zeros_like(arr, dtype=float)
        arr = arr.astype(np.uint8)
    return arr


def _dicom_meta(ds: Any) -> Dict[str, str]:
    """DICOM teglaridan ko‘rinish uchun matn yig‘adi.

    Args:
        ds: pydicom dataset.

    Returns:
        series, protocol, view, manufacturer.
    """

    def ol(nom: str) -> str:
        qiymat = getattr(ds, nom, "") or ""
        return str(qiymat).strip()

    return {
        "series_description": ol("SeriesDescription"),
        "protocol_name": ol("ProtocolName"),
        "view_name": ol("ViewName") or ol("ViewPosition"),
        "image_comments": ol("ImageComments"),
        "manufacturer": ol("Manufacturer"),
        "modality": ol("Modality"),
    }


def dicom_dan_kadrlar(bayt: bytes) -> Dict[str, Any]:
    """Ultrasound/DICOM baytidan kadrlar va meta oladi.

    Args:
        bayt: .dcm fayl.

    Returns:
        ok, kadrlar (kulrang), meta, xabar.
    """
    if pydicom is None:
        return {"ok": False, "kadrlar": [], "meta": {}, "xabar": "pydicom o‘rnatilmagan."}
    try:
        ds = pydicom.dcmread(io.BytesIO(bayt), force=True)
    except Exception as exc:
        return {"ok": False, "kadrlar": [], "meta": {}, "xabar": f"DICOM o‘qilmadi: {exc}"}
    meta = _dicom_meta(ds)
    if not hasattr(ds, "PixelData") or ds.get("PixelData") is None:
        return {
            "ok": False,
            "kadrlar": [],
            "meta": meta,
            "xabar": "DICOM da PixelData yo‘q (faqat meta).",
        }
    try:
        pix = ds.pixel_array
        if apply_voi_lut is not None:
            try:
                pix = apply_voi_lut(apply_modality_lut(pix, ds) if apply_modality_lut else pix, ds)
            except Exception:
                pass
    except Exception as exc:
        return {"ok": False, "kadrlar": [], "meta": meta, "xabar": f"PixelData ochilmadi: {exc}"}

    kadrlar: List[np.ndarray] = []
    if pix.ndim == 2:
        kadrlar.append(_kulrang(pix))
    elif pix.ndim == 3:
        # (frames, H, W) yoki (H, W, C)
        if pix.shape[-1] in (1, 3, 4) and pix.shape[0] > 8 and pix.shape[1] > 8:
            kadrlar.append(_kulrang(pix))
        else:
            for i in range(pix.shape[0]):
                kadrlar.append(_kulrang(pix[i]))
    elif pix.ndim == 4:
        for i in range(pix.shape[0]):
            kadrlar.append(_kulrang(pix[i]))
    else:
        return {"ok": False, "kadrlar": [], "meta": meta, "xabar": f"Pixel shakli {pix.shape} noma’lum."}

    n = len(kadrlar)
    if n > 12:
        indekslar = np.linspace(0, n - 1, 12).astype(int)
        kadrlar = [kadrlar[i] for i in indekslar]
    meta["kadrlar_soni_xom"] = n
    return {
        "ok": True,
        "kadrlar": kadrlar,
        "meta": meta,
        "xabar": f"DICOM: {n} kadr, {meta.get('series_description') or meta.get('protocol_name') or 'meta yo‘q'}.",
    }


def video_dan_kadrlar(bayt: bytes) -> Dict[str, Any]:
    """mp4/avi baytidan bir nechta kadr oladi.

    Args:
        bayt: Video fayl.

    Returns:
        ok, kadrlar, meta, xabar.
    """
    if cv2 is None:
        return {"ok": False, "kadrlar": [], "meta": {}, "xabar": "opencv o‘rnatilmagan (video)."}
    tmp = Path("/tmp/cardio_echo_video.bin")
    try:
        tmp.write_bytes(bayt)
        cap = cv2.VideoCapture(str(tmp))
        if not cap.isOpened():
            return {"ok": False, "kadrlar": [], "meta": {}, "xabar": "Video ochilmadi."}
        jami = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        olish = min(8, max(1, jami if jami > 0 else 8))
        kadrlar: List[np.ndarray] = []
        if jami > 0:
            indekslar = np.linspace(0, max(jami - 1, 0), olish).astype(int)
            for idx in indekslar:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if ok and frame is not None:
                    kadrlar.append(_kulrang(frame))
        else:
            for _ in range(olish):
                ok, frame = cap.read()
                if not ok:
                    break
                kadrlar.append(_kulrang(frame))
        cap.release()
        if not kadrlar:
            return {"ok": False, "kadrlar": [], "meta": {}, "xabar": "Videodan kadr chiqmadi."}
        return {
            "ok": True,
            "kadrlar": kadrlar,
            "meta": {"fps": fps, "frames": jami, "series_description": ""},
            "xabar": f"Video: {len(kadrlar)} kadr (jami {jami}).",
        }
    except Exception as exc:
        return {"ok": False, "kadrlar": [], "meta": {}, "xabar": f"Video xato: {exc}"}
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def rasm_dan_kadr(bayt: bytes) -> Dict[str, Any]:
    """PNG/JPEG baytidan bitta kadr.

    Args:
        bayt: Rasm.

    Returns:
        ok, kadrlar, meta, xabar.
    """
    try:
        from matplotlib import image as mpimg

        img = mpimg.imread(io.BytesIO(bayt), format=None)
        kadr = _kulrang(np.asarray(img))
        return {
            "ok": True,
            "kadrlar": [kadr],
            "meta": {"series_description": ""},
            "xabar": f"Rasm kadr {kadr.shape}.",
        }
    except Exception:
        pass
    if cv2 is not None:
        buf = np.frombuffer(bayt, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return {
                "ok": True,
                "kadrlar": [_kulrang(img)],
                "meta": {},
                "xabar": f"Rasm kadr {img.shape}.",
            }
    return {"ok": False, "kadrlar": [], "meta": {}, "xabar": "Rasm o‘qilmadi."}


def echo_fayldan_kadrlar(nom: str, bayt: bytes) -> Dict[str, Any]:
    """Fayl turiga qarab DICOM/video/rasm o‘qiydi.

    Args:
        nom: Fayl nomi (kengaytma).
        bayt: Tarkib.

    Returns:
        dicom_dan_kadrlar bilan bir xil shakl.
    """
    past = (nom or "").lower()
    if past.endswith((".dcm", ".dicom")) or bayt[:4] == b"DICM" or b"DICM" in bayt[:132]:
        return dicom_dan_kadrlar(bayt)
    if past.endswith((".mp4", ".avi", ".mov", ".mpg", ".mpeg", ".mkv")):
        return video_dan_kadrlar(bayt)
    if past.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
        return rasm_dan_kadr(bayt)
    dicom_urinish = dicom_dan_kadrlar(bayt)
    if dicom_urinish.get("ok"):
        return dicom_urinish
    rasm = rasm_dan_kadr(bayt)
    if rasm.get("ok"):
        return rasm
    return {
        "ok": False,
        "kadrlar": [],
        "meta": {},
        "xabar": f"Echo formati tanilmadi ({nom}). .dcm, video yoki rasm yuklang.",
    }


def echo_yollardan_oqish(
    fayllar: Sequence[Tuple[str, bytes]],
) -> Dict[str, Any]:
    """Bir yoki bir nechta echo faylni kadr to‘plamiga yig‘adi.

    Args:
        fayllar: (nom, bayt) juftliklari.

    Returns:
        ok, yozuvlar [{nom, kadrlar, meta, xabar}], xabar.
    """
    if not fayllar:
        return {"ok": False, "yozuvlar": [], "xabar": "Echo fayl yuklanmadi."}
    yozuvlar: List[Dict[str, Any]] = []
    xatolar: List[str] = []
    for nom, bayt in fayllar:
        nat = echo_fayldan_kadrlar(nom, bayt)
        if nat.get("ok") and nat.get("kadrlar"):
            yozuvlar.append(
                {
                    "nom": nom,
                    "kadrlar": nat["kadrlar"],
                    "meta": nat.get("meta") or {},
                    "xabar": nat.get("xabar"),
                }
            )
        else:
            xatolar.append(f"{nom}: {nat.get('xabar')}")
    if not yozuvlar:
        return {"ok": False, "yozuvlar": [], "xabar": "Hech bir echo fayldan kadr chiqmadi. " + "; ".join(xatolar)}
    xabar = f"{len(yozuvlar)} echo yozuv o‘qildi."
    if xatolar:
        xabar += " Xato: " + "; ".join(xatolar)
    return {"ok": True, "yozuvlar": yozuvlar, "xabar": xabar}
