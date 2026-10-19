"""Echocardiography segmenter: A4C chap qorincha maskasi va taxminiy EF.

Asosiy yo‘l — DeepLabv3-ResNet50 (EchoNet-Dynamic og‘irligi, Zhang 2024
area-length EF). Og‘irlik yo‘q bo‘lsa OpenCV kavak, EF bermaydi.
Natija tashxis emas. Kadr yo‘q bo‘lsa maska yaratilmaydi.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from tools.echo_tool import _yuklamalarni_yig
from tools.echo_yuklash import _kulrang, echo_yollardan_oqish

APICAL = {"A4C", "A2C", "A3C", "A5C"}
SAX = {"Parasternal_Short", "PSAX-AV", "PSAX-MV", "PSAX-PM", "PSAX-APEX"}
_OGIRLIK = Path(__file__).resolve().parents[2] / "models" / "echonet" / "deeplabv3_resnet50.pt"
_KIRISH = 112
_KADR_CHEGARA = 128
_EF_MIN_KADR = 21
_CHOQ_QADAM = 20
MODEL_DEEPLAB = "zhang2024-deeplab"
MODEL_OPENCV = "opencv-kavak"
_LV: Optional[object] = None
_LV_XATO: Optional[str] = None


def overlay_qur(kadr: np.ndarray, maska: np.ndarray) -> np.ndarray:
    """Kulrang kadr ustiga cyan maska va sariq kontur chizadi.

    Args:
        kadr: uint8 (H, W).
        maska: 0/1 yoki 0/255.

    Returns:
        RGB uint8 overlay. Ko‘rish uchun, o‘lchov emas.
    """
    g = _kulrang(kadr)
    m = (np.asarray(maska) > 0).astype(np.uint8)
    if m.shape != g.shape:
        if cv2 is not None:
            m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            m = np.zeros_like(g, dtype=np.uint8)
    rgb = np.stack([g, g, g], axis=-1).astype(np.float32)
    cyan = np.array([40.0, 200.0, 220.0])
    rgb[m > 0] = rgb[m > 0] * 0.45 + cyan * 0.55
    if cv2 is not None:
        kontur, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rgb_u = np.clip(rgb, 0, 255).astype(np.uint8)
        cv2.drawContours(rgb_u, kontur, -1, (255, 220, 40), 1)
        return rgb_u
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _png_bayt(rgb: np.ndarray) -> Optional[bytes]:
    """RGB kadrni PNG baytiga siqadi (UI/graf uchun).

    Args:
        rgb: (H, W, 3) uint8.

    Returns:
        PNG yoki None.
    """
    if cv2 is None:
        return None
    bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return None
    return bytes(buf)


def _fan_maska(g: np.ndarray) -> np.ndarray:
    """Ultratovush sektori taxminiy maskasi.

    Args:
        g: Kulrang kadr.

    Returns:
        bool (H, W). Sektor tashqarisi olinmaydi.
    """
    h, w = g.shape
    yy, xx = np.indices((h, w))
    fan = np.abs(xx - w / 2.0) < (0.10 * w + yy * 0.62)
    if cv2 is not None:
        yorug = (g > np.percentile(g, 18)).astype(np.uint8)
        yorug = cv2.dilate(yorug, np.ones((7, 7), np.uint8), iterations=1)
        return fan & (yorug > 0)
    return fan & (g > 12)


def _kavak_maska(g: np.ndarray, korinish: Optional[str]) -> Tuple[Optional[np.ndarray], str]:
    """Qorong‘i kamerani (taxminiy LV) ajratadi.

    Args:
        g: Kulrang echo kadr.
        korinish: A4C/A2C/... yoki None.

    Returns:
        (maska uint8 0/1, izoh). Topilmasa (None, sabab).
    """
    if cv2 is None:
        return None, "opencv yo‘q"
    h, w = g.shape
    blur = cv2.GaussianBlur(g, (5, 5), 0)
    fan = _fan_maska(blur).astype(np.uint8)
    if int(fan.sum()) < 80:
        return None, "ultratovush sektori topilmadi"

    # Bo‘shliqlar qorong‘i
    fan_pix = blur[fan > 0]
    chegara = int(np.percentile(fan_pix, 32))
    qorongi = ((blur < chegara) & (fan > 0)).astype(np.uint8)
    yadro = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    qorongi = cv2.morphologyEx(qorongi, cv2.MORPH_OPEN, yadro, iterations=1)
    qorongi = cv2.morphologyEx(qorongi, cv2.MORPH_CLOSE, yadro, iterations=2)

    n, yorliq, stats, centroids = cv2.connectedComponentsWithStats(qorongi, connectivity=8)
    fan_may = float(fan.sum())
    nomzod: List[Tuple[float, int]] = []
    for i in range(1, n):
        may = float(stats[i, cv2.CC_STAT_AREA])
        ulush = may / fan_may
        if ulush < 0.012 or ulush > 0.42:
            continue
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        ww, hh = float(stats[i, cv2.CC_STAT_WIDTH]), float(stats[i, cv2.CC_STAT_HEIGHT])
        dumaloq = may / (max(np.pi * (ww / 2) * (hh / 2), 1.0))
        k = str(korinish or "").upper()
        ball = ulush
        if k in {a.upper() for a in APICAL}:
            ball += 0.35 * (cy / h) + 0.25 * (cx / w)
        elif k in {a.upper() for a in SAX}:
            ball += 0.45 * min(dumaloq, 1.2)
        else:
            ball += 0.2 * (cy / h)
        nomzod.append((ball, i))

    if not nomzod:
        return None, "yetarli kattalikdagi qorong‘i kavak yo‘q"

    _, eng = max(nomzod, key=lambda t: t[0])
    maska = (yorliq == eng).astype(np.uint8)
    # Teshiklarni to‘ldirish
    kontur, _ = cv2.findContours(maska, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    toldir = np.zeros_like(maska)
    cv2.drawContours(toldir, kontur, -1, 1, thickness=cv2.FILLED)
    return toldir, "kavak ajratildi"


def ogirlik_yoli() -> Path:
    """EchoNet DeepLabv3 LV og‘irligi joyini qaytaradi.

    Returns:
        models/echonet/deeplabv3_resnet50.pt. Gitga kirmaydi.
    """
    return _OGIRLIK


def area_length_hajm(maydon: float, uzunlik: float) -> Optional[float]:
    """Ellipsoid single-plane hajm: V = 8·A² / (3·π·L).

    Args:
        maydon: LV maydoni, piksel².
        uzunlik: Halqa o‘rtasidan apexgacha, piksel.

    Returns:
        Hajm (piksel³) yoki None. EF nisbatida birlik qisqaradi. Tashxis emas.
    """
    if maydon <= 0 or uzunlik <= 0:
        return None
    return 8.0 * float(maydon) ** 2 / (3.0 * math.pi * float(uzunlik))


def ef_foiz(edv: float, esv: float) -> Optional[float]:
    """Bitta sikl EF foizi: 100·(EDV−ESV)/EDV.

    Args:
        edv: Diastola oxiri hajmi.
        esv: Sistola oxiri hajmi.

    Returns:
        0–100 oralig‘idagi foiz yoki None. Klinik o‘lchov emas.
    """
    if edv <= 0 or esv < 0 or esv >= edv:
        return None
    return 100.0 * (float(edv) - float(esv)) / float(edv)


def _choqqilar(qiymat: np.ndarray, past: bool = False) -> List[int]:
    """Maydon egri chizig‘idagi ED (tepalar) yoki ES (pastliklar).

    Args:
        qiymat: Kadr bo‘yicha LV maydoni.
        past: True bo‘lsa minimumlar (ES).

    Returns:
        Indekslar. Qadam 20, prominence diapazonning yarmi. Zhang 2024.
    """
    s = np.asarray(qiymat, dtype=float)
    if past:
        s = -s
    if s.size < 3:
        return []
    diap = float(s.max() - s.min())
    if diap <= 0:
        return []
    pol = float(s.min()) + 0.5 * diap
    nomzod = [
        i
        for i in range(1, int(s.size) - 1)
        if s[i] >= s[i - 1] and s[i] > s[i + 1] and s[i] >= pol
    ]
    tanlov: List[int] = []
    for i in sorted(nomzod, key=lambda j: float(s[j]), reverse=True):
        if all(abs(i - j) >= _CHOQ_QADAM for j in tanlov):
            tanlov.append(i)
    return sorted(tanlov)


def sikl_juftliklari(maydonlar: Sequence[float]) -> List[Tuple[int, int]]:
    """ED tepasidan keyingi ES pastligini sikl jufti qiladi.

    Args:
        maydonlar: Kadr maydonlari.

    Returns:
        (ed_indeks, es_indeks) ro‘yxati. Yetarli cho‘qqi bo‘lmasa bo‘sh.
    """
    y = np.asarray(list(maydonlar), dtype=float)
    eds = _choqqilar(y, past=False)
    ess = _choqqilar(y, past=True)
    juft: List[Tuple[int, int]] = []
    for ed in eds:
        keyin = [es for es in ess if es > ed]
        if not keyin:
            continue
        es = keyin[0]
        keyingi_ed = [e for e in eds if e > ed]
        if keyingi_ed and es >= keyingi_ed[0]:
            continue
        juft.append((ed, es))
    return juft


def jeffrey_maydon(maydonlar: Sequence[float], ed: int, es: int, diastola: bool) -> float:
    """ED/ES maydonini sikl va global 10% o‘rtacha bilan tekislaydi.

    Args:
        maydonlar: Barcha kadr maydonlari.
        ed: Diastola oxiri indeksi.
        es: Sistola oxiri indeksi.
        diastola: True — katta maydonlar; False — kichik maydonlar.

    Returns:
        Tekislangan maydon (piksel²). Zhang 2024 improved Jeffrey.
    """
    m = np.asarray(list(maydonlar), dtype=float)
    n = max(int(m.size), 1)
    k = max(1, int(round(0.1 * n)))
    oraliq = m[ed : es + 1]
    if oraliq.size == 0:
        oraliq = m
    kk = max(1, int(round(0.1 * int(oraliq.size))))
    if diastola:
        nuqta = float(m[ed])
        global10 = np.sort(m)[::-1][:k]
        sikl10 = np.sort(oraliq)[::-1][:kk]
    else:
        nuqta = float(m[es])
        global10 = np.sort(m)[:k]
        sikl10 = np.sort(oraliq)[:kk]
    return float(np.mean(np.concatenate([np.array([nuqta]), global10, sikl10])))


def lv_uzunlik_px(maska: np.ndarray) -> Optional[float]:
    """Maskadan halqa o‘rtasi–apex masofasini pikselda oladi.

    Args:
        maska: 0/1 LV maskasi.

    Returns:
        Uzunlik yoki None. Nashr etilmagan ansambl o‘rniga geometriya.
    """
    m = np.asarray(maska) > 0
    if int(m.sum()) < 10:
        return None
    ys, xs = np.nonzero(m)
    coords = np.stack([ys.astype(float), xs.astype(float)], axis=1)
    qoldiq = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(qoldiq, full_matrices=False)
    oq = vh[0]
    yon = vh[1] if vh.shape[0] > 1 else np.zeros(2)
    proy = qoldiq @ oq
    past = np.percentile(proy, 12)
    yuqori = np.percentile(proy, 88)
    uchi_a = coords[proy <= past]
    uchi_b = coords[proy >= yuqori]
    if uchi_a.size == 0 or uchi_b.size == 0:
        return None

    def kenglik(nuqta: np.ndarray) -> float:
        q = (nuqta - coords.mean(axis=0)) @ yon
        return float(np.std(q)) if q.size else 0.0

    if kenglik(uchi_a) <= kenglik(uchi_b):
        apex, halqa = uchi_a.mean(axis=0), uchi_b.mean(axis=0)
    else:
        apex, halqa = uchi_b.mean(axis=0), uchi_a.mean(axis=0)
    uzun = float(np.linalg.norm(apex - halqa))
    if uzun < 2:
        return None
    return uzun


def ef_sikl_ortacha(
    maydonlar: Sequence[float],
    uzunliklar: Sequence[Optional[float]],
) -> Tuple[Optional[float], List[int], List[int]]:
    """Barcha sikllar area-length EF o‘rtachasini qaytaradi.

    Args:
        maydonlar: Kadr maydonlari.
        uzunliklar: Har kadr uzunligi yoki None.

    Returns:
        (ef foiz yoki None, ED indekslar, ES indekslar). 21 kadrdan kam bo‘lsa EF yo‘q.
    """
    if len(maydonlar) < _EF_MIN_KADR:
        return None, [], []
    juft = sikl_juftliklari(maydonlar)
    if not juft:
        return None, [], []
    eflar: List[float] = []
    eds: List[int] = []
    ess: List[int] = []
    for ed, es in juft:
        if ed >= len(uzunliklar) or es >= len(uzunliklar):
            continue
        lu_ed = uzunliklar[ed]
        lu_es = uzunliklar[es]
        if lu_ed is None or lu_es is None:
            continue
        a_ed = jeffrey_maydon(maydonlar, ed, es, True)
        a_es = jeffrey_maydon(maydonlar, ed, es, False)
        edv = area_length_hajm(a_ed, float(lu_ed))
        esv = area_length_hajm(a_es, float(lu_es))
        if edv is None or esv is None:
            continue
        foiz = ef_foiz(edv, esv)
        if foiz is None:
            continue
        eflar.append(foiz)
        eds.append(ed)
        ess.append(es)
    if not eflar:
        return None, eds, ess
    return float(np.mean(eflar)), eds, ess


def _lv_ol() -> Optional[object]:
    """DeepLabv3-ResNet50 ni bir marta yuklaydi.

    Returns:
        eval modeli yoki None. Kalitlar mos kelmasa zaxira ishlaydi.
    """
    global _LV, _LV_XATO
    if _LV is not None:
        return _LV
    if _LV_XATO:
        return None
    yol = ogirlik_yoli()
    if not yol.is_file():
        return None
    try:
        import torch
        import torchvision

        model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights=None,
            weights_backbone=None,
            num_classes=1,
            aux_loss=False,
        )
        try:
            holat = torch.load(yol, map_location="cpu", weights_only=False)
        except TypeError:
            holat = torch.load(yol, map_location="cpu")
        if isinstance(holat, dict) and "state_dict" in holat:
            holat = holat["state_dict"]
        if not isinstance(holat, dict):
            raise TypeError("checkpoint lug‘at emas")
        toza = {}
        for kalit, qiymat in holat.items():
            nom = kalit[7:] if kalit.startswith("module.") else kalit
            if nom.startswith("aux_classifier"):
                continue
            toza[nom] = qiymat
        natija = model.load_state_dict(toza, strict=False)
        if natija.missing_keys:
            raise RuntimeError("yetishmaydigan kalitlar: " + ", ".join(natija.missing_keys[:8]))
        model.eval()
        _LV = model
        return model
    except Exception as exc:
        _LV_XATO = str(exc)
        return None


def tasnif_model_nomi() -> str:
    """Hozir qaysi LV modeli ishlatilishini aytadi.

    Returns:
        zhang2024-deeplab yoki opencv-kavak. Tashxis nomi emas.
    """
    return MODEL_DEEPLAB if _lv_ol() is not None else MODEL_OPENCV


def _kadr_112(kadr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Kulrang kadrni 112×112 RGB ga keltiradi.

    Args:
        kadr: Echo kadr.

    Returns:
        (float32 HWC, asl (h, w)).
    """
    g = _kulrang(kadr)
    asl = (int(g.shape[0]), int(g.shape[1]))
    if cv2 is not None:
        kichik = cv2.resize(g, (_KIRISH, _KIRISH), interpolation=cv2.INTER_AREA)
    else:
        ys = np.linspace(0, g.shape[0] - 1, _KIRISH).astype(int)
        xs = np.linspace(0, g.shape[1] - 1, _KIRISH).astype(int)
        kichik = g[ys][:, xs]
    rgb = np.repeat(kichik[:, :, None], 3, axis=2).astype(np.float32)
    return rgb, asl


def _deeplab_maskalar(kadrlar: Sequence[np.ndarray], model: object) -> List[np.ndarray]:
    """Kadrlar ketma-ketligini LV maskasiga ajratadi.

    Args:
        kadrlar: Kulrang kadrlar, ko‘pi bilan 128.
        model: deeplabv3_resnet50.

    Returns:
        Har kadr uchun 0/1 maska, asl o‘lchamda. Logit > 0. Tashxis emas.
    """
    import torch

    maskalar: List[np.ndarray] = []
    paket = 4
    for bosh in range(0, len(kadrlar), paket):
        bolak = list(kadrlar[bosh : bosh + paket])
        teng, shakllar = [], []
        for kadr in bolak:
            rgb, asl = _kadr_112(kadr)
            t = torch.from_numpy(rgb).permute(2, 0, 1)
            mean = t.mean(dim=(1, 2), keepdim=True)
            std = t.std(dim=(1, 2), keepdim=True).clamp_min(1e-3)
            teng.append((t - mean) / std)
            shakllar.append(asl)
        x = torch.stack(teng, dim=0)
        with torch.no_grad():
            chiq = model(x)["out"]
        for i, (h, w) in enumerate(shakllar):
            logit = chiq[i, 0].cpu().numpy()
            kichik = (logit > 0).astype(np.uint8)
            if cv2 is not None and kichik.shape != (h, w):
                kichik = cv2.resize(kichik, (w, h), interpolation=cv2.INTER_NEAREST)
            maskalar.append(kichik)
    return maskalar


def _massiv_kadrlar(kadrlar: Any) -> List[np.ndarray]:
    """(H,W) yoki (T,H,W) massivni kulrang kadrlar ro‘yxatiga aylantiradi.

    Args:
        kadrlar: Numpy massiv.

    Returns:
        Ko‘pi bilan 128 ta kadr.
    """
    massiv = np.asarray(kadrlar)
    if massiv.ndim == 2:
        return [_kulrang(massiv)]
    if massiv.ndim >= 3:
        return [_kulrang(massiv[i]) for i in range(min(int(massiv.shape[0]), _KADR_CHEGARA))]
    return []


def _kliplar_yig(
    bemor: Dict[str, Any],
    echo_natija: Optional[Dict[str, Any]],
    kadrlar: Optional[Any],
) -> List[Dict[str, Any]]:
    """A4C afzal klipni yig‘adi. Doppler keyinroq tashlanadi.

    Args:
        bemor: echo fayllari.
        echo_natija: Ko‘rinish yorlig‘i.
        kadrlar: Berilsa shu massiv alohida klip.

    Returns:
        [{nom, korinish, kadrlar}]. Har klipda ko‘pi bilan 128 kadr.
    """
    chiq: List[Dict[str, Any]] = []
    if kadrlar is not None:
        try:
            kadr_roy = _massiv_kadrlar(kadrlar)
        except Exception:
            kadr_roy = []
        if kadr_roy:
            chiq.append({"nom": "kadr", "korinish": None, "kadrlar": kadr_roy})
    yozuvlar = {z.get("fayl"): z for z in (echo_natija or {}).get("yozuvlar") or []}
    yuk = _yuklamalarni_yig(bemor)
    if yuk and kadrlar is None:
        oqish = echo_yollardan_oqish(yuk, kadr_limit=_KADR_CHEGARA)
        for yoz in oqish.get("yozuvlar") or []:
            nom = yoz.get("nom") or ""
            kor = (yozuvlar.get(nom) or {}).get("asosiy")
            kadr_roy = [_kulrang(k) for k in (yoz.get("kadrlar") or [])[:_KADR_CHEGARA]]
            if kadr_roy:
                chiq.append({"nom": nom, "korinish": kor, "kadrlar": kadr_roy})
    if not chiq and bemor.get("echo_kadrlar") is not None:
        try:
            kadr_roy = _massiv_kadrlar(bemor.get("echo_kadrlar"))
        except Exception:
            kadr_roy = []
        if kadr_roy:
            chiq.append({"nom": "echo_kadrlar", "korinish": None, "kadrlar": kadr_roy})
    return chiq


def _klip_tanla(kliplar: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Dopplerni tashlab, A4C va eng uzun klipni tanlaydi.

    Args:
        kliplar: Fayl kliplari.

    Returns:
        Bitta klip yoki None.
    """

    def tartib(klip: Dict[str, Any]) -> int:
        k = str(klip.get("korinish") or "")
        if "Doppler" in k:
            return 4
        if k == "A4C":
            return 0
        if k in APICAL:
            return 1
        if k in SAX:
            return 2
        return 3

    mos = [k for k in kliplar if tartib(k) < 4]
    if not mos:
        return None
    eng = min(tartib(k) for k in mos)
    bir = [k for k in mos if tartib(k) == eng]
    bir.sort(key=lambda k: len(k.get("kadrlar") or []), reverse=True)
    return bir[0]


def _bosh_javob(xabar: str, model: Optional[str] = None, kadrlar_soni: Optional[int] = None) -> Dict[str, Any]:
    """Maskasiz javob. EF o‘ylab topilmaydi.

    Args:
        xabar: Nima uchun maska yo‘q.
        model: Model nomi yoki None.
        kadrlar_soni: Ko‘rilgan kadrlar.

    Returns:
        ok=False lug‘at.
    """
    javob: Dict[str, Any] = {
        "ok": False,
        "xabar": xabar,
        "maska": None,
        "model": model,
        "ef": None,
        "ed": None,
        "es": None,
        "hfref": None,
    }
    if kadrlar_soni is not None:
        javob["kadrlar_soni"] = kadrlar_soni
    return javob


def _opencv_javob(klip: Dict[str, Any]) -> Dict[str, Any]:
    """OpenCV kavak zaxirasi. EF hisoblanmaydi.

    Args:
        klip: Tanlangan kadrlar.

    Returns:
        ok va overlay, yoki ok=False. ef doim None.
    """
    if cv2 is None:
        return _bosh_javob("opencv o‘rnatilmagan — LV maska hisoblanmadi.", None)
    kor = klip.get("korinish")
    oxirgi = "kavak topilmadi"
    for kadr in (klip.get("kadrlar") or [])[:6]:
        maska, sabab = _kavak_maska(kadr, kor)
        if maska is None:
            oxirgi = sabab
            continue
        return _muvaffaq(
            kadr,
            maska,
            kor,
            klip.get("nom"),
            MODEL_OPENCV,
            None,
            None,
            None,
            len(klip.get("kadrlar") or []),
            "algoritmik kavak",
        )
    return _bosh_javob(
        f"Kadr bor, lekin LV kavagi ajratilmadi ({oxirgi}). Yolg‘on maska berilmaydi.",
        MODEL_OPENCV,
        len(klip.get("kadrlar") or []),
    )


def _muvaffaq(
    kadr: np.ndarray,
    maska: np.ndarray,
    kor: Optional[str],
    nom: Optional[str],
    model: str,
    ef: Optional[float],
    ed: Optional[List[int]],
    es: Optional[List[int]],
    kadrlar_soni: int,
    usul: str,
) -> Dict[str, Any]:
    """Overlay va EF maydonlari bilan muvaffaqiyat javobi.

    Args:
        kadr: Overlay uchun kadr.
        maska: 0/1 LV.
        kor: Ko‘rinish.
        nom: Fayl nomi.
        model: zhang2024-deeplab yoki opencv-kavak.
        ef: Taxminiy foiz yoki None.
        ed: ED kadr indekslari.
        es: ES kadr indekslari.
        kadrlar_soni: Klip uzunligi.
        usul: Xabar ichidagi qisqa usul nomi.

    Returns:
        segment_lv muvaffaqiyat lug‘ati. Tashxis emas.
    """
    maydon = int(np.asarray(maska).sum())
    ulush = round(maydon / float(maska.size), 4) if maska.size else 0.0
    hfref = None if ef is None else bool(ef < 40.0)
    if ef is None:
        ef_matn = "EF hisoblanmadi (sikl yetarli emas)."
    elif hfref:
        ef_matn = f"Taxminiy EF {ef:.0f}% — HFrEF chegarasidan past (<40). Tashxis emas."
    else:
        ef_matn = f"Taxminiy EF {ef:.0f}%. Tashxis emas."
    xabar = (
        f"LV kontur ({usul}): ko‘rinish={kor or 'noma’lum'}, fayl={nom}, "
        f"{maydon} px ({ulush:.1%} kadr). {ef_matn}"
    )
    return {
        "ok": True,
        "xabar": xabar,
        "maska": maska,
        "overlay_png": _png_bayt(overlay_qur(kadr, maska)),
        "maydon_px": maydon,
        "ulush": ulush,
        "shakl": list(maska.shape),
        "korinish": kor,
        "fayl": nom,
        "model": model,
        "ef": None if ef is None else round(float(ef), 1),
        "ed": ed,
        "es": es,
        "hfref": hfref,
        "kadrlar_soni": kadrlar_soni,
    }


def segment_lv(
    kadrlar: Optional[Any] = None,
    bemor: Optional[Dict[str, Any]] = None,
    echo_natija: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Chap qorincha maskasi va, sikl yetarli bo‘lsa, taxminiy EF.

    Args:
        kadrlar: (T,H,W) yoki (H,W). Berilsa fayldan oldin shu klip.
        bemor: echo_fayllar, echo_bayt, echo_kadrlar.
        echo_natija: 11 ko‘rinish yorlig‘i (A4C afzal, Doppler emas).

    Returns:
        ok, overlay, ef, ed, es, model. Og‘irlik yo‘q bo‘lsa OpenCV va ef=None.
        Kadr yo‘q yoki kavak yo‘q — ok=False, yolg‘on maska yo‘q. Tashxis emas.
    """
    bemor = bemor or {}
    if echo_natija is None:
        echo_natija = bemor.get("echo_natija")
    try:
        kliplar = _kliplar_yig(bemor, echo_natija, kadrlar)
    except Exception as exc:
        return _bosh_javob(f"Kadrlar o‘qilmadi: {exc}", None)
    if not kliplar:
        return _bosh_javob("LV segmentatsiya uchun kadrlar yo‘q.", None)
    klip = _klip_tanla(kliplar)
    if klip is None:
        return _bosh_javob("LV uchun Doppler bo‘lmagan kadr yo‘q.", None, 0)

    model = _lv_ol()
    if model is None:
        return _opencv_javob(klip)

    kadr_roy = list(klip.get("kadrlar") or [])[:_KADR_CHEGARA]
    try:
        maskalar = _deeplab_maskalar(kadr_roy, model)
    except Exception:
        return _opencv_javob(klip)
    if not maskalar or not any(int(np.asarray(m).sum()) > 0 for m in maskalar):
        return _opencv_javob(klip)

    maydonlar = [float(np.asarray(m).sum()) for m in maskalar]
    uzunliklar = [lv_uzunlik_px(m) for m in maskalar]
    ef, eds, ess = ef_sikl_ortacha(maydonlar, uzunliklar)
    indeks = eds[0] if eds else int(np.argmax(maydonlar))
    return _muvaffaq(
        kadr_roy[indeks],
        maskalar[indeks],
        klip.get("korinish"),
        klip.get("nom"),
        MODEL_DEEPLAB,
        ef,
        eds or None,
        ess or None,
        len(kadr_roy),
        "DeepLabv3-ResNet50, area-length",
    )
