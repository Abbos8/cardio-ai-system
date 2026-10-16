"""EchoPrime 11 ta yirik ko‘rinish: A5C va Doppler bilan.

Asosiy tasnif — EchoPrime view_classifier (ConvNeXt, klinik og‘irlik).
Og‘irlik yo‘q bo‘lsa sintetik geometrik regressiya zaxira. Tashxis emas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# EchoPrime COARSE_VIEWS tartibi — view_classifier.pt indeksi shu ro‘yxat
ECHO_KORINISHLAR: List[str] = [
    "A2C",
    "A3C",
    "A4C",
    "A5C",
    "Apical_Doppler",
    "Doppler_Parasternal_Long",
    "Doppler_Parasternal_Short",
    "Parasternal_Long",
    "Parasternal_Short",
    "SSN",
    "Subcostal",
]

# Eski mayda yorliqlar EchoPrime yirik sinfiga tushadi
_ESKI_NOMLAR = {
    "PLAX": "Parasternal_Long",
    "PSAX-AV": "Parasternal_Short",
    "PSAX-MV": "Parasternal_Short",
    "PSAX-PM": "Parasternal_Short",
    "PSAX-APEX": "Parasternal_Short",
    "SUBCOSTAL-4C": "Subcostal",
    "SUBCOSTAL-IVC": "Subcostal",
}

_OGIRLIK = Path(__file__).resolve().parents[2] / "models" / "echoprime" / "view_classifier.pt"
_EP_MEAN = (29.110628, 28.076836, 29.096405)
_EP_STD = (47.989223, 46.456997, 47.20083)

# Aniqroq iboralar oldin: «apical doppler» «apical 4» dan oldin tursin
_TEG_KALIT: List[Tuple[str, str]] = [
    ("doppler parasternal long", "Doppler_Parasternal_Long"),
    ("parasternal long doppler", "Doppler_Parasternal_Long"),
    ("plax doppler", "Doppler_Parasternal_Long"),
    ("doppler parasternal short", "Doppler_Parasternal_Short"),
    ("parasternal short doppler", "Doppler_Parasternal_Short"),
    ("psax doppler", "Doppler_Parasternal_Short"),
    ("apical doppler", "Apical_Doppler"),
    ("a4c doppler", "Apical_Doppler"),
    ("mv inflow", "Apical_Doppler"),
    ("apical 5", "A5C"),
    ("apical five", "A5C"),
    ("five chamber", "A5C"),
    ("5 chamber", "A5C"),
    ("a5c", "A5C"),
    ("5ch", "A5C"),
    ("subcostal", "Subcostal"),
    ("sc4c", "Subcostal"),
    ("ivc", "Subcostal"),
    ("parasternal long", "Parasternal_Long"),
    ("plax", "Parasternal_Long"),
    ("parasternal short", "Parasternal_Short"),
    ("psax", "Parasternal_Short"),
    ("short axis", "Parasternal_Short"),
    ("suprasternal", "SSN"),
    ("ssn", "SSN"),
    ("apical 4", "A4C"),
    ("apical four", "A4C"),
    ("four chamber", "A4C"),
    ("4 chamber", "A4C"),
    ("4ch", "A4C"),
    ("a4c", "A4C"),
    ("apical 3", "A3C"),
    ("apical three", "A3C"),
    ("three chamber", "A3C"),
    ("apical long", "A3C"),
    ("a3c", "A3C"),
    ("3ch", "A3C"),
    ("apical 2", "A2C"),
    ("apical two", "A2C"),
    ("two chamber", "A2C"),
    ("a2c", "A2C"),
    ("2ch", "A2C"),
    ("long axis", "Parasternal_Long"),
]


def korinish_nomi(nom: str) -> str:
    """Eski mayda yorliqni EchoPrime 11 tasidan biriga keltiradi.

    Args:
        nom: PLAX, A5C yoki allaqachon yirik nom.

    Returns:
        ECHO_KORINISHLAR dagi nom. Noma’lum satr o‘zgarmaydi.
    """
    k = str(nom or "").strip()
    return _ESKI_NOMLAR.get(k, k)


def tegdan_korinish(meta: Optional[Dict[str, str]]) -> Optional[str]:
    """DICOM/protokol matnidan EchoPrime 11 ko‘rinishidan birini aniqlaydi.

    Args:
        meta: series_description, protocol_name, view_name, image_comments.

    Returns:
        A5C va Doppler bilan yirik yorliq, yoki None. Fayl nomiga qaramaydi.
    """
    if not meta:
        return None
    matn = " ".join(
        str(meta.get(k) or "")
        for k in ("series_description", "protocol_name", "view_name", "image_comments")
    )
    s = matn.lower().replace("_", " ").replace("-", " ")
    if not s.strip():
        return None
    for kalit, yorliq in _TEG_KALIT:
        if kalit in s:
            return yorliq
    return None


def _ellips(img: np.ndarray, cy: float, cx: float, ry: float, rx: float, qiymat: int = 20) -> None:
    """Kulrang kadrga to‘ldirilgan ellips chizadi (sintetik kamera).

    Args:
        img: O‘zgaruvchan (H, W) uint8.
        cy, cx: Markaz.
        ry, rx: Yarim o‘qlar.
        qiymat: Intensivlik (bo‘shliqlar qorong‘i).
    """
    h, w = img.shape
    yy, xx = np.ogrid[:h, :w]
    mask = ((yy - cy) / max(ry, 1e-3)) ** 2 + ((xx - cx) / max(rx, 1e-3)) ** 2 <= 1.0
    img[mask] = qiymat


def yasama_echo_kadr(korinish: str, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Berilgan ko‘rinishga mos ultratovushga o‘xshash kadr yasaydi.

    Args:
        korinish: ECHO_KORINISHLAR dan.
        rng: Shovqin uchun.

    Returns:
        (128, 128) uint8. Haqiqiy bemor echo si emas.
    """
    rng = rng or np.random.default_rng(0)
    img = np.full((128, 128), 18, dtype=np.uint8)
    yy, xx = np.ogrid[:128, :128]
    fan = (np.abs(xx - 64) < (8 + yy * 0.55)) & (yy > 8)
    img[fan] = 90
    k = korinish_nomi(korinish)
    if k == "A4C":
        _ellips(img, 48, 44, 16, 14)
        _ellips(img, 48, 84, 16, 14)
        _ellips(img, 88, 46, 18, 16)
        _ellips(img, 88, 82, 18, 16)
    elif k == "A2C":
        _ellips(img, 42, 64, 18, 16)
        _ellips(img, 90, 64, 22, 18)
    elif k == "A3C":
        _ellips(img, 40, 52, 14, 12)
        _ellips(img, 42, 80, 12, 10)
        _ellips(img, 88, 64, 20, 18)
    elif k == "A5C":
        _ellips(img, 46, 40, 14, 12)
        _ellips(img, 46, 88, 14, 12)
        _ellips(img, 86, 48, 16, 14)
        _ellips(img, 86, 80, 16, 14)
        _ellips(img, 28, 64, 8, 10)
    elif k == "Parasternal_Long":
        _ellips(img, 50, 40, 12, 22)
        _ellips(img, 58, 78, 16, 26)
        _ellips(img, 88, 70, 10, 18)
    elif k == "Parasternal_Short":
        _ellips(img, 64, 64, 28, 30)
        _ellips(img, 64, 52, 8, 6, qiymat=70)
        _ellips(img, 64, 76, 8, 6, qiymat=70)
    elif k == "Subcostal":
        _ellips(img, 52, 48, 12, 11)
        _ellips(img, 52, 76, 12, 11)
        _ellips(img, 82, 50, 14, 12)
        _ellips(img, 82, 74, 14, 12)
    elif k in ("Apical_Doppler", "Doppler_Parasternal_Long", "Doppler_Parasternal_Short"):
        img[fan] = 40
        img[70:78, 20:110] = 200
        img[40:100, 30] = 210
        if k == "Doppler_Parasternal_Short":
            img[50:90, 90:100] = 180
    elif k == "SSN":
        _ellips(img, 40, 64, 10, 28)
        _ellips(img, 58, 48, 16, 8)
        _ellips(img, 58, 80, 16, 8)
    else:
        _ellips(img, 64, 64, 20, 20)
    shovqin = rng.integers(0, 12, size=img.shape, dtype=np.uint8)
    return np.clip(img.astype(np.int16) + shovqin - 4, 0, 255).astype(np.uint8)


def kadr_belgilari(kadr: np.ndarray) -> np.ndarray:
    """Kulrang kadrning geometrik belgilarini hisoblaydi.

    Args:
        kadr: 2D echo kadr.

    Returns:
        1D float vektor (model kirishi). Kavaklar, dumaloqlik, joylashuv.
    """
    from tools.echo_yuklash import _kulrang

    g = _kulrang(kadr)
    h, w = 128, 128
    ys = np.linspace(0, g.shape[0] - 1, h).astype(int)
    xs = np.linspace(0, g.shape[1] - 1, w).astype(int)
    g = g[ys][:, xs]
    # Bo‘shliqlar qorong‘i
    chegara = max(int(np.percentile(g, 35)), 1)
    mask = g < chegara
    yy, xx = np.indices((h, w))
    fan = np.abs(xx - 64) < (10 + yy * 0.6)
    mask &= fan
    n_pix = int(mask.sum())
    y_mean = float(yy[mask].mean()) / h if n_pix else 0.5
    x_mean = float(xx[mask].mean()) / w if n_pix else 0.5
    y_std = float(yy[mask].std()) / h if n_pix else 0.0
    x_std = float(xx[mask].std()) / w if n_pix else 0.0

    # Bog‘langan komponentlar
    lab = np.zeros_like(mask, dtype=np.int32)
    yorliq = 0
    for i in range(h):
        for j in range(w):
            if not mask[i, j] or lab[i, j]:
                continue
            yorliq += 1
            stek = [(i, j)]
            lab[i, j] = yorliq
            while stek:
                y, x = stek.pop()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and lab[ny, nx] == 0:
                        lab[ny, nx] = yorliq
                        stek.append((ny, nx))
    maydonlar: List[int] = []
    dumaloq: List[float] = []
    for k in range(1, yorliq + 1):
        kom = lab == k
        m = int(kom.sum())
        if m < 40:
            continue
        maydonlar.append(m)
        ys, xs = np.where(kom)
        ry = max((ys.max() - ys.min()) / 2.0, 1.0)
        rx = max((xs.max() - xs.min()) / 2.0, 1.0)
        ell = np.pi * ry * rx
        dumaloq.append(float(m) / ell if ell else 0.0)
    n_k = len(maydonlar)
    eng = max(maydonlar) / float(h * w) if maydonlar else 0.0
    circ = float(np.mean(dumaloq)) if dumaloq else 0.0
    nisbat = (x_std + 1e-6) / (y_std + 1e-6)
    return np.array(
        [n_k, circ, y_mean, x_mean, y_std, x_std, nisbat, eng, n_pix / float(h * w)],
        dtype=float,
    )


_MODEL_W: Optional[np.ndarray] = None
_MODEL_B: Optional[np.ndarray] = None
_VC: Optional[object] = None
_VC_XATO: Optional[str] = None


def ogirlik_yoli() -> Path:
    """EchoPrime view_classifier.pt joyini qaytaradi.

    Returns:
        models/echoprime/view_classifier.pt. Gitga kirmaydi.
    """
    return _OGIRLIK


def _vc_ol() -> Optional[object]:
    """ConvNeXt view classifier ni bir marta yuklaydi.

    Returns:
        eval rejimidagi model yoki None. Og‘irlik yoki torchvision yo‘q bo‘lsa zaxira ishlaydi.
    """
    global _VC, _VC_XATO
    if _VC is not None:
        return _VC
    if _VC_XATO:
        return None
    yol = ogirlik_yoli()
    if not yol.is_file():
        return None
    try:
        import torch
        import torchvision

        model = torchvision.models.convnext_base(weights=None)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(ECHO_KORINISHLAR))
        try:
            holat = torch.load(yol, map_location="cpu", weights_only=True)
        except TypeError:
            holat = torch.load(yol, map_location="cpu")
        model.load_state_dict(holat)
        model.eval()
        _VC = model
        return model
    except Exception as exc:
        _VC_XATO = str(exc)
        return None


def tasnif_model_nomi() -> str:
    """Hozir qaysi tasnif og‘irligi ishlatilishini aytadi.

    Returns:
        echoprime yoki geometrik_model. Tashxis nomi emas.
    """
    return "echoprime" if _vc_ol() is not None else "geometrik_model"


def _echoprime_tasnif(kadr: np.ndarray, model: object) -> Dict[str, float]:
    """Bitta kadrni EchoPrime ConvNeXt bilan 11 sinfga ajratadi.

    Args:
        kadr: Kulrang yoki rangli echo kadr.
        model: view_classifier.

    Returns:
        yorliq → ehtimol. Birinchi kadr, EchoPrime get_views kabi. Tashxis emas.
    """
    import torch

    from tools.echo_yuklash import _kulrang

    g = _kulrang(kadr)
    try:
        import cv2

        g = cv2.resize(g, (224, 224), interpolation=cv2.INTER_CUBIC)
    except Exception:
        ys = np.linspace(0, g.shape[0] - 1, 224).astype(int)
        xs = np.linspace(0, g.shape[1] - 1, 224).astype(int)
        g = g[ys][:, xs]
    rgb = np.repeat(g[:, :, None], 3, axis=2).astype(np.float32)
    t = torch.from_numpy(rgb).permute(2, 0, 1)
    mean = torch.tensor(_EP_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(_EP_STD, dtype=torch.float32).view(3, 1, 1)
    t = (t - mean) / std
    with torch.no_grad():
        logit = model(t.unsqueeze(0))
        p = torch.softmax(logit, dim=1).squeeze(0).cpu().numpy()
    return {nom: float(p[i]) for i, nom in enumerate(ECHO_KORINISHLAR)}


def _modelni_oqit() -> Tuple[np.ndarray, np.ndarray]:
    """Sintetik kadrlar bo‘yicha logistik regressiya og‘irliklarini hisoblaydi.

    Returns:
        W (11, F), b (11,). Demo model; klinik validatsiya yo‘q.
    """
    global _MODEL_W, _MODEL_B
    if _MODEL_W is not None and _MODEL_B is not None:
        return _MODEL_W, _MODEL_B
    rng = np.random.default_rng(21)
    X: List[np.ndarray] = []
    y: List[int] = []
    for idx, nom in enumerate(ECHO_KORINISHLAR):
        for _ in range(6):
            kadr = yasama_echo_kadr(nom, rng)
            X.append(kadr_belgilari(kadr))
            y.append(idx)
    Xa = np.vstack(X)
    ya = np.array(y)
    try:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=400)
        clf.fit(Xa, ya)
        _MODEL_W = np.asarray(clf.coef_, dtype=float)
        _MODEL_B = np.asarray(clf.intercept_, dtype=float)
    except Exception:
        # Zaxira: sinf markaziga masofa
        markaz = np.stack([Xa[ya == i].mean(axis=0) for i in range(len(ECHO_KORINISHLAR))])
        _MODEL_W = -2.0 * markaz
        _MODEL_B = np.sum(markaz**2, axis=1)
    return _MODEL_W, _MODEL_B


def geometrik_tasnif(kadr: np.ndarray) -> Dict[str, float]:
    """Bitta kadr uchun EchoPrime 11 ko‘rinish ehtimolini geometriyadan beradi.

    Args:
        kadr: Kulrang echo.

    Returns:
        yorliq → ehtimol. Tashxis emas.
    """
    W, b = _modelni_oqit()
    x = kadr_belgilari(kadr)
    logit = W @ x + b
    logit = logit - logit.max()
    p = np.exp(logit)
    p = p / p.sum()
    return {nom: float(p[i]) for i, nom in enumerate(ECHO_KORINISHLAR)}


def kadrlar_tasnif(kadrlar: Sequence[np.ndarray]) -> Dict[str, float]:
    """Bir nechta kadr ehtimolini o‘rtacha qiladi.

    Args:
        kadrlar: Video/DICOM kadrlari.

    Returns:
        yorliq → o‘rtacha ehtimol.
    """
    if not kadrlar:
        return {n: 0.0 for n in ECHO_KORINISHLAR}
    model = _vc_ol()
    if model is not None:
        return _echoprime_tasnif(kadrlar[0], model)
    jami = np.zeros(len(ECHO_KORINISHLAR), dtype=float)
    for kadr in kadrlar[:6]:
        eht = geometrik_tasnif(kadr)
        jami += np.array([eht[n] for n in ECHO_KORINISHLAR])
    jami /= max(len(kadrlar[:6]), 1)
    return {n: float(jami[i]) for i, n in enumerate(ECHO_KORINISHLAR)}
