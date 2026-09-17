"""Git va matnda sir/PII izlarini tekshirish (demo himoya).

Kalit, bemor identifikatori va xom yuklamalar gitga kirmasligi kerak.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List, Sequence

# Track qilinmasligi kerak bo‘lgan yo‘llar (nisbiy)
TAQIQLANGAN_GIT = (
    ".env",
    ".env.local",
    "credentials.json",
)

TAQIQLANGAN_PREFIKS = (
    "data/raw/",
    "data/processed/",
    "data/audit/",
    "data/uploads/",
    "venv/",
)

_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_TEL = re.compile(r"\+\d{10,15}\b")
_KALIT = re.compile(
    r"(sk-[A-Za-z0-9]{8,}|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY|api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,})",
    re.I,
)


def gitda_taqiqlanganlar(ildiz: Path) -> List[str]:
    """git ls-files ichida sir/xom ma’lumot yo‘llarini qaytaradi.

    Args:
        ildiz: Loyiha ildizi (.git bor).

    Returns:
        Taqiqlangan track qilingan yo‘llar. Bo‘sh — toza.
    """
    ildiz = Path(ildiz)
    try:
        chiq = subprocess.check_output(
            ["git", "ls-files"],
            cwd=str(ildiz),
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return []
    xato: List[str] = []
    for satr in chiq.splitlines():
        nom = satr.strip()
        if not nom:
            continue
        if nom in TAQIQLANGAN_GIT or nom.endswith("/.env"):
            xato.append(nom)
            continue
        if any(nom.startswith(p) for p in TAQIQLANGAN_PREFIKS):
            xato.append(nom)
    return xato


def pii_izlari(matn: str) -> List[str]:
    """Matnda email, telefon yoki kalitga o‘xshash qismlarni belgilaydi.

    Args:
        matn: Tekshiriladigan matn (audit yoki xulosa).

    Returns:
        Topilgan turlar ro‘yxati (email, telefon, kalit).
    """
    if not matn:
        return []
    turlar: List[str] = []
    if _EMAIL.search(matn):
        turlar.append("email")
    if _KALIT.search(matn):
        turlar.append("kalit")
    if _TEL.search(matn) and not _KALIT.search(matn):
        turlar.append("telefon")
    return turlar


def kalit_maydonlarini_ol(obyekt: object, chuqurlik: int = 4) -> List[str]:
    """Lug‘at/ro‘yxatda api_key yoki signal baytlarini qidiradi.

    Args:
        obyekt: Audit yozuvi.
        chuqurlik: Rekursiya chegarasi.

    Returns:
        Topilgan maydon nomlari.
    """
    topilgan: List[str] = []
    _yur(obyekt, chuqurlik, topilgan, "")
    return topilgan


def _yur(obyekt: object, qolgan: int, topilgan: List[str], yol: str) -> None:
    """Ichki daraxtni aylanadi.

    Args:
        obyekt: Tugun.
        qolgan: Qolgan chuqurlik.
        topilgan: Natija ro‘yxati.
        yol: Hozirgi yo‘l.
    """
    if qolgan < 0 or obyekt is None:
        return
    if isinstance(obyekt, dict):
        for k, v in obyekt.items():
            past = str(k).lower()
            yangi = f"{yol}.{k}" if yol else str(k)
            if past in {"api_key", "api_kalit", "password", "ecg_signal", "bayt", "overlay_png"}:
                if v not in (None, "", [], {}):
                    topilgan.append(yangi)
            _yur(v, qolgan - 1, topilgan, yangi)
    elif isinstance(obyekt, (list, tuple)):
        for i, v in enumerate(obyekt[:20]):
            _yur(v, qolgan - 1, topilgan, f"{yol}[{i}]")


def matndan_sirni_kes(matn: str) -> str:
    """Auditga yozishdan oldin email/kalitni yulduzcha bilan almashtiradi.

    Args:
        matn: Xom satr.

    Returns:
        Qisqartirilgan satr. Klinik tashxis emas.
    """
    if not matn:
        return ""
    t = _EMAIL.sub("[email]", matn)
    t = _KALIT.sub("[kalit]", t)
    return t[:800]


def git_fayllarini_sirga_tekshir(ildiz: Path, fayllar: Sequence[str] | None = None) -> List[str]:
    """Track qilingan matnli fayllarda kalit naqshini qidiradi.

    Args:
        ildiz: Loyiha ildizi.
        fayllar: None — git ls-files.

    Returns:
        Kalit izi bor yo‘llar.
    """
    ildiz = Path(ildiz)
    if fayllar is None:
        try:
            chiq = subprocess.check_output(
                ["git", "ls-files"],
                cwd=str(ildiz),
                text=True,
                stderr=subprocess.DEVNULL,
            )
            fayllar = [s.strip() for s in chiq.splitlines() if s.strip()]
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return []
    ruxsat = {".py", ".md", ".txt", ".json", ".example", ".sh", ".yml", ".yaml", ".toml", ".csv"}
    xato: List[str] = []
    for nom in fayllar:
        if nom.endswith(".example"):
            continue
        if Path(nom).suffix.lower() not in ruxsat:
            continue
        yol = ildiz / nom
        if not yol.is_file() or yol.stat().st_size > 400_000:
            continue
        try:
            matn = yol.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if _KALIT.search(matn):
            xato.append(nom)
    return xato
