"""Hujjatlarni toza matnga keltirish va ustma-ust chunklarga ajratish.

BeautifulSoup (HTML) va Docling (PDF) ixtiyoriy; yo‘q bo‘lsa oddiy matn kesish.
Manbalar: Mayo Clinic, NHS, MedlinePlus, ESC — faqat foydalanuvchi bergan fayllar.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

# Chunk o‘lchami (so‘z) va ustma-ust qismi — retrieval sifatini saqlash uchun
CHUNK_SOZ = 180
CHUNK_USTMA_UST = 40


def matnni_tozala(matn: str) -> str:
    """Bo‘sh qator va ortiqcha bo‘shliqlarni siqadi.

    Args:
        matn: HTML/PDF dan olingan xom matn.

    Returns:
        Bir qatorli-ishlatishga qulay toza matn.
    """
    qatorlar = [" ".join(q.split()) for q in matn.splitlines()]
    return "\n".join(q for q in qatorlar if q).strip()


def html_dan_matn(html: str) -> str:
    """HTML ni BeautifulSoup bilan toza matnga aylantiradi.

    Args:
        html: Sahifa manbasi.

    Returns:
        Toza matn; bs4 yo‘q bo‘lsa teglarsiz taxminiy matn.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        import re

        return matnni_tozala(re.sub(r"<[^>]+>", " ", html))
    shorba = BeautifulSoup(html, "html.parser")
    for teg in shorba(
        ["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "iframe", "svg"]
    ):
        teg.decompose()
    asos = shorba.find("main") or shorba.find("article") or shorba.body or shorba
    return matnni_tozala(asos.get_text(" "))


def pdf_dan_matn(yol: Path) -> str:
    """PDF ni Docling orqali o‘qiydi; kutubxona yo‘q bo‘lsa bo‘sh qator.

    Args:
        yol: PDF fayl yo‘li.

    Returns:
        Toza matn yoki bo‘sh satr.
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        return ""
    natija = DocumentConverter().convert(str(yol))
    hujjat = getattr(natija, "document", None)
    if hujjat is None:
        return ""
    if hasattr(hujjat, "export_to_text"):
        return matnni_tozala(hujjat.export_to_text())
    return matnni_tozala(str(hujjat))


def chunklarga_ajrat(
    matn: str,
    soz_soni: int = CHUNK_SOZ,
    ustma_ust: int = CHUNK_USTMA_UST,
) -> List[str]:
    """Matnni ustma-ust tushuvchi kichik bo‘laklarga ajratadi.

    Args:
        matn: Toza tibbiy matn.
        soz_soni: Bo‘lakdagi so‘zlar.
        ustma_ust: Keyingi bo‘lakka o‘tadigan so‘zlar.

    Returns:
        Bo‘laklar ro‘yxati. CardiacRAG indeksi uchun.
    """
    sozlar = matn.split()
    if not sozlar:
        return []
    qadam = max(1, soz_soni - ustma_ust)
    bolaklar: List[str] = []
    i = 0
    while i < len(sozlar):
        bolaklar.append(" ".join(sozlar[i : i + soz_soni]))
        i += qadam
        if i >= len(sozlar):
            break
    return [b for b in bolaklar if len(b.split()) >= 40]


def fayldan_bolaklar(yol: Path) -> List[str]:
    """Fayl turiga qarab matn olib, chunklarga ajratadi.

    Args:
        yol: .html, .htm, .pdf, .txt, .md.

    Returns:
        Indekslanadigan bo‘laklar.
    """
    yol = Path(yol)
    if not yol.exists() or yol.name.endswith(".url.txt"):
        return []
    suffix = yol.suffix.lower()
    if suffix in {".html", ".htm"}:
        matn = html_dan_matn(yol.read_text(encoding="utf-8", errors="ignore"))
    elif suffix == ".pdf":
        matn = pdf_dan_matn(yol)
    elif suffix in {".txt", ".md"}:
        matn = matnni_tozala(yol.read_text(encoding="utf-8", errors="ignore"))
    else:
        return []
    pref = f"[{yol.parent.name}/{yol.stem}] "
    return [pref + b for b in chunklarga_ajrat(matn)]


def katalogdan_bolaklar(katalog: Path, glob_andoza: str = "*.*") -> List[str]:
    """Katalogdagi (ichki jildlar bilan) mos fayllardan bo‘laklar yig‘adi.

    Args:
        katalog: data/raw yoki knowledge papkasi.
        glob_andoza: Fayl filtri (rglob).

    Returns:
        Barcha chunklar. CardiacRAG indeksi uchun.
    """
    katalog = Path(katalog)
    if not katalog.exists():
        return []
    yigindi: List[str] = []
    for fayl in sorted(katalog.rglob(glob_andoza)):
        if fayl.is_file():
            yigindi.extend(fayldan_bolaklar(fayl))
    return yigindi


def korpus_bolaklari(seed: Path, raw_katalog: Path) -> List[str]:
    """Seed matn va data/raw hujjatlaridan overlapping chunklarni yig‘adi.

    Args:
        seed: knowledge/cardiology_seed.txt.
        raw_katalog: Yuklab olingan HTML/PDF jildi.

    Returns:
        Indekslanadigan bo‘laklar (manba prefiksi bilan).
    """
    yigindi: List[str] = []
    if seed.exists():
        yigindi.extend(chunklarga_ajrat(matnni_tozala(seed.read_text(encoding="utf-8"))))
    yigindi.extend(katalogdan_bolaklar(raw_katalog))
    return yigindi
