"""Hujjatlarni toza matnga keltirish va ustma-ust chunklarga ajratish.

HTML — BeautifulSoup. PDF — pypdf, bo‘lsa pdfplumber, oxirida Docling.
Manbalar: StatPearls, ESC/AHA-ACC PDF, Mayo, NHS, MedlinePlus.
Bo‘laklar lokal CardiacRAG uchun. Tashxis o‘rnini bosmaydi.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

# Eski so‘z kesimi (test va qiyos). Asosiy indeks belgi kesimidan foydalanadi.
CHUNK_SOZ = 180
CHUNK_USTMA_UST = 40
# Semantik bo‘lak: belgi. BioClinicalBERT 256 token oynasiga sig‘adigan hajm.
CHUNK_BELGI = 500
CHUNK_USTMA = 50
# Juda qisqa qoldiqni indeksga yozmaslik
MIN_BELGI = 80


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


def _pypdf_matn(yol: Path) -> str:
    """PDF sahifalarini pypdf bilan matnga yig‘adi.

    Args:
        yol: Lokal PDF (yo‘riqnoma yoki slayd to‘plami).

    Returns:
        Sahifa matnlari. Kutubxona yoki o‘qish xatosida bo‘sh satr.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        oquvchi = PdfReader(str(yol))
        sahifalar = [(sahifa.extract_text() or "") for sahifa in oquvchi.pages]
    except Exception:
        return ""
    return matnni_tozala("\n".join(sahifalar))


def _pdfplumber_matn(yol: Path) -> str:
    """PDF ni pdfplumber bilan o‘qiydi (jadval va ustunlar uchun zaxira).

    Args:
        yol: Lokal PDF.

    Returns:
        Toza matn. pdfplumber yo‘q bo‘lsa bo‘sh satr.
    """
    try:
        import pdfplumber
    except ImportError:
        return ""
    try:
        qismlar: List[str] = []
        with pdfplumber.open(str(yol)) as pdf:
            for sahifa in pdf.pages:
                qismlar.append(sahifa.extract_text() or "")
    except Exception:
        return ""
    return matnni_tozala("\n".join(qismlar))


def _docling_matn(yol: Path) -> str:
    """PDF ni Docling orqali o‘qiydi, agar kutubxona o‘rnatilgan bo‘lsa.

    Args:
        yol: PDF fayl yo‘li.

    Returns:
        Toza matn yoki bo‘sh satr.
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        return ""
    try:
        natija = DocumentConverter().convert(str(yol))
    except Exception:
        return ""
    hujjat = getattr(natija, "document", None)
    if hujjat is None:
        return ""
    if hasattr(hujjat, "export_to_text"):
        return matnni_tozala(hujjat.export_to_text())
    return matnni_tozala(str(hujjat))


def pdf_dan_matn(yol: Path) -> str:
    """PDF ni pypdf, so‘ng pdfplumber, so‘ng Docling bilan matnga aylantiradi.

    Args:
        yol: data/raw/guidelines yoki boshqa jilddagi PDF.

    Returns:
        Eng uzun muvaffaqiyatli matn. Hech biri o‘qilmasa bo‘sh satr.
        Klinik tashxis emas.
    """
    nomzodlar = (_pypdf_matn(yol), _pdfplumber_matn(yol), _docling_matn(yol))
    return max(nomzodlar, key=len)


def belgilarga_ajrat(
    matn: str,
    chunk_size: int = CHUNK_BELGI,
    overlap: int = CHUNK_USTMA,
) -> List[str]:
    """Matnni paragraf chegarasida belgi bo‘yicha ustma-ust bo‘laklaydi.

    Args:
        matn: Toza tibbiy matn.
        chunk_size: Bo‘lakdagi eng ko‘p belgi (odatiy 500).
        overlap: Keyingi bo‘lak boshiga o‘tadigan belgilar (odatiy 50).

    Returns:
        Semantik bo‘laklar. 80 belgidan qisqa qoldiq tashlanadi.
        CardiacRAG embeddingi shu bo‘laklarni ko‘radi.
    """
    if chunk_size < MIN_BELGI:
        chunk_size = MIN_BELGI
    overlap = max(0, min(overlap, chunk_size // 3))
    qatorlar = [q.strip() for q in matn.splitlines() if q.strip()]
    if not qatorlar:
        return []
    qismlar: List[str] = []
    for qator in qatorlar:
        if len(qator) <= chunk_size:
            qismlar.append(qator)
            continue
        boshi = 0
        while boshi < len(qator):
            qismlar.append(qator[boshi : boshi + chunk_size])
            if boshi + chunk_size >= len(qator):
                break
            boshi += max(1, chunk_size - overlap)
    bolaklar: List[str] = []
    buf = ""
    for qism in qismlar:
        if not buf:
            buf = qism
            continue
        if len(buf) + 1 + len(qism) <= chunk_size:
            buf = buf + "\n" + qism
            continue
        bolaklar.append(buf.strip())
        dum = buf[-overlap:].strip() if overlap else ""
        if dum and len(dum) + 1 + len(qism) <= chunk_size + overlap:
            buf = dum + "\n" + qism
        else:
            buf = qism
    if buf.strip():
        bolaklar.append(buf.strip())
    return [b for b in bolaklar if len(b) >= MIN_BELGI]


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


def fayldan_bolaklar(
    yol: Path,
    chunk_size: int = CHUNK_BELGI,
    overlap: int = CHUNK_USTMA,
) -> List[str]:
    """Fayl turiga qarab matn olib, belgi bo‘laklariga ajratadi.

    Args:
        yol: .html, .htm, .pdf, .txt, .md.
        chunk_size: Bo‘lak belgisi (odatiy 500).
        overlap: Ustma-ust belgi (odatiy 50).

    Returns:
        Indekslanadigan bo‘laklar, [jild/fayl] prefiksi bilan.
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
    return [pref + b for b in belgilarga_ajrat(matn, chunk_size, overlap)]


def katalogdan_bolaklar(
    katalog: Path,
    glob_andoza: str = "*.*",
    chunk_size: int = CHUNK_BELGI,
    overlap: int = CHUNK_USTMA,
) -> List[str]:
    """Katalogdagi (ichki jildlar bilan) mos fayllardan bo‘laklar yig‘adi.

    Args:
        katalog: data/raw yoki knowledge papkasi.
        glob_andoza: Fayl filtri (rglob).
        chunk_size: Bo‘lak belgisi.
        overlap: Ustma-ust belgi.

    Returns:
        Barcha chunklar. CardiacRAG indeksi uchun.
    """
    katalog = Path(katalog)
    if not katalog.exists():
        return []
    yigindi: List[str] = []
    for fayl in sorted(katalog.rglob(glob_andoza)):
        if fayl.is_file():
            yigindi.extend(fayldan_bolaklar(fayl, chunk_size, overlap))
    return yigindi


def korpus_bolaklari(
    seed: Path,
    raw_katalog: Path,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> List[str]:
    """Seed matn va data/raw hujjatlaridan overlapping chunklarni yig‘adi.

    Args:
        seed: knowledge/cardiology_seed.txt.
        raw_katalog: Yuklab olingan HTML/PDF/StatPearls jildi.
        chunk_size: Belgi bo‘lak o‘lchami. None — 500.
        overlap: Ustma-ust. None — 50.

    Returns:
        Indekslanadigan bo‘laklar (manba prefiksi bilan).
    """
    hajm = CHUNK_BELGI if chunk_size is None else chunk_size
    ustma = CHUNK_USTMA if overlap is None else overlap
    yigindi: List[str] = []
    if seed.exists():
        seed_matn = matnni_tozala(seed.read_text(encoding="utf-8"))
        yigindi.extend("[seed/cardiology_seed] " + b for b in belgilarga_ajrat(seed_matn, hajm, ustma))
    yigindi.extend(katalogdan_bolaklar(raw_katalog, chunk_size=hajm, overlap=ustma))
    return yigindi
