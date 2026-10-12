"""Ochiq ESC / AHA-ACC PDF yo‘riqnomalarini data/raw/guidelines/ ga yuklaydi.

Faqat haqiqiy PDF (%PDF) saqlanadi. Nashriyot 403 qaytarsa xato yoziladi,
indeks to‘xtamaydi. Papkada turgan qo‘lda qo‘yilgan PDF lar ham o‘qiladi
(`ingest.pdf_dan_matn`: pypdf, pdfplumber, Docling).
Matn gitga kiritilmaydi. Tashxis o‘rnini bosmaydi.
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

LOYIHA = Path(__file__).resolve().parents[2]
GUIDELINES_DIR = LOYIHA / "data" / "raw" / "guidelines"
USER_AGENT = (
    "CardioAIResearchBot/0.1 (local RAG ingest; educational CDS; not for redistribution)"
)

# Ochiq URL lar. ESC slayd to‘plamlari shu tarmoqda PDF qaytaradi.
# AHA/ACC to‘liq PDF ko‘pincha nashriyot tomonidan 403 bilan yopiladi — urinish qoladi.
OCHIQ_PDF = [
    {
        "id": "esc_hf_2026_slayd",
        "nom": "2026 ESC heart failure — rasmiy slayd to‘plami",
        "url": "https://dam-assets.escardio.org/download/b2e587389baa11f185de06bdfb3e4be9",
    },
    {
        "id": "esc_valvular_slayd",
        "nom": "ESC valvular heart disease — rasmiy slayd to‘plami",
        "url": "https://yjxzhi.files.cmp.optimizely.com/download/f718bde6bfbc11f08f5ddabefdf356e7",
    },
    {
        "id": "aha_acc_chest_pain_2021",
        "nom": "2021 AHA/ACC/ASE chest pain guideline",
        "url": "https://www.ahajournals.org/doi/pdf/10.1161/CIR.0000000000001029",
    },
    {
        "id": "aha_acc_hfsa_hf_2022",
        "nom": "2022 AHA/ACC/HFSA heart failure guideline",
        "url": "https://www.ahajournals.org/doi/pdf/10.1161/CIR.0000000000001063",
    },
]


def pdf_yaroqli(yol: Path) -> bool:
    """Fayl haqiqiy PDF ekanini bosh baytidan tekshiradi.

    Args:
        yol: Nomzod fayl.

    Returns:
        True — %PDF va hajmi yetarli.
    """
    yol = Path(yol)
    if not yol.is_file() or yol.stat().st_size < 1000:
        return False
    with yol.open("rb") as fayl:
        return fayl.read(5) == b"%PDF-"


def pdf_ni_yukla(url: str, timeout: int = 120) -> tuple:
    """Bitta ochiq URL dan PDF baytlarini oladi.

    Args:
        url: HTTPS manzil.
        timeout: Sekund.

    Returns:
        (ok, bayt_yoki_xato_matn).
    """
    talab = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/pdf,*/*;q=0.8",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(talab, timeout=timeout) as javob:
            turi = (javob.headers.get("Content-Type") or "").lower()
            xom = javob.read()
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return False, str(exc).encode()
    if not xom.startswith(b"%PDF-"):
        return False, f"PDF emas ({turi or 'noma’lum'})".encode()
    if len(xom) > 40_000_000:
        return False, b"PDF 40 MB dan katta"
    return True, xom


def ochiq_pdflarni_yukla(kutish: float = 1.0) -> Dict[str, object]:
    """OCHIQ_PDF ro‘yxatini data/raw/guidelines/ ga yozadi.

    Args:
        kutish: So‘rovlar orasidagi pauza. Serverni bosmaslik uchun.

    Returns:
        ok, bor, xato, xabarlar. Papkadagi boshqa PDF larga tegilmaydi.
    """
    pauza = max(1.0, float(kutish))
    GUIDELINES_DIR.mkdir(parents=True, exist_ok=True)
    hisob = {"ok": 0, "bor": 0, "xato": 0}
    xabarlar: List[str] = []
    for qator in OCHIQ_PDF:
        yol = GUIDELINES_DIR / f"{qator['id']}.pdf"
        meta = yol.with_suffix(".url.txt")
        if pdf_yaroqli(yol):
            hisob["bor"] += 1
            continue
        ok, xom = pdf_ni_yukla(str(qator["url"]))
        time.sleep(pauza)
        if not ok or not xom.startswith(b"%PDF-"):
            hisob["xato"] += 1
            xabar = xom.decode("utf-8", errors="replace")[:180]
            xabarlar.append(f"{qator['id']}: {xabar}")
            continue
        yol.write_bytes(xom)
        meta.write_text(f"{qator['url']}\n{qator['nom']}\n", encoding="utf-8")
        hisob["ok"] += 1
    natija: Dict[str, object] = dict(hisob)
    natija["xabarlar"] = xabarlar
    return natija


if __name__ == "__main__":
    natija = ochiq_pdflarni_yukla()
    print(
        f"Yo‘riqnoma PDF: yangi={natija.get('ok')} bor={natija.get('bor')} xato={natija.get('xato')}"
    )
    for xabar in natija.get("xabarlar") or []:
        print("  xato:", xabar)
