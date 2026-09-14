"""Rasmiy sahifalarni data/raw/{manba}/ ga HTML sifatida yuklab oladi.

Matn gitga kiritilmaydi. MedlinePlus odatda public domain; Mayo/NHS/ESC
mualliflik huquqli — faqat lokal RAG indeksi uchun.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

LOYIHA = Path(__file__).resolve().parents[2]
RAW_DIR = LOYIHA / "data" / "raw"
MANBALAR_JSON = Path(__file__).resolve().parent / "manbalar.json"
USER_AGENT = (
    "CardioAIResearchBot/0.1 (local RAG ingest; educational CDS; not for redistribution)"
)


def _fayl_nomi(url: str) -> str:
    """URL dan xavfsiz fayl nomi yasaydi.

    Args:
        url: Yuklanadigan sahifa.

    Returns:
        .html bilan tugaydigan nom.
    """
    parsed = urlparse(url)
    qism = (parsed.path.strip("/") or "index").replace("/", "_")
    qism = re.sub(r"[^A-Za-z0-9._-]", "_", qism)
    if not qism.lower().endswith(".html") and not qism.lower().endswith(".htm"):
        qism = qism + ".html"
    return qism[:180]


def manbalarni_oqish() -> List[Dict]:
    """manbalar.json dagi guruhlarni o‘qiydi.

    Returns:
        id, nom, urls ro‘yxati.
    """
    data = json.loads(MANBALAR_JSON.read_text(encoding="utf-8"))
    return list(data.get("manbalar") or [])


def sahifani_yukla(url: str, timeout: int = 45) -> Tuple[bool, str]:
    """Bitta URL ni HTML matn sifatida oladi.

    Args:
        url: HTTPS manzil.
        timeout: Sekund.

    Returns:
        (ok, html_yoki_xato).
    """
    talab = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.8",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(talab, timeout=timeout) as javob:
            xom = javob.read()
        return True, xom.decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return False, str(exc)


def barcha_sahifalarni_yukla(kutish: float = 0.8) -> Dict[str, int]:
    """manbalar.json dagi URL larni data/raw/{id}/ ga yozadi.

    Args:
        kutish: So‘rovlar orasidagi pauza (serverni bosmaslik).

    Returns:
        muvaffaqiyat, xato, o‘tkazib_yuborilgan sonlari.
    """
    hisob = {"ok": 0, "xato": 0, "bor": 0}
    xatolar: List[str] = []
    for guruh in manbalarni_oqish():
        jild = RAW_DIR / str(guruh.get("id") or "boshqa")
        jild.mkdir(parents=True, exist_ok=True)
        for url in guruh.get("urls") or []:
            yol = jild / _fayl_nomi(str(url))
            meta = yol.with_suffix(".url.txt")
            if yol.exists() and yol.stat().st_size > 500:
                hisob["bor"] += 1
                continue
            ok, matn = sahifani_yukla(str(url))
            time.sleep(kutish)
            if not ok or len(matn) < 400:
                hisob["xato"] += 1
                xatolar.append(f"{url}: {matn[:200]}")
                continue
            yol.write_text(matn, encoding="utf-8")
            meta.write_text(str(url), encoding="utf-8")
            hisob["ok"] += 1
    hisob["xabarlar"] = xatolar  # type: ignore[assignment]
    return hisob


if __name__ == "__main__":
    natija = barcha_sahifalarni_yukla()
    print(natija)
