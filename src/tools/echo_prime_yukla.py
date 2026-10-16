"""EchoPrime view_classifier.pt ni models/echoprime ga yuklaydi.

To‘liq model_data.zip dan faqat ko‘rinish og‘irligi olinadi.
Gitga yozilmaydi. Tashxis modeli emas — faqat 11 ko‘rinish tasnifi.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/echonet/EchoPrime/releases/download/v1.0.0/model_data.zip"
ICHDAGI = "model_data/weights/view_classifier.pt"


def yukla(manzil: Path | None = None) -> Path:
    """view_classifier.pt ni yuklab, berilgan joyga yozadi.

    Args:
        manzil: Chiqish fayli. Bo‘sh bo‘lsa models/echoprime/view_classifier.pt.

    Returns:
        Yozilgan .pt yo‘li. Fayl bor bo‘lsa qayta yuklamaydi.
    """
    from tools.echo_view_model import ogirlik_yoli

    yol = manzil or ogirlik_yoli()
    if yol.is_file() and yol.stat().st_size > 1_000_000:
        return yol
    yol.parent.mkdir(parents=True, exist_ok=True)
    arxiv_yol = yol.with_suffix(".zip")
    with urllib.request.urlopen(URL, timeout=600) as javob, arxiv_yol.open("wb") as chiqish:
        while True:
            bolak = javob.read(1 << 20)
            if not bolak:
                break
            chiqish.write(bolak)
    with zipfile.ZipFile(arxiv_yol) as arxiv:
        nomlar = [n for n in arxiv.namelist() if n.endswith("view_classifier.pt")]
        if not nomlar:
            raise FileNotFoundError("model_data.zip ichida view_classifier.pt yo‘q")
        yol.write_bytes(arxiv.read(nomlar[0]))
    arxiv_yol.unlink(missing_ok=True)
    return yol


if __name__ == "__main__":
    print(yukla())
