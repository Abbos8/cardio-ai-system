"""EchoNet-Dynamic DeepLabv3-ResNet50 LV og‘irligini models/echonet ga yuklaydi.

Zhang 2024 o‘z ACNN og‘irligini nashr etmagan. Shu arxitekturaning ochiq
EchoNet-Dynamic checkpointi ishlatiladi. Gitga yozilmaydi. Tashxis modeli emas.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

URL = "https://github.com/echonet/dynamic/releases/download/v1.0.0/deeplabv3_resnet50_random.pt"


def yukla(manzil: Path | None = None) -> Path:
    """deeplabv3_resnet50.pt ni yuklab, berilgan joyga yozadi.

    Args:
        manzil: Chiqish fayli. Bo‘sh bo‘lsa models/echonet/deeplabv3_resnet50.pt.

    Returns:
        Yozilgan .pt yo‘li. Fayl bor bo‘lsa qayta yuklamaydi.
    """
    from tools.echo_segmenter import ogirlik_yoli

    yol = manzil or ogirlik_yoli()
    if yol.is_file() and yol.stat().st_size > 1_000_000:
        return yol
    yol.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=600) as javob, yol.open("wb") as chiqish:
        while True:
            bolak = javob.read(1 << 20)
            if not bolak:
                break
            chiqish.write(bolak)
    return yol


if __name__ == "__main__":
    print(yukla())
