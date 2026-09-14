"""Ko‘p tarmoqli munozara (MDT): MedGemma va Qwen2.5-VL rollari.

VLM og‘irliklari bo‘lmasa, LLM yoki shablon bilan I (xom) va Z (oraliq) qayta kiritiladi.
Konsensus yoki max_raund da to‘xtaydi. Tashxis o‘rnini bosmaydi.
"""

from __future__ import annotations

from typing import Any, Dict, List

from llm.client import llm_chat

MAX_RAUND = 3


def _rol_javobi(rol: str, xom_i: str, oraliq_z: str, oldingi: str) -> str:
    """Bitta ekspert rolida munozara matnini oladi.

    Args:
        rol: MedGemma yoki Qwen2.5-VL tavsifi.
        xom_i: Dastlabki xom ma’lumotlar.
        oraliq_z: Joriy vosita natijalari.
        oldingi: Boshqa modelning oxirgi fikri.

    Returns:
        Qisqa ekspert fikri. Tashxis emas.
    """
    javob = llm_chat(
        [
            {
                "role": "system",
                "content": (
                    f"Siz {rol}. Gallyutsinatsiyadan saqlaning: faqat berilgan I va Z ga tayaning. "
                    "Tashxis qo‘ymang. O‘zbek yoki ingliz tilida 5–8 jumla. "
                    "Yetarsiz dalil bo‘lsa shunday yozing."
                ),
            },
            {
                "role": "user",
                "content": f"I (xom):\n{xom_i}\n\nZ (oraliq):\n{oraliq_z}\n\nHamkasb:\n{oldingi}",
            },
        ]
    )
    if javob:
        return javob.strip()
    return (
        f"{rol}: VLM/LLM ulanmagan. I va Z qayta ko‘rildi. "
        "Tasvir/video modeli yo‘qligi sababli konsensus ehtiyotkor: yetarli vizual dalil yo‘q. "
        "Shifokor ko‘rigi shart."
    )


def mdt_munozara(
    xom_i: str,
    oraliq_z: str,
    max_raund: int = MAX_RAUND,
) -> Dict[str, Any]:
    """Ikkita rol mustaqil yozadi, keyin o‘zaro almashadi; bosh umumlashtiradi.

    Args:
        xom_i: Bemor va xom signallar qisqasi.
        oraliq_z: Lab, EKG, echo, fellow.
        max_raund: Maksimal munozara raundlari.

    Returns:
        raundlar, umumlashtirish, konsensus. Tashxis emas.
    """
    raundlar: List[Dict[str, str]] = []
    med = ""
    qwen = ""
    for raund in range(1, max(1, max_raund) + 1):
        med = _rol_javobi(
            "MedGemma — tibbiy tasvirlar bo‘yicha ekspert",
            xom_i,
            oraliq_z,
            qwen,
        )
        qwen = _rol_javobi(
            "Qwen2.5-VL — video/echo bo‘yicha ekspert",
            xom_i,
            oraliq_z,
            med,
        )
        raundlar.append({"raund": str(raund), "medgemma": med, "qwen": qwen})
        birlash = (med + " " + qwen).lower()
        if "yetarli" in birlash and "yetarsiz" not in birlash and "insufficient" not in birlash:
            break
        if "konsensus" in birlash or "agree" in birlash:
            break

    umumiy = llm_chat(
        [
            {
                "role": "system",
                "content": (
                    "Bosh kardiolog MDT xulosasini qisqa yozadi. Tashxis qo‘ymang. "
                    "I va Z ni inobatga oling."
                ),
            },
            {
                "role": "user",
                "content": f"I:\n{xom_i}\nZ:\n{oraliq_z}\nRaundlar:\n{raundlar}",
            },
        ]
    )
    if not umumiy:
        umumiy = (
            "MDT: tasvir va video modellari yoki API yo‘q. "
            "Xom ma’lumot (I) va oraliq natijalar (Z) qayta kiritildi. "
            "Konsensus: qo‘shimcha klinik tasdiq kerak. Tashxis emas."
        )
    return {
        "ok": True,
        "raundlar": raundlar,
        "umumlashtirish": umumiy.strip(),
        "raund_soni": len(raundlar),
        "xabar": "MDT yakunlandi. Shifokor o‘rnini bosmaydi.",
    }
