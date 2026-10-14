"""Jonli tahlil: bosqich aniqlash, xabarlar va fon oqimi (API kalitsiz).

Natija tashxis emas — interfeys ko‘rinishi mantig‘ini tekshiradi.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ui.jonli_tahlil import (
    TahlilOmbori,
    bosqich_holati,
    ish_holati,
    jonli_xabarlar,
    md_toza,
    oqim_bosqichlari,
    paket_xulosasi,
)


def _reja() -> list:
    """Ikki qadamli sinov rejasi (lab, EKG)."""
    return [
        {"id": "p1", "vosita": "lab_technician", "tavsif": "lab", "holat": "pending"},
        {"id": "p2", "vosita": "ecg_technician", "tavsif": "ekg", "holat": "pending"},
    ]


class _SoxtaAgent:
    """Graf holatlarini ketma-ket beruvchi sinov agenti."""

    def __init__(self, holatlar, xato: bool = False, kutish: float = 0.0) -> None:
        self.holatlar = holatlar
        self.xato = xato
        self.kutish = kutish

    def run_oqim(self, paket):
        """Holatlarni birma-bir yield qiladi."""
        for h in self.holatlar:
            time.sleep(self.kutish)
            yield h
        if self.xato:
            raise RuntimeError("LLM yiqildi")

    def run(self, paket):
        """Oqim bo‘sh bo‘lsa zaxira."""
        return {"xulosa": "zaxira", "qadam_tarixi": ["x"]}


class TestBosqichHolati(unittest.TestCase):
    """Tugunlar ketma-ketligi bo‘yicha joriy bosqich."""

    def test_ketma_ketlik(self) -> None:
        """qabul → murakkablik → reja → vosita → stepwise → xulosa."""
        self.assertEqual(bosqich_holati(None)[0], 0)
        h = {"qadam_tarixi": ["qabul_qilish: ok"], "reja": [], "xulosa": ""}
        self.assertEqual(bosqich_holati(h)[0], 1)
        h["qadam_tarixi"].append("murakkablik_baholash: murakkab — sabab")
        self.assertEqual(bosqich_holati(h)[0], 2)
        h["qadam_tarixi"].append("cardiac_rag_reja: 2 qadam")
        h["reja"] = _reja()
        idx, matn = bosqich_holati(h)
        self.assertEqual(idx, 3)
        self.assertIn("lab_technician", matn)
        h["qadam_tarixi"].append("qadam_bajarish: lab_technician — ok")
        h["reja"][0]["holat"] = "done"
        h["reja_indeks"] = 1
        self.assertEqual(bosqich_holati(h)[0], 3)
        h["qadam_tarixi"].append("stepwise: CONTINUE — davom mdt=False")
        h["amal"] = "CONTINUE"
        _idx, matn = bosqich_holati(h)
        self.assertIn("ecg_technician", matn)
        h["qadam_tarixi"].append("stepwise: STOP — yetarli mdt=True")
        h["amal"], h["mdt_kerak"] = "STOP", True
        self.assertEqual(bosqich_holati(h)[0], 4)
        h["qadam_tarixi"].append("mdt: 1 raund konsensus=True")
        self.assertEqual(bosqich_holati(h)[0], 5)
        h["xulosa"] = "tayyor"
        self.assertEqual(bosqich_holati(h)[0], 6)

    def test_chiplar_mdt_o_tkazildi(self) -> None:
        """MDT bo‘lmagan oddiy holatda 5-chip skip, xulosadan keyin hammasi done."""
        h = {"qadam_tarixi": ["stepwise: STOP — ok mdt=False"], "amal": "STOP", "mdt_kerak": False}
        nomlar = oqim_bosqichlari(h)
        self.assertEqual(nomlar[:4], ["done"] * 4)
        self.assertEqual(nomlar[4], "skip")
        self.assertEqual(nomlar[5], "current")
        h["xulosa"] = "tayyor"
        nomlar = oqim_bosqichlari(h)
        self.assertEqual(nomlar[4], "skip")
        self.assertEqual(nomlar[5], "done")
        self.assertEqual(oqim_bosqichlari(None)[0], "current")


class TestXabarlar(unittest.TestCase):
    """Jonli xabarlar ro‘yxati."""

    def test_birinchi_xabar_darhol(self) -> None:
        """Hali holat yo‘qligida ham «yuklandi» qatori bor."""
        paket = {
            "yosh": 52,
            "jins": "erkak",
            "laboratoriya": {"a": 1, "b": 2},
            "ecg_signal": np.zeros((4000, 12)),
            "sampling_rate": 500.0,
            "echo_fayllar": [1, 2],
        }
        self.assertIn("12 tasma", paket_xulosasi(paket))
        q = jonli_xabarlar(paket, None, 0)
        self.assertIn("yuklandi", q[0]["matn"])
        self.assertEqual(q[1]["belgi"], "▶")

    def test_reja_va_natijalar(self) -> None:
        """Reja tuzilgach qadamlar belgi bilan chiqadi."""
        h = {
            "murakkablik": "murakkab",
            "murakkablik_sababi": "sabab",
            "reja": _reja(),
            "reja_indeks": 1,
            "lab_natija": {"qiymatlar": {"x": 1}, "tokenlar": ["t"]},
            "rag_dalillar": [1, 2, 3],
            "xulosa": "",
            "qadam_tarixi": ["stepwise: CONTINUE — davom"],
            "amal": "CONTINUE",
        }
        h["reja"][0]["holat"] = "done"
        q = jonli_xabarlar({"yosh": 1}, h, 3)
        matnlar = " | ".join(x["matn"] for x in q)
        self.assertIn("Klinik reja tuzildi", matnlar)
        ichki = [x for x in q if x["ichki"]]
        self.assertEqual([x["belgi"] for x in ichki], ["✅", "▶"])
        # Vosita tugab stepwise baholanayotganda keyingi qadam hali ▶ emas
        h["qadam_tarixi"].append("qadam_bajarish: ecg_technician — ok")
        q = jonli_xabarlar({"yosh": 1}, h, 3)
        self.assertEqual([x["belgi"] for x in q if x["ichki"]], ["✅", "○"])

    def test_md_toza(self) -> None:
        """Markdown belgilari ekranlanadi va matn qisqaradi."""
        self.assertNotIn("*", md_toza("a*b").replace("\\*", ""))
        self.assertLessEqual(len(md_toza("x" * 500, 50)), 50)


class TestFonOqimi(unittest.TestCase):
    """TahlilOmbori fon oqimi."""

    def _kut(self, ombor: TahlilOmbori, soniya: float = 5.0) -> None:
        """Ish tugashini kutadi."""
        t = time.time()
        while ombor.ishlayapti() and time.time() - t < soniya:
            time.sleep(0.02)

    def test_oraliq_holat_va_natija(self) -> None:
        """Skript bloklanmaydi: boshlash darhol qaytadi, holat oraliqda yangilanadi."""
        h1 = {"qadam_tarixi": ["qabul_qilish: ok"], "xulosa": ""}
        h2 = {"qadam_tarixi": ["qabul_qilish: ok", "stepwise: STOP"], "xulosa": "tayyor"}
        ombor = TahlilOmbori()
        t0 = time.time()
        job = ombor.boshlash(_SoxtaAgent([h1, h2], kutish=0.3), {"yosh": 1})
        self.assertLess(time.time() - t0, 0.2)
        self.assertTrue(ombor.ishlayapti())
        self.assertIsNone(ombor.boshlash(_SoxtaAgent([h1]), {"yosh": 2}))
        time.sleep(0.45)
        self.assertEqual(job["holat"], h1)
        self._kut(ombor)
        self.assertTrue(job["tugadi"])
        self.assertEqual(ish_holati(job)["xulosa"], "tayyor")

    def test_xato_xulosaga_aylanadi(self) -> None:
        """Agent yiqilsa ham tushunarli ehtiyotkor holat qaytadi."""
        ombor = TahlilOmbori()
        job = ombor.boshlash(_SoxtaAgent([{"qadam_tarixi": ["q"], "xulosa": ""}], xato=True), {})
        self._kut(ombor)
        self.assertIn("LLM yiqildi", ish_holati(job)["xulosa"])

    def test_bekor(self) -> None:
        """Bekor qilingach yangi ish boshlanadi, qisman natija saqlanadi."""
        ombor = TahlilOmbori()
        h1 = {"qadam_tarixi": ["qabul_qilish: ok"], "xulosa": ""}
        job = ombor.boshlash(_SoxtaAgent([h1, h1, h1], kutish=0.3), {})
        time.sleep(0.4)
        ombor.bekor_qilish()
        self.assertFalse(ombor.ishlayapti())
        qisman = ish_holati(job)
        self.assertIn("kutilmasdan", qisman["xulosa"])
        yangi = ombor.boshlash(_SoxtaAgent([h1]), {})
        self.assertIsNotNone(yangi)
        self.assertNotEqual(yangi["id"], job["id"])
        self._kut(ombor)


if __name__ == "__main__":
    unittest.main()
