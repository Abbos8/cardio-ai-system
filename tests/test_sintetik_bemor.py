"""Uch sintetik holat: fayl, EKG, echo DICOM va agent paketi.

API kalitsiz. Natija tashxis emas — loyiha oqimini solishtirish uchun.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_ILDIG = Path(__file__).resolve().parents[1]
_SRC = _ILDIG / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tools.echo_view_model import ECHO_KORINISHLAR, tegdan_korinish
from tools.echo_yuklash import echo_fayldan_kadrlar
from tools.ekg_yuklash import csv_dan_ekg
from tools.sintetik_bemor import HOLAT_IDLAR, sintetik_yukla


def _kalitlarni_ol() -> None:
    """Test jarayonida API kalitlarini o‘chiradi.

    Returns:
        None. Kalit qiymatini yozmaydi.
    """
    for nom in (
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "MEDGEMMA_API_KEY",
        "QWEN_VL_API_KEY",
        "FELLOW_VISION_API_KEY",
    ):
        os.environ.pop(nom, None)
    os.environ["AUDIT_YOZ"] = "0"


_kalitlarni_ol()


class SintetikBemorTest(unittest.TestCase):
    """Sodda, murakkab va juda murakkab fayllar bir-biridan ajraladi."""

    @classmethod
    def setUpClass(cls) -> None:
        """Uch holatni bir marta yuklaydi (DICOM yozuvi bilan).

        Returns:
            None. cls.holatlar keyin o‘qiladi.
        """
        _kalitlarni_ol()
        cls.holatlar = {nom: sintetik_yukla(nom) for nom in HOLAT_IDLAR}

    def test_uchta_fayl_va_signal(self) -> None:
        """JSON, EKG CSV va (n, 12) signal bor."""
        for nom, holat in self.holatlar.items():
            self.assertTrue(Path(holat["fayllar"]["json"]).is_file(), nom)
            self.assertTrue(Path(holat["fayllar"]["ekg"]).is_file(), nom)
            signal = holat["signal"]
            self.assertEqual(signal.shape[1], 12, nom)
            self.assertGreater(signal.shape[0], 500, nom)
            oqilgan = csv_dan_ekg(Path(holat["fayllar"]["ekg"]).read_bytes(), sampling_rate_hint=500)
            self.assertTrue(oqilgan.get("ok"), oqilgan.get("xabar"))
            self.assertEqual(oqilgan["signal"].shape, signal.shape)

    def test_laboratoriya_darajalari(self) -> None:
        """Biomarkerlar sodda → murakkab → juda murakkab tomon og‘irlashadi."""
        sodda = self.holatlar["sodda"]["laboratoriya"]
        murakkab = self.holatlar["murakkab"]["laboratoriya"]
        juda = self.holatlar["juda_murakkab"]["laboratoriya"]
        self.assertLess(sodda["troponin_i"], 34)
        self.assertGreaterEqual(murakkab["troponin_i"], 34)
        self.assertLess(murakkab["troponin_i"], 100)
        self.assertGreaterEqual(juda["troponin_i"], 100)
        self.assertLess(sodda["nt_probnp"], 300)
        self.assertGreaterEqual(murakkab["nt_probnp"], 300)
        self.assertGreaterEqual(juda["nt_probnp"], 900)
        self.assertLess(sodda["kaliy"], 5.1)
        self.assertGreater(juda["kaliy"], 5.5)
        self.assertEqual(self.holatlar["sodda"]["yurak_chastotasi"], 72)
        self.assertGreater(self.holatlar["murakkab"]["yurak_chastotasi"], 100)
        self.assertLess(self.holatlar["juda_murakkab"]["yurak_chastotasi"], 50)

    def test_echo_dicom_korinishlari(self) -> None:
        """Echo soni 2 / 6 / 11 va DICOM tegi ko‘rinishni beradi."""
        kutilgan = {"sodda": 2, "murakkab": 6, "juda_murakkab": len(ECHO_KORINISHLAR)}
        for nom, son in kutilgan.items():
            echo = self.holatlar[nom]["echo_fayllar"]
            self.assertEqual(len(echo), son, nom)
            for element in echo:
                yol = Path(self.holatlar[nom]["fayllar"]["echo"][0]).parent / element["nom"]
                self.assertTrue(yol.is_file(), str(yol))
                oq = echo_fayldan_kadrlar(element["nom"], element["bayt"])
                self.assertTrue(oq.get("ok"), oq.get("xabar"))
                self.assertEqual(tegdan_korinish(oq.get("meta")), element["korinish"])


if __name__ == "__main__":
    unittest.main()
