"""EchoPrime 11 ko‘rinish: A5C, Doppler va teg xaritasi.

Og‘irlik bo‘lsa ConvNeXt chaqiriladi. Yo‘q bo‘lsa geometrik zaxira.
Natija tashxis emas.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tools.echo_view_model import (
    ECHO_KORINISHLAR,
    kadrlar_tasnif,
    korinish_nomi,
    ogirlik_yoli,
    tasnif_model_nomi,
    tegdan_korinish,
    yasama_echo_kadr,
)


class EchoKorinishTest(unittest.TestCase):
    """EchoPrime yorliq to‘plami va DICOM iboralari."""

    def test_on_bir_ichida_a5c_va_doppler(self) -> None:
        """Yirik ro‘yxat 11 ta va A5C hamda uch Doppler sinfini o‘z ichiga oladi."""
        self.assertEqual(len(ECHO_KORINISHLAR), 11)
        self.assertIn("A5C", ECHO_KORINISHLAR)
        self.assertIn("Apical_Doppler", ECHO_KORINISHLAR)
        self.assertIn("Doppler_Parasternal_Long", ECHO_KORINISHLAR)
        self.assertIn("Doppler_Parasternal_Short", ECHO_KORINISHLAR)

    def test_teg_a5c_doppler_va_eski_nom(self) -> None:
        """Apikal 5 va Doppler iboralari to‘g‘ri sinfga, PLAX esa parasternal longga tushadi."""
        self.assertEqual(tegdan_korinish({"series_description": "Apical 5 chamber"}), "A5C")
        self.assertEqual(tegdan_korinish({"protocol_name": "apical doppler PW"}), "Apical_Doppler")
        self.assertEqual(
            tegdan_korinish({"series_description": "Doppler parasternal long CW"}),
            "Doppler_Parasternal_Long",
        )
        self.assertEqual(tegdan_korinish({"view_name": "PLAX"}), "Parasternal_Long")
        self.assertEqual(korinish_nomi("PSAX-MV"), "Parasternal_Short")
        self.assertEqual(korinish_nomi("SUBCOSTAL-IVC"), "Subcostal")

    def test_kadr_ehtimoli_yigindisi(self) -> None:
        """Har bir yorliq uchun ehtimol bor va yig‘indi 1 ga yaqin."""
        for nom in ECHO_KORINISHLAR:
            eht = kadrlar_tasnif([yasama_echo_kadr(nom)])
            self.assertEqual(set(eht), set(ECHO_KORINISHLAR), nom)
            self.assertAlmostEqual(sum(eht.values()), 1.0, places=4)

    def test_ogirlik_bolsa_echoprime(self) -> None:
        """view_classifier.pt joyida bo‘lsa model nomi echoprime."""
        if not ogirlik_yoli().is_file():
            self.skipTest("EchoPrime og‘irligi yuklanmagan")
        self.assertEqual(tasnif_model_nomi(), "echoprime")
