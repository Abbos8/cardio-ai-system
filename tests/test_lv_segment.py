"""LV area-length EF va segment_lv zaxirasi.

Zhang 2024 formulasi tekshiriladi. Og‘irlik yo‘q bo‘lsa DeepLab testi o‘tkazilmaydi.
Natija tashxis emas.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from tools.echo_segmenter import (
    MODEL_DEEPLAB,
    _deeplab_maskalar,
    _lv_ol,
    area_length_hajm,
    ef_foiz,
    ef_sikl_ortacha,
    ogirlik_yoli,
    segment_lv,
    sikl_juftliklari,
    tasnif_model_nomi,
)


class LvSegmentTest(unittest.TestCase):
    """Area-length va bo‘sh kadr holati."""

    def test_hajm_va_ef_formulasi(self) -> None:
        """8·A²/(3·π·L) va (EDV−ESV)/EDV qo‘lda hisobga mos."""
        edv = area_length_hajm(200, 20)
        esv = area_length_hajm(100, 15)
        self.assertIsNotNone(edv)
        self.assertIsNotNone(esv)
        kutilgan_ed = 8.0 * 200**2 / (3.0 * math.pi * 20)
        kutilgan_es = 8.0 * 100**2 / (3.0 * math.pi * 15)
        self.assertAlmostEqual(edv, kutilgan_ed, places=4)
        self.assertAlmostEqual(esv, kutilgan_es, places=4)
        foiz = ef_foiz(edv, esv)
        self.assertAlmostEqual(foiz, 100.0 * (kutilgan_ed - kutilgan_es) / kutilgan_ed, places=4)
        self.assertIsNone(area_length_hajm(0, 10))
        self.assertIsNone(ef_foiz(10, 10))

    def test_qisqa_klipda_ef_yoq(self) -> None:
        """21 kadrdan kam bo‘lsa EF o‘ylab topilmaydi."""
        ef, ed, es = ef_sikl_ortacha([10, 20, 5], [8, 9, 7])
        self.assertIsNone(ef)
        self.assertEqual(ed, [])
        self.assertEqual(es, [])

    def test_ikki_sikl_choqqi(self) -> None:
        """80 kadrli sinusda kamida bitta ED–ES jufti chiqadi."""
        i = np.arange(80)
        maydon = 100 + 40 * np.sin(2 * math.pi * i / 40)
        juft = sikl_juftliklari(maydon)
        self.assertGreaterEqual(len(juft), 1)
        for ed, es in juft:
            self.assertLess(ed, es)
            self.assertGreater(maydon[ed], maydon[es])

    def test_kadr_yoq(self) -> None:
        """Kadr berilmasa ok=False va EF yo‘q."""
        javob = segment_lv()
        self.assertFalse(javob.get("ok"))
        self.assertIsNone(javob.get("maska"))
        self.assertIsNone(javob.get("ef"))

    def test_ogirlik_bolsa_deeplab(self) -> None:
        """deeplabv3_resnet50.pt joyida bo‘lsa model nomi zhang2024-deeplab."""
        if not ogirlik_yoli().is_file() or ogirlik_yoli().stat().st_size < 1_000_000:
            self.skipTest("EchoNet LV og‘irligi yuklanmagan")
        self.assertEqual(tasnif_model_nomi(), MODEL_DEEPLAB)
        kadr = np.zeros((112, 112), dtype=np.uint8)
        kadr[20:90, 30:80] = 40
        kadr[40:70, 45:65] = 10
        maskalar = _deeplab_maskalar([kadr], _lv_ol())
        self.assertEqual(len(maskalar), 1)
        self.assertEqual(maskalar[0].shape, kadr.shape)
        self.assertTrue(set(np.unique(maskalar[0])).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
