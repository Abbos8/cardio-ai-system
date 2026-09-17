"""14-bosqich: PII gitda yo‘q, audit (model/C), ogohlantirish.

Tashxis o‘rnini bosmaydi. Klinik production emas.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_ILDIG = Path(__file__).resolve().parents[1]
_SRC = _ILDIG / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ["AUDIT_YOZ"] = "0"

from agents.chief_agent import ChiefCardiologist
from llm.client import model_qisqacha
from xavfsizlik.audit import (
    KLINIK_OGOHLANTIRISH,
    KLINIK_PRODUCTION,
    audit_saqla,
    audit_yig,
    c_manbalar,
)
from xavfsizlik.himoya import git_fayllarini_sirga_tekshir, gitda_taqiqlanganlar, kalit_maydonlarini_ol


class XavfsizlikTest(unittest.TestCase):
    """Git himoya, C manbalari va audit tarkibi."""

    def test_klinik_production_emas(self) -> None:
        """Demo production deb belgilanmasin."""
        self.assertFalse(KLINIK_PRODUCTION)
        self.assertIn("shifokor", KLINIK_OGOHLANTIRISH.lower())

    def test_c_manbalar(self) -> None:
        """C dan faqat [jild/fayl] id olinadi."""
        dalil = [
            "[medlineplus/Atrial-fibrillation] AF is a common arrhythmia ...",
            "[nhs/atrial-fibrillation] NHS page ...",
            "Seed matn prefikssiz ST-elevation ...",
        ]
        ids = c_manbalar(dalil)
        self.assertEqual(ids[0], "medlineplus/Atrial-fibrillation")
        self.assertEqual(ids[1], "nhs/atrial-fibrillation")
        self.assertIn("seed/cardiology", ids)
        self.assertTrue(all("[" not in i for i in ids))

    def test_model_qisqacha_kalitsiz(self) -> None:
        """Kalit yo‘qida model nomlari shablon, kalit maydoni yo‘q."""
        os.environ.pop("DEEPSEEK_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)
        qisqa = model_qisqacha()
        self.assertFalse(qisqa.get("llm_ulangan"))
        self.assertEqual(qisqa.get("chat"), "shablon")
        self.assertNotIn("api_key", qisqa)
        self.assertNotIn("api_kalit", qisqa)

    def test_audit_pii_va_signal_yoq(self) -> None:
        """Auditda signal, kalit va to‘liq C matni yo‘q."""
        holat = {
            "murakkablik": "oddiy",
            "amal": "STOP",
            "rag_dalillar": ["[esc/acs] long clinical chunk text " * 20],
            "fellow_natija": {"manba": "shablon", "model": "shablon"},
            "mdt_natija": {
                "medgemma_manba": "shablon",
                "qwen_manba": "shablon",
                "medgemma_model": "shablon",
                "qwen_model": "shablon",
            },
            "qadam_tarixi": ["qabul_qilish"],
            "bemor": {"ecg_signal": [0.1] * 100, "ism": "Test"},
        }
        yozuv = audit_yig(holat)
        self.assertEqual(yozuv["C"], ["esc/acs"])
        self.assertFalse(yozuv["klinik_production"])
        self.assertIn("shifokor", yozuv["ogohlantirish"].lower())
        self.assertEqual(kalit_maydonlarini_ol(yozuv), [])
        self.assertNotIn("ecg_signal", json.dumps(yozuv))
        self.assertNotIn("ism", yozuv)
        self.assertNotIn("long clinical chunk", json.dumps(yozuv))

    def test_audit_diskka(self) -> None:
        """AUDIT_YOZ=1 da JSON gitdan tashqari katalogga yoziladi."""
        os.environ["AUDIT_YOZ"] = "1"
        tmp = tempfile.mkdtemp(prefix="audit_")
        try:
            yozuv = audit_yig({"amal": "STOP", "rag_dalillar": [], "qadam_tarixi": []})
            yol = audit_saqla(yozuv, katalog=Path(tmp))
            self.assertIsNotNone(yol)
            assert yol is not None
            qayta = json.loads(yol.read_text(encoding="utf-8"))
            self.assertEqual(qayta.get("id"), yozuv["id"])
            self.assertFalse(qayta.get("klinik_production"))
        finally:
            os.environ["AUDIT_YOZ"] = "0"

    def test_agent_xulosa_audit(self) -> None:
        """Agent STOP da audit va ogohlantirish qaytaradi."""
        holat = ChiefCardiologist(rag=None, max_qadam=16).run(
            {"yosh": 41, "shikoyatlar": "tekshiruv"}
        )
        self.assertEqual(holat.get("amal"), "STOP")
        xulosa = holat.get("xulosa") or ""
        self.assertIn("shifokor", xulosa.lower())
        audit = holat.get("audit") or {}
        self.assertTrue(audit.get("modellar"))
        self.assertEqual((audit.get("modellar") or {}).get("chat"), "shablon")
        self.assertIn("C", audit)

    def test_git_sir_yoq(self) -> None:
        """Track qilingan fayllarda .env va kalit naqshi yo‘q."""
        taqiqlangan = gitda_taqiqlanganlar(_ILDIG)
        self.assertEqual(taqiqlangan, [])
        sir = git_fayllarini_sirga_tekshir(_ILDIG)
        self.assertEqual(sir, [])


if __name__ == "__main__":
    unittest.main()
