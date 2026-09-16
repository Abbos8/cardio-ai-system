"""Asosiy klinik yo‘llar: lab/EKG/echo, bo‘sh, LangGraph limiti, shablon/hashing.

API kalitsiz ishlaydi. Tashxis o‘rnini bosmaydi.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

_ILDIG = Path(__file__).resolve().parents[1]
_SRC = _ILDIG / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agents.chief_agent import ChiefCardiologist
from agents.mdt import mdt_munozara
from llm.client import llm_chat, llm_mavjud
from tools.ecg_tool import namuna_12_tasma_ekg
from tools.lab_tool import process_lab

# medical_rag load_dotenv qiladi — kalitni test uchun qayta olib tashlash
def _kalitlarni_ol() -> None:
    """Test jarayonida API kalitlarini o‘chiradi (shablon yo‘li).

    Returns:
        None. Kalit qiymatini yozmaydi.
    """
    for nom in (
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "MEDGEMMA_API_KEY",
        "QWEN_VL_API_KEY",
        "FELLOW_VISION_MODEL",
    ):
        os.environ.pop(nom, None)


_kalitlarni_ol()


def _ehtiyot_stop(holat: dict) -> None:
    """STOP va ehtiyotkor xulosani talab qiladi.

    Args:
        holat: Agent run natijasi.

    Returns:
        None. AssertionError — yo‘l buzilgan.
    """
    assert holat.get("amal") == "STOP"
    xulosa = (holat.get("xulosa") or "").lower()
    assert xulosa
    assert "shifokor" in xulosa or "tashxis" in xulosa


def _agent(max_qadam: int = 16) -> ChiefCardiologist:
    """RAG/BERT yo‘q agent (shablon reja).

    Args:
        max_qadam: LangGraph qadam chegarasi.

    Returns:
        ChiefCardiologist.
    """
    return ChiefCardiologist(rag=None, max_qadam=max_qadam)


def _echo_fayl() -> dict:
    """Namuna A4C DICOM ni bemor maydoniga qo‘yadi.

    Returns:
        echo_fayllar lug‘ati.
    """
    yol = _SRC / "tools" / "namuna_echo_a4c.dcm"
    return {"echo_fayllar": [{"nom": yol.name, "bayt": yol.read_bytes()}]}


class BarqarorlikTest(unittest.TestCase):
    """Lab/EKG/echo kombinatsiyalari va zaxira rejimlari."""

    def setUp(self) -> None:
        """Har testda kalitsiz shablon rejimini tiklaydi."""
        _kalitlarni_ol()

    def test_kalitsiz_llm_none(self) -> None:
        """API yo‘qida llm_chat None qaytaradi (agent to‘xtamasligi uchun)."""
        self.assertFalse(llm_mavjud())
        self.assertIsNone(llm_chat([{"role": "user", "content": "ping"}]))

    def test_lab_csv_namuna(self) -> None:
        """Namuna lab CSV tokenlanadi."""
        csv_yol = _SRC / "tools" / "namuna_lab.csv"
        nat = process_lab(
            {
                "lab_fayl_bayt": csv_yol.read_bytes(),
                "lab_fayl_nomi": "namuna_lab.csv",
                "dorilar": "aspirin | 75 mg | once daily",
            }
        )
        self.assertTrue(nat.get("ok") or nat.get("rag_satr") or nat.get("qiymatlar"))
        satr = (str(nat.get("rag_satr") or "") + " " + str(nat.get("qiymatlar") or "")).lower()
        self.assertIn("kaliy", satr)
        token = " ".join(str(t) for t in (nat.get("tokenlar") or [])).lower()
        self.assertTrue("aspirin" in token or "aspirin" in satr)

    def test_bosh_bemor_stop(self) -> None:
        """Hech narsa yo‘q — STOP + ehtiyotkor xulosa, yiqilmaydi."""
        holat = _agent().run({"yosh": 40, "shikoyatlar": "tekshiruv"})
        _ehtiyot_stop(holat)
        fellow = holat.get("fellow_natija") or {}
        yoq = fellow.get("yoq_dalillar") or []
        self.assertTrue("ecg" in yoq or "echo" in yoq)

    def test_lab_only(self) -> None:
        """Faqat laboratoriya."""
        holat = _agent().run(
            {
                "yosh": 62,
                "laboratoriya": {"kaliy": 3.2, "troponin_i": 12.0},
                "dorilar": "aspirin | 75 mg | once daily",
            }
        )
        _ehtiyot_stop(holat)
        lab = holat.get("lab_natija") or {}
        self.assertTrue(lab.get("rag_satr") or lab.get("qiymatlar") or lab.get("ok"))
        self.assertFalse(_ecg_ok(holat))

    def test_ekg_only(self) -> None:
        """Faqat sintetik EKG."""
        holat = _agent().run(
            {
                "yosh": 58,
                "ecg_signal": namuna_12_tasma_ekg(sampling_rate=500.0, davomiylik=4.0),
                "sampling_rate": 500.0,
            }
        )
        _ehtiyot_stop(holat)
        self.assertTrue(_ecg_ok(holat) or (holat.get("ecg_technician_natija") or {}).get("ok"))
        self.assertFalse((holat.get("echo_natija") or {}).get("ok"))

    def test_echo_only(self) -> None:
        """Faqat echo DICOM."""
        bemor = {"yosh": 70, "shikoyatlar": "echo"}
        bemor.update(_echo_fayl())
        holat = _agent().run(bemor)
        _ehtiyot_stop(holat)
        echo = holat.get("echo_natija") or {}
        self.assertTrue(echo.get("ok"))
        self.assertTrue(echo.get("korinishlar"))
        mask = holat.get("echo_mask") or {}
        self.assertIn("ok", mask)

    def test_hammasi_bor(self) -> None:
        """Lab + EKG + echo birga; fellow yo‘q dalil o‘ylab topmaydi."""
        bemor = {
            "yosh": 62,
            "jins": "erkak",
            "shikoyatlar": "ko‘krak og‘rig‘i",
            "laboratoriya": {"kaliy": 3.2, "troponin_i": 88.0},
            "ecg_signal": namuna_12_tasma_ekg(sampling_rate=500.0, davomiylik=4.0),
            "sampling_rate": 500.0,
        }
        bemor.update(_echo_fayl())
        holat = _agent().run(bemor)
        _ehtiyot_stop(holat)
        fellow = holat.get("fellow_natija") or {}
        dalil = fellow.get("dalillar") or []
        self.assertTrue("lab" in dalil)
        self.assertNotIn("echo", fellow.get("yoq_dalillar") or [])
        self.assertEqual(fellow.get("manba"), "shablon")
        mdt = holat.get("mdt_natija")
        self.assertTrue(mdt is None or mdt.get("ok"))

    def test_langgraph_limiti(self) -> None:
        """Qadam limiti — STOP, xulosa ehtiyotkor."""
        holat = _agent(max_qadam=2).run(
            {
                "yosh": 50,
                "ecg_signal": namuna_12_tasma_ekg(sampling_rate=500.0, davomiylik=3.0),
                "sampling_rate": 500.0,
            }
        )
        _ehtiyot_stop(holat)
        birlash = " ".join(holat.get("qadam_tarixi") or []) + " " + str(holat.get("baho_sababi") or "")
        self.assertTrue("limit" in birlash.lower() or holat.get("amal") == "STOP")

    def test_hashing_rag_zaxira(self) -> None:
        """BERT o‘chirilganda hashing indeksi quriladi (asosiy indeksga tegmaydi)."""
        from rag.medical_rag import MedicalRAG

        eski_bert = os.environ.get("USE_BIOCLINICAL_BERT")
        eski_dir = os.environ.get("RAG_INDEX_DIR")
        tmp = tempfile.mkdtemp(prefix="rag_hash_")
        try:
            os.environ["USE_BIOCLINICAL_BERT"] = "0"
            os.environ["RAG_INDEX_DIR"] = tmp
            rag = MedicalRAG()
            rag.urug_indeks()
            self.assertFalse(rag.bert_ishlatildi)
            self.assertGreater(len(rag.bolaklar or []), 0)
            nat = rag.retrieve("atrial fibrillation ECG")
            self.assertTrue(nat)
        finally:
            if eski_bert is None:
                os.environ.pop("USE_BIOCLINICAL_BERT", None)
            else:
                os.environ["USE_BIOCLINICAL_BERT"] = eski_bert
            if eski_dir is None:
                os.environ.pop("RAG_INDEX_DIR", None)
            else:
                os.environ["RAG_INDEX_DIR"] = eski_dir

    def test_mdt_shablon_bosh(self) -> None:
        """VLM yo‘qida MDT shablon, yo‘q dalil o‘ylab topilmaydi."""
        nat = mdt_munozara("yosh=40", "(oraliq natija yo‘q)", bemor={})
        self.assertTrue(nat.get("ok"))
        self.assertEqual(nat.get("medgemma_manba"), "shablon")
        self.assertIn("echo", nat.get("yoq_dalillar") or [])


def _ecg_ok(holat: dict) -> bool:
    """EP yoki technician muvaffaqiyatini tekshiradi.

    Args:
        holat: Agent holati.

    Returns:
        True agar EKG o‘lchovi chiqqan bo‘lsa.
    """
    ep = holat.get("ep_natija") or {}
    ecg = holat.get("ecg_natija") or {}
    return bool(ep.get("ok") or ecg.get("ok"))


if __name__ == "__main__":
    unittest.main()
