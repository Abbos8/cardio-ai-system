"""StatPearls Markdown, belgi bo‘laklari va PDF matnini tekshiradi.

Tarmoq va BioClinicalBERT chaqirilmaydi. Tashxis emas.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag.ingest import belgilarga_ajrat, pdf_dan_matn
from rag.qollanma_pdf import pdf_yaroqli
from rag.statpearls_yukla import sarlavha_balli, xml_dan_yozuvlar, yozuvdan_markdown


_XML = """<?xml version="1.0" ?>
<PubmedArticleSet>
<PubmedBookArticle><BookDocument>
<PMID Version="1">29489148</PMID>
<ArticleIdList><ArticleId IdType="bookaccession">NBK482255</ArticleId></ArticleIdList>
<Book><PubDate><Year>2023</Year></PubDate></Book>
<ArticleTitle>Cardiogenic Shock</ArticleTitle>
<AuthorList><Author><LastName>Kosaraju</LastName><ForeName>Ateet</ForeName></Author></AuthorList>
<Abstract><AbstractText>Cardiogenic shock is a low cardiac output state with hypoperfusion. Clinical criteria include systolic pressure at or below 90 mm Hg. This sentence keeps the abstract long enough for the local RAG chunk filter in tests.</AbstractText>
<CopyrightInformation>Copyright 2023, StatPearls Publishing LLC.</CopyrightInformation></Abstract>
<Sections><Section><SectionTitle>Introduction</SectionTitle></Section>
<Section><SectionTitle>Treatment / Management</SectionTitle></Section></Sections>
</BookDocument></PubmedBookArticle>
</PubmedArticleSet>
"""


class KorpusTest(unittest.TestCase):
    """Bo‘lak o‘lchami va StatPearls yozuvi."""

    def test_belgi_bolak_ustma_ust(self) -> None:
        """500 belgidan oshgan matn bir necha bo‘lakka, ustma-ust bilan tushadi."""
        matn = "\n".join(["Heart failure natriuretic peptide echocardiography guideline."] * 40)
        bolaklar = belgilarga_ajrat(matn, chunk_size=500, overlap=50)
        self.assertGreaterEqual(len(bolaklar), 2)
        self.assertTrue(all(len(b) >= 80 for b in bolaklar))
        self.assertLessEqual(max(len(b) for b in bolaklar), 500 + 50)

    def test_statpearls_markdown(self) -> None:
        """XML dan sarlavha, NBK va xulosa Markdown ga tushadi."""
        yozuvlar = xml_dan_yozuvlar(_XML)
        self.assertIn("29489148", yozuvlar)
        md = yozuvdan_markdown(yozuvlar["29489148"])
        self.assertIn("# Cardiogenic Shock", md)
        self.assertIn("NBK482255", md)
        self.assertIn("## Xulosa", md)
        self.assertIn("## Bo‘limlar", md)
        self.assertIn("Treatment / Management", md)

    def test_pdf_matn(self) -> None:
        """pypdf haqiqiy PDF dan matn o‘qiydi."""
        from pypdf import PdfReader

        manba = Path("/tmp/esc_hf.pdf")
        if not pdf_yaroqli(manba):
            self.skipTest("namuna PDF yo‘q")
        matn = pdf_dan_matn(manba)
        self.assertIn("heart failure", matn.lower())
        self.assertGreater(len(matn), 500)
        self.assertGreater(len(PdfReader(str(manba)).pages), 1)

    def test_asosiy_maqola_nursingdan_ustun(self) -> None:
        """Nursing va Archived variant asosiy StatPearls sarlavhasidan past ball oladi."""
        asosiy = sarlavha_balli("Atrial Fibrillation", "Atrial Fibrillation")
        nursing = sarlavha_balli("Atrial Fibrillation", "Atrial Fibrillation (Nursing)")
        self.assertGreater(asosiy, nursing)
        self.assertGreater(asosiy, sarlavha_balli("Ventricular Tachycardia", "Pulseless Ventricular Tachycardia (Archived)"))

    def test_yaroqsiz_pdf_rad(self) -> None:
        """HTML fayl PDF deb qabul qilinmaydi."""
        with tempfile.TemporaryDirectory() as jild:
            yol = Path(jild) / "sahifa.pdf"
            yol.write_text("<html>not a pdf</html>", encoding="utf-8")
            self.assertFalse(pdf_yaroqli(yol))


if __name__ == "__main__":
    unittest.main()
