"""CardiacRAG indeksini quradi: sahifalarni yuklab, chunklab, FAISS ga yozadi.

Ishlatish (venv, loyiha ildizi):
  PYTHONPATH=src python src/rag/indeks_qur.py
  PYTHONPATH=src python src/rag/indeks_qur.py --qayta   # keshni yangilash
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rag.ingest import korpus_bolaklari
from rag.medical_rag import MedicalRAG, ODATIY_RAW_DIR
from rag.yuklab_ol import barcha_sahifalarni_yukla


def asosiy(qayta: bool = False, yuklama: bool = True) -> None:
    """Manbalarni yuklab, korpusni indekslaydi va retrieve ni tekshiradi.

    Args:
        qayta: True — mavjud FAISS keshini qayta qurish.
        yuklama: False — faqat diskdagi data/raw ni indekslash.

    Returns:
        None. Natija stdout da. Tashxis emas.
    """
    if yuklama:
        print("Sahifalar yuklanmoqda → data/raw/ ...")
        hisob = barcha_sahifalarni_yukla()
        print(
            f"Yuklash: yangi={hisob.get('ok')} bor={hisob.get('bor')} xato={hisob.get('xato')}"
        )
        for x in hisob.get("xabarlar") or []:
            print("  xato:", x)
    seed = Path(__file__).resolve().parent / "knowledge" / "cardiology_seed.txt"
    n_chunk = len(korpus_bolaklari(seed, ODATIY_RAW_DIR))
    print(f"Korpus chunklari (indekslashdan oldin): {n_chunk}")
    rag = MedicalRAG()
    soni = rag.urug_indeks(qayta_qur=qayta)
    print(f"FAISS: {soni} bo‘lak, BERT={rag.bert_ishlatildi}")
    for sorov in (
        "troponin chest pain ACS STEMI ECG",
        "heart failure NT-proBNP echocardiography",
        "atrial fibrillation anticoagulation",
    ):
        top = rag.retrieve(sorov, n=3)
        print("---", sorov)
        for i, t in enumerate(top, 1):
            print(f"  {i}. {t[:220].replace(chr(10), ' ')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CardiacRAG indeksini qurish")
    parser.add_argument("--qayta", action="store_true", help="Keshni qayta qurish")
    parser.add_argument("--siz-yuklama", action="store_true", help="Yuklamasdan faqat indeks")
    args = parser.parse_args()
    asosiy(qayta=args.qayta, yuklama=not args.siz_yuklama)
