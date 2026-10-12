"""CardiacRAG indeksini quradi: sahifalarni yuklab, chunklab, FAISS ga yozadi.

Ishlatish (venv, loyiha ildizi):
  PYTHONPATH=src python src/rag/indeks_qur.py
  PYTHONPATH=src python src/rag/indeks_qur.py --qayta
  PYTHONPATH=src python src/rag/indeks_qur.py --rebuild

--rebuild StatPearls va ochiq PDF larni yuklab, belgi bo‘laklari
(chunk_size=500, overlap=50) bilan FAISS ni noldan yozadi.
Indeks: data/processed/rag_index (CardiacRAG shu yerni o‘qiydi).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rag.ingest import CHUNK_BELGI, CHUNK_USTMA, korpus_bolaklari
from rag.medical_rag import INDEKS_FAYL, META_FAYL, MedicalRAG, ODATIY_INDEKS_DIR, ODATIY_RAW_DIR
from rag.qollanma_pdf import ochiq_pdflarni_yukla
from rag.statpearls_yukla import barcha_statpearls
from rag.yuklab_ol import barcha_sahifalarni_yukla


def _oldingi_bolak() -> int:
    """Diskdagi indeks meta dan oldingi bo‘lak sonini o‘qiydi.

    Returns:
        chunk_soni yoki 0 (indeks yo‘q).
    """
    meta = ODATIY_INDEKS_DIR / META_FAYL
    if not meta.exists():
        return 0
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    return int(data.get("chunk_soni") or 0)


def _hujjat_soni(jild: Path) -> int:
    """Jilddagi indekslanadigan hujjat fayllarini sanaydi.

    Args:
        jild: data/raw ostidagi manba papkasi.

    Returns:
        html/pdf/md/txt soni. .url.txt hisobga kirmaydi.
    """
    if not jild.exists():
        return 0
    soni = 0
    for fayl in jild.rglob("*"):
        if not fayl.is_file() or fayl.name.endswith(".url.txt"):
            continue
        if fayl.suffix.lower() in {".html", ".htm", ".pdf", ".txt", ".md"}:
            soni += 1
    return soni


def _prefiks_soni(bolaklar: list, prefiks: str) -> int:
    """Manba prefiksi bilan boshlangan bo‘laklarni sanaydi.

    Args:
        bolaklar: Indeksdagi matnlar.
        prefiks: Masalan `[statpearls/`.

    Returns:
        Shu manbadagi vektor bo‘laklari.
    """
    return sum(1 for bolak in bolaklar if str(bolak).startswith(prefiks))


def asosiy(qayta: bool = False, yuklama: bool = True, rebuild: bool = False) -> None:
    """Manbalarni yuklab, korpusni indekslaydi va retrieve ni tekshiradi.

    Args:
        qayta: True — mavjud FAISS keshini qayta qurish.
        yuklama: False — faqat diskdagi data/raw ni indekslash.
        rebuild: True — StatPearls + ochiq PDF, so‘ng indeksni noldan yozish.

    Returns:
        None. Natija stdout da. Tashxis emas.
    """
    oldingi = _oldingi_bolak()
    if rebuild:
        qayta = True
        yuklama = False
        print("StatPearls yuklanmoqda → data/raw/statpearls/ ...")
        sp = barcha_statpearls()
        print(f"StatPearls: yangi={sp.get('ok')} bor={sp.get('bor')} xato={sp.get('xato')}")
        for xabar in sp.get("xabarlar") or []:
            print(" ", xabar)
        print("Yo‘riqnoma PDF yuklanmoqda → data/raw/guidelines/ ...")
        pdf = ochiq_pdflarni_yukla()
        print(f"Yo‘riqnoma PDF: yangi={pdf.get('ok')} bor={pdf.get('bor')} xato={pdf.get('xato')}")
        for xabar in pdf.get("xabarlar") or []:
            print("  xato:", xabar)
    elif yuklama:
        print("Sahifalar yuklanmoqda → data/raw/ ...")
        hisob = barcha_sahifalarni_yukla()
        print(
            f"Yuklash: yangi={hisob.get('ok')} bor={hisob.get('bor')} xato={hisob.get('xato')}"
        )
        for x in hisob.get("xabarlar") or []:
            print("  xato:", x)
    n_sp = _hujjat_soni(ODATIY_RAW_DIR / "statpearls")
    n_pdf = _hujjat_soni(ODATIY_RAW_DIR / "guidelines")
    n_mayo = _hujjat_soni(ODATIY_RAW_DIR / "mayo")
    print(f"Hujjatlar diskda: statpearls={n_sp} guidelines={n_pdf} mayo={n_mayo}")
    seed = Path(__file__).resolve().parent / "knowledge" / "cardiology_seed.txt"
    reja = korpus_bolaklari(seed, ODATIY_RAW_DIR, chunk_size=CHUNK_BELGI, overlap=CHUNK_USTMA)
    print(
        "Bo‘laklar (indekslashdan oldin): "
        f"jami={len(reja)} statpearls={_prefiks_soni(reja, '[statpearls/')} "
        f"guidelines={_prefiks_soni(reja, '[guidelines/')} mayo={_prefiks_soni(reja, '[mayo/')}"
    )
    print(
        f"FAISS yozilmoqda (chunk_size={CHUNK_BELGI}, overlap={CHUNK_USTMA}, "
        f"oldingi={oldingi}). BioClinicalBERT bir necha daqiqa olishi mumkin."
    )
    rag = MedicalRAG()
    soni = rag.urug_indeks(qayta_qur=qayta, chunk_size=CHUNK_BELGI, overlap=CHUNK_USTMA)
    qoshilgan = soni - oldingi
    print(f"FAISS: {soni} bo‘lak ({ODATIY_INDEKS_DIR / INDEKS_FAYL}), BERT={rag.bert_ishlatildi}")
    print(
        "Yangi manba bo‘laklari: "
        f"statpearls={_prefiks_soni(rag.bolaklar, '[statpearls/')} "
        f"guidelines={_prefiks_soni(rag.bolaklar, '[guidelines/')} "
        f"mayo={_prefiks_soni(rag.bolaklar, '[mayo/')} "
        f"(oldingiga nisbatan {qoshilgan:+d} bo‘lak)"
    )
    print(f"Yangi hujjat fayllari shu yurishda kiritilgan papkalar: statpearls={n_sp}, guidelines={n_pdf}, mayo={n_mayo}")
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
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="StatPearls + ochiq PDF yuklab, FAISS ni noldan qayta yozish",
    )
    parser.add_argument("--siz-yuklama", action="store_true", help="Yuklamasdan faqat indeks")
    args = parser.parse_args()
    asosiy(qayta=args.qayta, yuklama=not args.siz_yuklama, rebuild=args.rebuild)
