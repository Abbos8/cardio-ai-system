"""Kardiologik AI agent uchun gibrid RAG: FAISS + BioClinicalBERT + TF-IDF.

1-bosqich: vektor o‘xshashligi bo‘yicha 3n ta yaqin tibbiy bo‘lak.
2-bosqich: TF-IDF, tibbiy lug‘at og‘irligi (MW) va dastlabki 30% uchun PB=1.2.
Natija shifokor tashxisining o‘rnini bosmaydi.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import faiss  # Eng yaqin vektorlarni tez qidirish uchun
import numpy as np  # Embeddinglarni massiv sifatida saqlash uchun
import torch  # BioClinicalBERT ni GPU/CPU da ishlatish uchun
from transformers import AutoModel, AutoTokenizer  # BERT model va tokenizer

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

_LOG = logging.getLogger(__name__)

# Klinik matnlar uchun oldindan o‘qitilgan BioClinicalBERT nomi
MODEL_NOMI = "emilyalsentzer/Bio_ClinicalBERT"
# BioClinicalBERT yashirin qatlam o‘lchami (768)
EMBEDDING_OLCHAMI = 768
# Yakuniy kontekst bo‘laklari n; FAISS dan 3n nomzod
N_KONTEKST = 3
BOSQICH1_SONI = 9
BOSQICH2_SONI = 3
# Matn boshidagi kalit so‘z uchun pozitsion bonus
POZITSION_BONUS = 1.2
BOSH_ULUSH = 0.30
# Tibbiy lug‘at og‘irligi (MW)
MW_KOEF = 1.5

TIBBIY_LUGAT = {
    "ecg",
    "ekg",
    "qrs",
    "qt",
    "pr",
    "stemi",
    "nstemi",
    "troponin",
    "bnp",
    "ntprobnp",
    "heart",
    "failure",
    "arrhythmia",
    "bradycardia",
    "tachycardia",
    "echo",
    "ejection",
    "interval",
    "infarction",
    "ischemia",
    "potassium",
    "hrv",
    "sdnn",
    "guidelines",
    "acs",
    "angina",
}
# Uzun tibbiy matnni kesish chegarasi (token)
MAX_TOKEN = 256
# FAISS + bo‘laklar disk kesh (har UI ochilganda qayta embedding qilmaslik)
LOYIHA_ILDZI = Path(__file__).resolve().parents[2]
ODATIY_INDEKS_DIR = LOYIHA_ILDZI / "data" / "processed" / "rag_index"
ODATIY_RAW_DIR = LOYIHA_ILDZI / "data" / "raw"
INDEKS_FAYL = "index.faiss"
BOLAK_FAYL = "chunks.json"
META_FAYL = "meta.json"

# Jarayon ichida tokenizer/model bir marta yuklanadi (HF keshdan)
_BERT_KESH: dict = {}

# Kalit so‘z tanlashda e’tiborsiz qoldiriladigan oddiy so‘zlar
TOXTATISH_SOZLARI = {
    "a",
    "an",
    "and",
    "the",
    "of",
    "in",
    "on",
    "for",
    "to",
    "with",
    "is",
    "are",
    "this",
    "that",
    "va",
    "yoki",
    "uchun",
    "bilan",
    "bu",
    "shu",
    "ham",
    "dan",
    "ga",
    "ni",
    "ning",
}


def _bert_yoqilgan() -> bool:
    """USE_BIOCLINICAL_BERT muhit bayrog‘ini o‘qiydi.

    Returns:
        True — BioClinicalBERT; False — hashing zaxirasi.
    """
    return os.getenv("USE_BIOCLINICAL_BERT", "1").strip() in {"1", "true", "True", "yes"}


def _indeks_katalogi() -> Path:
    """FAISS kesh katalogini qaytaradi (RAG_INDEX_DIR yoki odatiy yo‘l).

    Returns:
        Mavjud bo‘lmasa ham yo‘l; yozishda mkdir qilinadi.
    """
    maxsus = (os.getenv("RAG_INDEX_DIR") or "").strip()
    return Path(maxsus) if maxsus else ODATIY_INDEKS_DIR


def _bert_yukla(model_nomi: str, qurilma: str) -> Tuple[object, object]:
    """BioClinicalBERT tokenizer va modelini jarayon keshidan oladi.

    Args:
        model_nomi: Hugging Face identifikatori.
        qurilma: cuda yoki cpu.

    Returns:
        (tokenizer, model). Yuklash xatosida istisno ko‘tariladi.
    """
    kalit = f"{model_nomi}::{qurilma}"
    if kalit in _BERT_KESH:
        return _BERT_KESH[kalit]
    tokenizer = AutoTokenizer.from_pretrained(model_nomi)
    model = AutoModel.from_pretrained(model_nomi)
    model.to(qurilma)
    model.eval()
    _BERT_KESH[kalit] = (tokenizer, model)
    _LOG.info("BioClinicalBERT yuklandi: %s (%s)", model_nomi, qurilma)
    return tokenizer, model


def _manba_imzo(yol: Path, model_nomi: str, bert: bool, raw_katalog: Optional[Path] = None) -> str:
    """Seed + data/raw fayllari va embedding rejimidan kesh kalitini hisoblaydi.

    Args:
        yol: cardiology_seed.txt yo‘li.
        model_nomi: BERT identifikatori.
        bert: Haqiqiy BERT ishlatilganmi.
        raw_katalog: Yuklab olingan HTML/PDF. None — data/raw.

    Returns:
        SHA-256 hex. Korpus o‘zgarsa indeks qayta quriladi.
    """
    from rag.ingest import CHUNK_SOZ, CHUNK_USTMA_UST

    h = hashlib.sha256()
    h.update(yol.read_bytes() if yol.exists() else b"")
    raw = Path(raw_katalog) if raw_katalog else ODATIY_RAW_DIR
    if raw.exists():
        for fayl in sorted(raw.rglob("*")):
            if not fayl.is_file() or fayl.name.endswith(".url.txt"):
                continue
            if fayl.suffix.lower() not in {".html", ".htm", ".pdf", ".txt", ".md"}:
                continue
            rel = str(fayl.relative_to(raw)).encode()
            h.update(rel)
            h.update(str(fayl.stat().st_size).encode())
            h.update(hashlib.sha256(fayl.read_bytes()).digest())
    h.update(f"|{model_nomi}|{int(bert)}|{EMBEDDING_OLCHAMI}|{CHUNK_SOZ}|{CHUNK_USTMA_UST}".encode())
    return h.hexdigest()


class MedicalRAG:
    """FAISS va BioClinicalBERT asosidagi ikki bosqichli tibbiy RAG."""

    def __init__(self, model_nomi: str = MODEL_NOMI) -> None:
        """BioClinicalBERT ni bir marta yuklaydi va bo‘sh FAISS indeksini ochadi.

        Args:
            model_nomi: Hugging Face dagi klinik BERT identifikatori.

        Returns:
            None. Indeks hali bo‘sh; urug_indeks yoki index_documents chaqiriladi.
        """
        self.model_nomi = model_nomi
        self.qurilma = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.bert_ishlatildi = False
        self.bert_xato: Optional[str] = None
        if _bert_yoqilgan():
            try:
                self.tokenizer, self.model = _bert_yukla(model_nomi, self.qurilma)
                self.bert_ishlatildi = True
            except Exception as xato:
                self.tokenizer = None
                self.model = None
                self.bert_xato = str(xato)
                _LOG.warning("BioClinicalBERT yuklanmadi, hashing zaxirasi: %s", xato)
        self.indeks = faiss.IndexFlatIP(EMBEDDING_OLCHAMI)
        self.bolaklar: List[str] = []  # Indeksdagi asl tibbiy matn bo‘laklari
        self._tf_idf_tayyor: bool = False
        self._idf: dict = {}
        self._df: dict = {}
        self.indeks_katalogi = _indeks_katalogi()

    def _ortacha_pul(self, yashirin: torch.Tensor, niqob: torch.Tensor) -> torch.Tensor:
        """Token vektorlarini e’tibor niqobi bilan o‘rtacha qiladi.

        Args:
            yashirin: Modelning oxirgi yashirin holatlari [B, T, H].
            niqob: Haqiqiy tokenlar 1, padding 0 [B, T].

        Returns:
            Har bir matn uchun bitta 768 o‘lchamli vektor [B, H].
        """
        kengaytirilgan = niqob.unsqueeze(-1).float()  # Niqobni [B, T, 1] shakliga keltiramiz
        yigindi = (yashirin * kengaytirilgan).sum(dim=1)  # Faqat haqiqiy tokenlarni qo‘shamiz
        soni = kengaytirilgan.sum(dim=1).clamp(min=1e-9)  # Nolga bo‘linishni oldini olamiz
        return yigindi / soni  # O‘rtacha pooling — jumla embeddingi

    def _embedding_yarat(self, matnlar: Sequence[str]) -> np.ndarray:
        """BioClinicalBERT orqali matnlarni normallangan vektorga aylantiradi.

        Args:
            matnlar: Embeddingga o‘giriladigan tibbiy matnlar.

        Returns:
            float32 massiv [N, 768], FAISS IndexFlatIP uchun L2-normallangan.
        """
        if self.model is None or self.tokenizer is None:
            massiv = np.zeros((len(matnlar), EMBEDDING_OLCHAMI), dtype="float32")
            for i, matn in enumerate(matnlar):
                for soz in str(matn).lower().split():
                    massiv[i, hash(soz) % EMBEDDING_OLCHAMI] += 1.0
            faiss.normalize_L2(massiv)
            return massiv
        kod = self.tokenizer(
            list(matnlar),
            padding=True,
            truncation=True,
            max_length=MAX_TOKEN,
            return_tensors="pt",
        )
        kod = {kalit: qiymat.to(self.qurilma) for kalit, qiymat in kod.items()}
        with torch.no_grad():
            chiqish = self.model(**kod)
        vektorlar = self._ortacha_pul(chiqish.last_hidden_state, kod["attention_mask"])
        massiv = vektorlar.cpu().numpy().astype("float32")
        faiss.normalize_L2(massiv)
        return massiv

    def index_documents(self, bolaklar: Sequence[str]) -> int:
        """Tibbiy matn bo‘laklarini BioClinicalBERT + FAISS indeksiga yozadi.

        Args:
            bolaklar: Kardiologiya/EKG/lab ma’lumotlaridan olingan matn bo‘laklari.

        Returns:
            Indeksga qo‘shilgan bo‘laklar soni.
        """
        toza = [b.strip() for b in bolaklar if b and b.strip()]  # Bo‘sh qatorlarni tashlaymiz
        if not toza:  # Hech narsa qolmasa
            return 0  # Indeks o‘zgarmaydi
        vektorlar = self._embedding_yarat(toza)  # Har bir bo‘lakning vektorini hisoblaymiz
        self.indeks.add(vektorlar)
        self.bolaklar.extend(toza)
        self._tf_idf_tayyor = False
        return len(toza)

    def indeksni_saqla(self, katalog: Optional[Path] = None, manba: Optional[Path] = None) -> Path:
        """FAISS indeksini va bo‘laklarni diskka yozadi.

        Args:
            katalog: Saqlash jildi. None — data/processed/rag_index.
            manba: Seed .txt (kesh kaliti uchun).

        Returns:
            Yozilgan katalog yo‘li. Keyingi UI ochilishida qayta hisoblanmaydi.
        """
        katalog = Path(katalog) if katalog else self.indeks_katalogi
        katalog.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.indeks, str(katalog / INDEKS_FAYL))
        (katalog / BOLAK_FAYL).write_text(
            json.dumps(self.bolaklar, ensure_ascii=False, indent=0),
            encoding="utf-8",
        )
        imzo = _manba_imzo(manba, self.model_nomi, self.bert_ishlatildi) if manba else ""
        meta = {
            "model_nomi": self.model_nomi,
            "embedding_olchami": EMBEDDING_OLCHAMI,
            "bert_ishlatildi": self.bert_ishlatildi,
            "chunk_soni": len(self.bolaklar),
            "manba_imzo": imzo,
        }
        (katalog / META_FAYL).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        _LOG.info("FAISS indeks saqlandi: %s (%s bo‘lak)", katalog, len(self.bolaklar))
        return katalog

    def indeksni_yukla(self, katalog: Optional[Path] = None, manba: Optional[Path] = None) -> bool:
        """Diskdagi FAISS indeksini o‘qiydi, agar kesh kaliti mos kelsa.

        Args:
            katalog: Kesh jildi.
            manba: Seed .txt; o‘zgargan bo‘lsa False (qayta qurish kerak).

        Returns:
            True — indeks tayyor; False — qayta embedding kerak.
        """
        katalog = Path(katalog) if katalog else self.indeks_katalogi
        indeks_yol = katalog / INDEKS_FAYL
        bolak_yol = katalog / BOLAK_FAYL
        meta_yol = katalog / META_FAYL
        if not (indeks_yol.exists() and bolak_yol.exists() and meta_yol.exists()):
            return False
        try:
            meta = json.loads(meta_yol.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        if int(meta.get("embedding_olchami") or 0) != EMBEDDING_OLCHAMI:
            return False
        if bool(meta.get("bert_ishlatildi")) != self.bert_ishlatildi:
            return False
        if str(meta.get("model_nomi") or "") != self.model_nomi:
            return False
        if manba is not None:
            kutilgan = _manba_imzo(manba, self.model_nomi, self.bert_ishlatildi)
            if str(meta.get("manba_imzo") or "") != kutilgan:
                return False
        try:
            bolaklar = json.loads(bolak_yol.read_text(encoding="utf-8"))
            indeks = faiss.read_index(str(indeks_yol))
        except Exception as xato:
            _LOG.warning("FAISS kesh o‘qilmadi: %s", xato)
            return False
        if not isinstance(bolaklar, list) or indeks.ntotal != len(bolaklar):
            return False
        self.indeks = indeks
        self.bolaklar = [str(b) for b in bolaklar]
        self._tf_idf_tayyor = False
        _LOG.info("FAISS indeks diskdan: %s (%s bo‘lak)", katalog, len(self.bolaklar))
        return True

    def _kalit_sozlar(self, matn: str) -> set:
        """So‘rovdan qisqa, ma’noli kalit so‘zlarni ajratadi.

        Args:
            matn: Foydalanuvchi yoki shifokor so‘rovi.

        Returns:
            To‘xtatish so‘zlarisiz kichik harfli so‘zlar to‘plami.
        """
        kichik = matn.lower()  # Katta-kichik farqini yo‘qotamiz
        qismlar = re.findall(r"[a-zA-Zа-яёўғҳқʼ']+", kichik)  # Faqat so‘z belgilari
        return {s for s in qismlar if s not in TOXTATISH_SOZLARI and len(s) > 2}  # Qisqa/oddiy so‘zlarni tashlaymiz

    def _tf_idf_qur(self) -> None:
        """Barcha bo‘laklar bo‘yicha IDF qiymatlarini hisoblaydi.

        Returns:
            None. retrieve() 2-bosqichi uchun.
        """
        self._df = {}
        n = len(self.bolaklar) or 1
        for bolak in self.bolaklar:
            for soz in self._kalit_sozlar(bolak):
                self._df[soz] = self._df.get(soz, 0) + 1
        self._idf = {s: math.log((1 + n) / (1 + d)) + 1.0 for s, d in self._df.items()}
        self._tf_idf_tayyor = True

    def _tf(self, matn: str) -> dict:
        """Matndagi so‘z chastotasini (TF) hisoblaydi.

        Args:
            matn: So‘rov yoki bo‘lak.

        Returns:
            so‘z → nisbiy chastota.
        """
        sozlar = list(self._kalit_sozlar(matn))
        if not sozlar:
            return {}
        hisob: dict = {}
        for s in sozlar:
            hisob[s] = hisob.get(s, 0) + 1
        n = float(len(sozlar))
        return {s: c / n for s, c in hisob.items()}

    def _tfidf_ball(self, sorov: str, bolak: str) -> float:
        """So‘rov va bo‘lak TF-IDF vektorlarining kosinusga yaqin ichki ko‘paytmasi.

        Args:
            sorov: Shifokor/agent so‘rovi.
            bolak: Nomzod matn.

        Returns:
            Moslik bali ≥ 0.
        """
        if not self._tf_idf_tayyor:
            self._tf_idf_qur()
        q_tf = self._tf(sorov)
        b_tf = self._tf(bolak)
        ball = 0.0
        for soz, qv in q_tf.items():
            idf = self._idf.get(soz, 1.0)
            ball += qv * idf * b_tf.get(soz, 0.0) * idf
        return ball

    def _mw_ball(self, bolak: str, kalitlar: set) -> float:
        """Tibbiy lug‘atdagi mos so‘zlar uchun MW og‘irligi.

        Args:
            bolak: Nomzod matn.
            kalitlar: So‘rov kalitlari.

        Returns:
            Qo‘shimcha ball.
        """
        bolak_soz = self._kalit_sozlar(bolak)
        mos = (kalitlar | bolak_soz) & TIBBIY_LUGAT
        return MW_KOEF * float(len(mos))

    def _pb_koef(self, bolak: str, kalitlar: set) -> float:
        """Kalit so‘z matnning dastlabki 30% qismida bo‘lsa 1.2 bonus.

        Args:
            bolak: Nomzod matn.
            kalitlar: So‘rov kalitlari.

        Returns:
            1.0 yoki POZITSION_BONUS.
        """
        if not kalitlar or not bolak:
            return 1.0
        chegara = max(1, int(len(bolak) * BOSH_ULUSH))
        bosh = bolak[:chegara].lower()
        for k in kalitlar:
            if k in bosh:
                return POZITSION_BONUS
        return 1.0

    def _gibrid_ball(self, sorov: str, bolak: str, kalitlar: set) -> float:
        """TF-IDF + MW, so‘ng PB koeffitsiyenti.

        Args:
            sorov: Klinik so‘rov.
            bolak: FAISS nomzodi.
            kalitlar: Kalit so‘z to‘plami.

        Returns:
            Yakuniy saralash bali.
        """
        asos = self._tfidf_ball(sorov, bolak) + self._mw_ball(bolak, kalitlar)
        return asos * self._pb_koef(bolak, kalitlar)

    def retrieve(
        self,
        sorov: str,
        bosqich1: int = BOSQICH1_SONI,
        bosqich2: int = BOSQICH2_SONI,
        n: Optional[int] = None,
    ) -> List[str]:
        """Gibrid RAG: FAISS dan 3n, so‘ng TF-IDF+MW+PB bilan n ta C.

        Args:
            sorov: Klinik savol (masalan, EKG yoki lab topilmalari).
            bosqich1: FAISS nomzodlari (odatiy 9 = 3n).
            bosqich2: Yakuniy n (odatiy 3).
            n: Agar berilsa, bosqich2=n, bosqich1=3n.

        Returns:
            Eng muhim n tibbiy matn. Tashxis o‘rnini bosmaydi.
        """
        if n is not None:
            bosqich2 = n
            bosqich1 = 3 * n
        if self.indeks.ntotal == 0:  # Hali hech qanday hujjat indekslanmagan
            return []  # Qidiriladigan narsa yo‘q
        if not sorov or not sorov.strip():  # Bo‘sh so‘rov
            return []  # Hech narsa qaytarmaymiz
        sorov_vek = self._embedding_yarat([sorov.strip()])  # So‘rovni BioClinicalBERT vektoriga
        k = min(bosqich1, self.indeks.ntotal)  # Hujjat kam bo‘lsa, mavjudlarini olamiz
        _ballar, indekslar = self.indeks.search(sorov_vek, k)  # 1-bosqich: FAISS 9 ta yaqin qo‘shni
        nomzodlar: List[str] = []  # 1-bosqich natijalari
        for idx in indekslar[0]:  # FAISS qaytargan indekslarni aylanamiz
            if idx < 0:  # FAISS bo‘sh o‘rinni -1 qilib qo‘yishi mumkin
                continue  # Bunday indekslarni o‘tkazib yuboramiz
            nomzodlar.append(self.bolaklar[int(idx)])  # Asl tibbiy matnni qo‘shamiz
        kalitlar = self._kalit_sozlar(sorov)
        tartiblangan = sorted(
            nomzodlar,
            key=lambda b: self._gibrid_ball(sorov, b, kalitlar),
            reverse=True,
        )
        return tartiblangan[:bosqich2]

    def urug_indeks(self, yol: Optional[Path] = None, qayta_qur: bool = False) -> int:
        """Seed + data/raw korpusini indekslaydi; kesh mos bo‘lsa diskdan.

        Args:
            yol: Ixtiyoriy seed .txt; None — knowledge/cardiology_seed.txt.
            qayta_qur: True — keshni e’tiborsiz qoldirib qayta hisoblash.

        Returns:
            Indeksdagi bo‘laklar soni.
        """
        from rag.ingest import korpus_bolaklari

        if yol is None:
            yol = Path(__file__).resolve().parent / "knowledge" / "cardiology_seed.txt"
        if qayta_qur:
            self.indeks = faiss.IndexFlatIP(EMBEDDING_OLCHAMI)
            self.bolaklar = []
            self._tf_idf_tayyor = False
        elif self.indeks.ntotal > 0 and self.bolaklar:
            return len(self.bolaklar)
        if not qayta_qur and self.indeksni_yukla(manba=yol):
            return len(self.bolaklar)
        bolaklar = korpus_bolaklari(yol, ODATIY_RAW_DIR)
        soni = self.index_documents(bolaklar)
        if soni:
            self.indeksni_saqla(manba=yol)
        return soni

    def reja_tuz(self, sorov: str, kontekst: Optional[Sequence[str]] = None) -> List[dict]:
        """Kontekst C asosida tartiblangan klinik qadamlar P ni tuzadi.

        Args:
            sorov: Bemor/shifokor so‘rovi.
            kontekst: retrieve() natijasi; None bo‘lsa o‘zi qidiradi.

        Returns:
            [{id, vosita, tavsif, holat}]. DeepSeek bo‘lsa undan, aks holda shablon.
        """
        from llm.client import llm_json

        c_royxat = list(kontekst) if kontekst is not None else self.retrieve(sorov)
        c_matn = "\n".join(c_royxat) if c_royxat else "(kontekst yo‘q)"
        zaxira = {
            "qadamlar": [
                {"id": "p1", "vosita": "lab_technician", "tavsif": "Laboratoriya va dori tarixini tokenlash"},
                {"id": "p2", "vosita": "ecg_technician", "tavsif": "12 tasmali EKG ni tozalash va filtr"},
                {
                    "id": "p3",
                    "vosita": "electrophysiologist",
                    "tavsif": "QRS, PR, QT, HRV va to‘lqin nuqtalari",
                },
                {"id": "p4", "vosita": "echo_technician", "tavsif": "11 ta echo ko‘rinishini tasniflash"},
                {"id": "p5", "vosita": "echo_segmenter", "tavsif": "Chap qorincha konturini maskalash"},
                {"id": "p6", "vosita": "cardiology_fellow", "tavsif": "Dastlabki multimodal xulosa"},
            ]
        }
        natija = llm_json(
            tizim=(
                "Siz CardiacRAG reja tuzuvchisisiz. Faqat JSON qaytaring: "
                '{"qadamlar":[{"id":"p1","vosita":"lab_technician|ecg_technician|'
                'electrophysiologist|echo_technician|echo_segmenter|cardiology_fellow",'
                '"tavsif":"..."}]}. Tashxis yozmang. Kerakli vositalarnigina tanlang.'
            ),
            foydalanuvchi=f"So‘rov:\n{sorov}\n\nKontekst C:\n{c_matn}",
            zaxira=zaxira,
        )
        qadamlar = natija.get("qadamlar") or zaxira["qadamlar"]
        toza: List[dict] = []
        ruxsat = {
            "lab_technician",
            "ecg_technician",
            "electrophysiologist",
            "echo_technician",
            "echo_segmenter",
            "cardiology_fellow",
        }
        for i, q in enumerate(qadamlar, start=1):
            if not isinstance(q, dict):
                continue
            vosita = str(q.get("vosita") or "")
            if vosita not in ruxsat:
                continue
            toza.append(
                {
                    "id": str(q.get("id") or f"p{i}"),
                    "vosita": vosita,
                    "tavsif": str(q.get("tavsif") or vosita),
                    "holat": "pending",
                }
            )
        return toza or [{**q, "holat": "pending"} for q in zaxira["qadamlar"]]
