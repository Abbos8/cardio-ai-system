"""Kardiologik AI agent uchun gibrid RAG: FAISS + BioClinicalBERT + TF-IDF.

1-bosqich: vektor o‘xshashligi bo‘yicha 3n ta yaqin tibbiy bo‘lak.
2-bosqich: TF-IDF, tibbiy lug‘at og‘irligi (MW) va dastlabki 30% uchun PB=1.2.
Natija shifokor tashxisining o‘rnini bosmaydi.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import List, Optional, Sequence

import faiss  # Eng yaqin vektorlarni tez qidirish uchun
import numpy as np  # Embeddinglarni massiv sifatida saqlash uchun
import torch  # BioClinicalBERT ni GPU/CPU da ishlatish uchun
from transformers import AutoModel, AutoTokenizer  # BERT model va tokenizer

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


class MedicalRAG:
    """FAISS va BioClinicalBERT asosidagi ikki bosqichli tibbiy RAG."""

    def __init__(self, model_nomi: str = MODEL_NOMI) -> None:
        """BioClinicalBERT va bo‘sh FAISS indeksini yuklaydi.

        Args:
            model_nomi: Hugging Face dagi klinik BERT identifikatori.

        Returns:
            None. Indeks hali bo‘sh; avval index_documents chaqiriladi.
        """
        self.model_nomi = model_nomi
        self.qurilma = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        bert_yoq = os.getenv("USE_BIOCLINICAL_BERT", "0").strip() in {"1", "true", "True", "yes"}
        if bert_yoq:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_nomi, local_files_only=False)
                self.model = AutoModel.from_pretrained(model_nomi)
                self.model.to(self.qurilma)
                self.model.eval()
            except Exception:
                self.tokenizer = None
                self.model = None
        self.indeks = faiss.IndexFlatIP(EMBEDDING_OLCHAMI)
        self.bolaklar: List[str] = []  # Indeksdagi asl tibbiy matn bo‘laklari
        self._tf_idf_tayyor: bool = False
        self._idf: dict = {}
        self._df: dict = {}

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

    def urug_indeks(self, yol: Optional[Path] = None) -> int:
        """Ichki kardiologiya seed matnini indekslaydi.

        Args:
            yol: Ixtiyoriy .txt; None — knowledge/cardiology_seed.txt.

        Returns:
            Qo‘shilgan bo‘laklar soni.
        """
        from rag.ingest import chunklarga_ajrat

        if yol is None:
            yol = Path(__file__).resolve().parent / "knowledge" / "cardiology_seed.txt"
        if not yol.exists():
            return 0
        matn = yol.read_text(encoding="utf-8")
        return self.index_documents(chunklarga_ajrat(matn))

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
