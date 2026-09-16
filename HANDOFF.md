# HANDOFF — cardio-ai-system

**Qoida:** yangi sessiyada butun loyihani qidirmang. Shu faylni o‘qing, «Hozir qayerdamiz» va 14 bosqich statusini qarang, keyingi **ochiq** bosqichni bajaring, oxirida shu fayldagi statusni yangilang, commit/push.

Tizim shifokor o‘rnini bosmaydi. Yakuniy qaror shifokorga tegishli.

---

## Hozir qayerdamiz (2026-09-16)

| Maydon | Qiymat |
|---|---|
| Keyingi ish | **8-bosqich** — Echo segmenter (LV maska); 2-bosqich DeepSeek ham ochiq |
| Oxirgi yopilgan | **7-bosqich** — Echo technician (DICOM/video, 11 ko‘rinish) |
| GitHub | https://github.com/Abbos8/cardio-ai-system (private, `main`) |
| UI | `./ishga_tushir.sh` → http://127.0.0.1:8501 |

Bosqichni yopganda: pastdagi jadvalda statusni `qilindi` qiling, «Hozir qayerdamiz» ni yangilang, qisqa «Sessiya yozuvi» qo‘shing.

---

## Git / ishga tushirish

- Remote: https://github.com/Abbos8/cardio-ai-system
- Uyda: `git clone` / `git pull`. Ishda: o‘zgarish → shu fayl → commit → `git push`.

```bash
cd cardio-ai-system
./ishga_tushir.sh
# yoki: conda deactivate && source venv/bin/activate && python -m streamlit run src/ui/app.py
```

**Ishlatmang:** conda `(base)` ochiqcha `streamlit run ...` — miniconda Streamlit/FAISS + NumPy 2.2.6 aralashadi (`numpy.core.multiarray failed to import`). To‘xtating (Ctrl+C) va `./ishga_tushir.sh`.

Smoke-test: `PYTHONPATH=src python3 -c "from agents.chief_agent import ChiefCardiologist"`

Kalitlarni faqat `.env` ga yozing; chatga va gitga tushirmang.

---

## 14 bosqich — ketma-ket reja

Har bosqich: **nima**, **qayerda**, **tayyor deb hisoblash**, **status**.

| # | Bosqich | Status |
|---|---|---|
| 1 | Ishga tushirish muhiti | **qilindi** (ish kompyuter) |
| 2 | DeepSeek-R1 ulash | ochiq — `.env` bor, API/32B tasdiqlanmagan |
| 3 | BioClinicalBERT + FAISS | **qilindi** |
| 4 | Bilimlar bazasi (Mayo/NHS/MedlinePlus/ESC) | **qilindi** (Mayo 403 — qo‘lda) |
| 5 | Laboratoriya vositasini chuqurlashtirish | **qilindi** |
| 6 | EKG texnik + EP sifati | **qilindi** |
| 7 | Echo technician (11 ko‘rinish, DICOM) | **qilindi** |
| 8 | Echo segmenter (LV maska) | ochiq |
| 9 | Cardiology fellow (multimodal) | ochiq |
| 10 | MDT: MedGemma + Qwen2.5-VL | ochiq |
| 11 | Vizual tekshirish paneli | ochiq |
| 12 | UI ni to‘liq oqimga bog‘lash | ochiq |
| 13 | Test va barqarorlik | ochiq |
| 14 | Xavfsizlik / klinik tayyorgarlik | ochiq |

---

### 1. Ishga tushirish muhiti — qilindi

**Nima:** venv, paketlar, `.env`, UI ochilishi.

**Qayerda:** `requirements.txt`, `.env.example`, `venv/`, `ishga_tushir.sh`.

**Tayyor:** `pip install` o‘tgan; NeuroKit2/FAISS/Streamlit/torch import; streamlit 8501.

**Qayd:** ish PC — torch CPU, CUDA yo‘q. UI faqat `./ishga_tushir.sh` (conda `streamlit` ni ishlatmang). Uyda venv + conda aralashmasin.

---

### 2. DeepSeek-R1 ni ulash — ochiq

**Nima:** bosh kardiolog murakkablik, reja `P`, stepwise `S/A`, fellow/xulosa uchun R1 (yoki Distill-Qwen-32B). Kalitsiz — qoida/shablon.

**Qayerda:** `.env` (`DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`); `src/llm/client.py`.

**Qilish:** kalit yoki mahalliy vLLM; bitta tahlil qilib `manba=llm` va murakkablik JSON kelishini ko‘rish.

**Tayyor:** API/lokal javob bor; kalitsiz ham agent to‘xtamaydi (zaxira saqlansin).

---

### 3. BioClinicalBERT + FAISS indeksi — qilindi

**Nima:** vektor qidiruv; indeksni diskka saqlash.

**Qayerda:** `.env` `USE_BIOCLINICAL_BERT=1`; `src/rag/medical_rag.py`; indeks `data/processed/rag_index/` (`index.faiss`, `chunks.json`, `meta.json`). Model: `emilyalsentzer/Bio_ClinicalBERT` (`_BERT_KESH`, HF `~/.cache/huggingface`).

**Tayyor:** birinchi yuklash ~400 MB internet; keyin disk. Seed SHA o‘zgarmasa UI qayta embedding qilmaydi. BERT yo‘q bo‘lsa hashing zaxira.

---

### 4. Bilimlar bazasini to‘ldirish — qilindi

**Nima:** CardiacRAG manbalari. Seed + rasmiy sahifalar.

**Qayerda:**
- URL ro‘yxati: `src/rag/manbalar.json` (gitda; xom HTML yo‘q)
- Yuklash: `src/rag/yuklab_ol.py` → `data/raw/{medlineplus,nhs,esc,mayo}/` (gitignore)
- Ingest: `src/rag/ingest.py` — overlapping chunk, manba prefiksi `[jild/fayl]`
- Qurish: `PYTHONPATH=src python src/rag/indeks_qur.py --qayta`
- Indeks: `data/processed/rag_index/` ; kesh kaliti seed **va** `data/raw` o‘lcham/hash

**Natija (ish PC):** 19 HTML (MedlinePlus 11, NHS 5, ESC 3). **166 chunk**, BERT=True. Retrieve: AF so‘rovi MedlinePlus AFib; ACS da seed + EKG encyclopedia.

**Cheklov:**
- Mayo Clinic avto-yuklash **403**. HTML ni brauzerdan saqlab `data/raw/mayo/` ga qo‘ying, so‘ng `indeks_qur.py --qayta`.
- ESC sahifalarida ko‘p huquqiy/nav matn; to‘liq PDF ko‘rsatma yo‘q (mualliflik).
- Docling qo‘shilmadi — PDF hali yo‘q.

**Qayta qurish:** `conda deactivate && source venv/bin/activate && PYTHONPATH=src python src/rag/indeks_qur.py --qayta`

---

### 5. Laboratoriya vositasini chuqurlashtirish — qilindi

**Nima:** CSV/PDF/forma + tuzilgan dori tarixi → barqaror `rag_satr` / tokenlar.

**Qayerda:** `src/tools/lab_tool.py`; namuna `src/tools/namuna_lab.csv`; UI lab yuklovchi; `pypdf` (`requirements.txt`).

**Format:** `LAB kaliy=3.2 troponin_i=88.0 | MEDS aspirin|75 mg|once daily`. Fayl forma qiymatini ustiga yozadi. Dorilar: `nom | doza | chastota`.

**Tekshiruv:** namuna CSV → 8 analit (sinonims: Troponin I, Potassium); CSV 88 ustun forma 8; tokenlarda aspirin.

**Cheklov:** skan-PDF OCR yo‘q; tashxis qo‘yilmaydi.

---

### 6. EKG ni texnik + EP darajasiga yetkazish — qilindi

**Nima:** technician tozalash/sifat; EP da P/QRS/T, PR/QRS/QT/QTc, HRV. CSV va WFDB (.hea+.dat format 16/212).

**Qayerda:** `src/tools/ecg_tool.py`, `src/tools/ekg_yuklash.py`; namuna `src/tools/namuna_ekg.csv` (+ `.hea`/`.dat`); UI 2-ustun va expanderlar.

**Format:** CSV `# sampling_rate=500`, `time` + I..V6 (yoki 12 ustun). WFDB: ikkala faylni birga yuklash.

**Tekshiruv:** namuna 8 s / 500 Hz → technician 12 tasma, 9 R, HR 72; EP P/T=9, PR/QRS/QT/QTc, UI da technician vs EP.

**Cheklov:** sintetik yozuvda QRS/QT DWT noaniq bo‘lishi mumkin (zaxira Q–S); tashxis qo‘yilmaydi.

---

### 7. Echo technician (11 ko‘rinish, DICOM) — qilindi

**Nima:** DICOM/video/rasm kadr + 11 yorliq (A2C, A4C, A3C, PLAX, PSAX-*, subcostal, SSN). Fayl-nomi heuristic asosiy emas.

**Qayerda:** `src/tools/echo_tool.py`, `echo_yuklash.py`, `echo_view_model.py`; UI yuklovchi; namuna `src/tools/namuna_echo_a4c.dcm`, `namuna_echo_plax.dcm`. Paketlar: `pydicom`, `opencv-python-headless`, `scikit-learn`.

**Tasnif:** avval DICOM SeriesDescription/Protocol; bo‘lmasa geometrik model (sintetik shablonlarda o‘qitilgan). `manba=dicom_teg|geometrik_model`.

**Tekshiruv:** namuna A4C DICOM → A4C (`dicom_teg+geometrik_model`); A4C+PLAX → ikkala yorliq; agent rejasida `echo_technician`.

**Cheklov:** geometrik model demo (klinik tarmoq emas); video tegsizda xato yorliq bo‘lishi mumkin. Tashxis emas.

---

### 8. Echo segmenter (LV maska) — ochiq

**Nima:** interfeys bor, maska yaratilmaydi (`src/tools/echo_segmenter.py`).

**Qilish:** LV kontur og‘irliklari; piksel maska + overlay. Yolg‘on maska berilmasin.

**Tayyor:** kadrlar bo‘lsa `ok=True` va ko‘rinadigan maska; yo‘q bo‘lsa aniq xabar.

---

### 9. Cardiology fellow ni multimodal qilish — ochiq

**Nima:** matn yig‘indisi + ixtiyoriy LLM (`src/tools/fellow_tool.py`).

**Qilish:** EKG grafik + echo kadr/maska + lab tokenlarini bir multimodal chaqiriqqa berish.

**Tayyor:** xulosa faqat mavjud dalillardan; `manba` llm/shablon aniq.

---

### 10. MDT: MedGemma + Qwen2.5-VL — ochiq

**Nima:** ikki rol, VLM yo‘qida shablon (`src/agents/mdt.py`).

**Qilish:** MedGemma (tasvir), Qwen2.5-VL (video); har raundda `I` va `Z`; konsensus yoki max raund; bosh umumlashtiradi.

**Tayyor:** GPU/API da ikki model javobi; gallyutsinatsiya uchun I/Z qayta kiritiladi.

---

### 11. Vizual tekshirish paneli — ochiq

**Nima:** asosan xom EKG + raqamlar (`src/ui/app.py` expander).

**Qilish:** tozalangan 12 tasma; echo 11 ko‘rinish; LV overlay; MDT raundlari yonma-yon.

**Tayyor:** shifokor oraliq vizual natijani panelda ko‘radi.

---

### 12. UI ni to‘liq oqimga bog‘lash — ochiq

**Qilish:** RAG/BERT spinner va xato; echo drag-and-drop; reja qadamlari jonli; API yo‘qida «shablon rejimida» yozuvi.

**Tayyor:** 6 bosqichli workflow UI dan boshidan-oxirigacha ko‘rinadi.

---

### 13. Test va barqarorlik — ochiq

**Qilish:** lab-only, EKG-only, echo-only, hammasi bor/yo‘q; LangGraph limiti; GPU yo‘q zaxira (hashing/shablon).

**Tayyor:** asosiy yo‘llar buzilmasdan `STOP` + ehtiyotkor xulosa.

---

### 14. Xavfsizlik va klinik foydalanishga tayyorlash — ochiq

**Qilish:** PII gitda yo‘q; audit (qaysi model, qaysi C); ogohlantirish har ekranda. Keyin alohida: auth, log, klinik validatsiya.

**Tayyor:** demo xavfsiz; klinik production emas — shu yozuv saqlansin.

---

## Kodda allaqachon ishlaydigan 6 bosqichli oqim

Graf: qabul → murakkablik → CardiacRAG reja P → vosita → stepwise CONTINUE/STOP → MDT → xulosa.

Fayllar: `src/agents/chief_agent.py`, `src/agents/mdt.py`, `src/rag/medical_rag.py`, `src/rag/ingest.py`, `src/tools/*`, `src/llm/client.py`, `src/ui/app.py`.

---

## Sessiya yozuvlari

### 2026-09-14 (ish)

- GitHub, LangGraph workflow, HANDOFF.
- 1: venv, paketlar, `.env` nusxa, UI 8501, torch CPU.
- 3: `USE_BIOCLINICAL_BERT=1`, indeks `data/processed/rag_index/`.
- 14-bosqichli reja shu faylga to‘liq kiritildi. Keyingi sessiya: **4**.

### 2026-09-14 (kech, 4-bosqich)

- `manbalar.json` + `yuklab_ol.py` + `indeks_qur.py`.
- Yuklandi: MedlinePlus, NHS, ESC HTML → `data/raw/` (git emas). Mayo 403.
- 166 chunk FAISS (BERT). `urug_indeks` endi seed+raw.
- Keyingi: **5** (lab) yoki **2** (DeepSeek).

### 2026-09-14 (5-bosqich)

- Lab CSV/PDF parser, sinonimlar, fayl forma ustidan.
- Dorilar `nom|doza|chastota`; `rag_satr` RAG va fellow ga.
- UI: lab yuklovchi, dori text_area, expander.
- Namuna: `src/tools/namuna_lab.csv`. Keyingi: **6**.

### 2026-09-16 (6-bosqich)

- Technician vs EP: tozalash/sifat va P/QRS/T + PR/QRS/QT/QTc/HRV alohida.
- CSV (sarlavha, Hz izohi, time) va WFDB 16/212 o‘qish.
- UI: tozalangan 12 tasma, P/R/T belgilar, ikki expander.
### 2026-09-16 (7-bosqich)

- Echo DICOM/video/rasm o‘qish; 11 ko‘rinish (teg + geometrik model).
- UI: fayl yuklash, kadr + yorliq, expander.
- Namuna: `namuna_echo_a4c.dcm`, `namuna_echo_plax.dcm`. Keyingi: **8** (LV segmenter) yoki **2** (DeepSeek).
