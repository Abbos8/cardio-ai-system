# HANDOFF — cardio-ai-system

Ish va uy kompyuterida davom ettirish uchun. Har muhim o‘zgarishdan keyin shu fayl yangilanadi.

## Git / GitHub

- Remote: https://github.com/Abbos8/cardio-ai-system (private)
- Tarmoq: `main`
- Uyda: `git clone https://github.com/Abbos8/cardio-ai-system.git` so‘ng `git pull`
- Ishda: o‘zgarish → `HANDOFF.md` ni yangilang → commit → `git push`

```bash
cd cardio-ai-system
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # DEEPSEEK_API_KEY ixtiyoriy
streamlit run src/ui/app.py
```

Smoke-test (venv ichida): `PYTHONPATH=src python3 -c "from agents.chief_agent import ChiefCardiologist"`

## Bugun qilingan (2026-09-14)

### Infratuzilma

- Git init (`main`), `.gitignore`, `.env.example`.
- GitHub: https://github.com/Abbos8/cardio-ai-system
- Cursor qoidasi: `.cursor/rules/handoff.mdc`

### 6 bosqichli workflow (kodda ishlaydi)

1. **Qabul + murakkablik** — `ChiefCardiologist._murakkablik_baholash`: DeepSeek JSON yoki qoida (troponin, NT-proBNP, echo, ko‘p modalitet). Murakkab → CardiacRAG.
2. **CardiacRAG** — `MedicalRAG`: FAISS 3n, TF-IDF + MW + PB=1.2 (dastlabki 30%). `reja_tuz` → P. Seed: `src/rag/knowledge/cardiology_seed.txt`. Ingest: BeautifulSoup/Docling ixtiyoriy. `USE_BIOCLINICAL_BERT=1` bo‘lmasa hashing embedding.
3. **Vositalar** — lab, EKG tozalash + EP (QRS/PR/QT/HRV), echo 11 ko‘rinish (heuristic), LV segmenter interfeysi (og‘irlik yo‘q), fellow.
4. **Stepwise** — har vositadan keyin S / CONTINUE|STOP / ixtiyoriy P_{s+1}.
5. **MDT** — MedGemma va Qwen2.5-VL rollari, har raundda I va Z. VLM yo‘q bo‘lsa shablon.
6. **Xulosa + vizual panel** — Streamlit: reja, C, MDT, EKG o‘lchovlari.

### Hali to‘liq emas

- DeepSeek-R1-32B mahalliy inferens (API `.env` orqali).
- BioClinicalBERT HF yuklash (`USE_BIOCLINICAL_BERT=1`).
- Mayo/NHS/MedlinePlus/ESC to‘liq crawl (`data/raw/`).
- DICOM klassifikator va LV segmentatsiya og‘irliklari.
- Haqiqiy MedGemma / Qwen2.5-VL.

## Keyingi sessiyada

1. `.env` ga `DEEPSEEK_API_KEY` yoki mahalliy vLLM.
2. Ko‘rsatma HTML/PDF ni `data/raw/` ga qo‘yish.
3. Echo/LV modellarni ulash.
4. MDT uchun VLM.

## Muhim fayllar

- `src/agents/chief_agent.py`
- `src/agents/mdt.py`
- `src/rag/medical_rag.py`, `src/rag/ingest.py`
- `src/tools/*`
- `src/llm/client.py`
- `src/ui/app.py`

## Tibbiy eslatma

Tizim shifokor o‘rnini bosmaydi. Yakuniy qaror shifokorga tegishli.
