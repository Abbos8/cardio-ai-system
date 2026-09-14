"""
# HANDOFF — cardio-ai-system

Ish va uy kompyuterida davom ettirish uchun. Har muhim o‘zgarishdan keyin shu fayl yangilanadi.

## Git / GitHub

- Remote: push qilingandan keyin `origin` (`Abbos8/cardio-ai-system`, private).
- Asosiy tarmoq: `main`.
- Uyda: `git clone https://github.com/Abbos8/cardio-ai-system.git` so‘ng `git pull`.
- Ishda: o‘zgarish → `HANDOFF.md` ni yangilang → commit → `git push`.

```bash
cd cardio-ai-system
python3 -m venv venv
source venv/bin/activate   # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env       # DEEPSEEK_API_KEY ixtiyoriy
streamlit run src/ui/app.py
```

## Bugun qilingan (2026-09-14)

### Infratuzilma

- Git init, `.gitignore`, `.env.example`.
- GitHub ga bog‘lash (Abbos8, `gh` auth bor).

### 6 bosqichli workflow (kodda)

1. **Qabul + murakkablik** — `ChiefCardiologist._murakkablik_baholash`: DeepSeek JSON yoki qoida (troponin, NT-proBNP, echo, ko‘p modalitet). Murakkab → CardiacRAG.
2. **CardiacRAG** — `MedicalRAG`: FAISS 3n, TF-IDF + MW + PB=1.2 (dastlabki 30%). `reja_tuz` → P qadamlar. Seed: `src/rag/knowledge/cardiology_seed.txt`. Ingest: BeautifulSoup/Docling ixtiyoriy (`src/rag/ingest.py`). BERT yuklanmasa hashing embedding.
3. **Vositalar** — lab (`lab_tool.py`), EKG tozalash + EP (QRS/PR/QT/HRV NeuroKit2), echo tasnif (11 ko‘rinish, fayl-nomi heuristic), LV segmenter interfeysi (model yo‘q — yolg‘on maska yo‘q), fellow.
4. **Stepwise** — har vositadan keyin S / A=CONTINUE|STOP / ixtiyoriy P_{s+1}.
5. **MDT** — MedGemma va Qwen2.5-VL rollari, har raundda I va Z. VLM yo‘q bo‘lsa LLM/shablon.
6. **Xulosa + vizual panel** — Streamlit 3 ustun + reja, C, MDT, EKG o‘lchovlari.

### Hali to‘liq emas

- Haqiqiy DeepSeek-R1-Distill-Qwen-32B mahalliy inferens (API kalit orqali ulanadi).
- BioClinicalBERT og‘irliklari (birinchi muvaffaqiyatli HF yuklash).
- Mayo/NHS/MedlinePlus/ESC to‘liq crawl; hozir seed + `data/raw/` ingest.
- DICOM 11-view klassifikator og‘irliklari.
- LV piksel segmentatsiya og‘irliklari.
- MedGemma / Qwen2.5-VL GPU inferens.

## Keyingi sessiyada

1. `.env` ga `DEEPSEEK_API_KEY` (yoki mahalliy vLLM).
2. `data/raw/` ga ko‘rsatma HTML/PDF qo‘yib indeksni kengaytirish.
3. Echo/LV modellarni `echo_tool` / `echo_segmenter` ga ulash.
4. MDT uchun haqiqiy VLM.

## Muhim fayllar

- `src/agents/chief_agent.py` — LangGraph workflow
- `src/agents/mdt.py`
- `src/rag/medical_rag.py`, `src/rag/ingest.py`
- `src/tools/*`
- `src/llm/client.py`
- `src/ui/app.py`
- `.cursor/rules/tibbiy_qoidalar.mdc`

## Tibbiy eslatma

Tizim shifokor o‘rnini bosmaydi. Xulosa tekshirilgan, ehtiyotkor yordam.
"""
