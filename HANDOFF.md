# HANDOFF — cardio-ai-system

**Qoida:** yangi sessiyada butun loyihani va bu faylning eski tarixini qidirmang. Shu bo‘lim va «Maqola kamchiliklari» jadvalini o‘qing. Sessiya kundaligini qo‘shmang — faqat «Hozir qayerdamiz» va ochiq qatorni yangilang. Modul tafsiloti: `DASTURIY_YORIQNOMA.md`.

Tizim shifokor o‘rnini bosmaydi. Yakuniy qaror shifokorga tegishli. **Klinik production emas.**

---

## Hozir qayerdamiz (2026-10-13)

| Maydon | Qiymat |
|---|---|
| Keyingi ish | K8 yopildi. Ochiq: K2 (foydalanuvchi, kod emas), K5, K6, K7, K10. K3/K4 ni qayta so‘ramang |
| Oxirgi kod | Murakkab reja yuklangan lab/EKG/echo ni to‘ldiradi; qadamlar tugaguncha CONTINUE; murakkab bo‘lsa MDT |
| GitHub | https://github.com/Abbos8/cardio-ai-system (private, `main`) |
| UI | `./ishga_tushir.sh` → http://127.0.0.1:8501 |
| Oqim | murakkablik → CardiacRAG reja → vosita → CONTINUE/STOP → MDT → xulosa. Vizual panel tugma bilan |

Maqola (`arXiv:2508.13256`) kamchiliklarini koddan qayta qidirmang. Jadval manba. Yopilganda status `qilindi`, keyingi ochiq qatorga o‘ting.

---

## Maqola kamchiliklari

Aniqliq raqamlari (MIMIC-IV 0.87, PTB-XL 0.96, PTB 0.77) bu loyihada o‘lchanmagan.

| # | Kamchilik | Holat | Qayerda | Yopish uchun |
|---|---|---|---|---|
| K1 | Muhokara 2 raund (T=4) | **qilindi** (2026-10-12) | `src/agents/mdt.py` | — |
| K2 | MedGemma jonli javob bermayapti (CUDA) | ochiq — foydalanuvchi | `.env` `MEDGEMMA_*`; `https://stt-pipeline.rifqat.uz/llm` (`/v1` siz) | Serverni qayta ishga tushirish. Kod o‘zgarmaydi |
| K3 | Mayo Clinic sahifalari | to‘xtatilgan — fayl yo‘q | `data/raw/mayo/` bo‘sh | HTML paydo bo‘lsa shu yerga qo‘yib `indeks_qur.py --rebuild` |
| K4 | To‘liq yo‘riqnoma PDF + Docling | to‘xtatilgan | `data/raw/guidelines/` — 2 ESC slayd PDF; `pypdf` o‘qiydi | Jurnal PDF yo‘q (AHA/ACC 403). Docling zaxira |
| K5 | Echo 11 ko‘rinish EchoPrime emas | ochiq | `src/tools/echo_view_model.py` | Klinik tasnif og‘irligi. Hozir DICOM teg + geometrik model. A5C va Doppler yo‘q |
| K6 | LV segment Zhang 2024 emas | ochiq | `src/tools/echo_segmenter.py` | Og‘irlik yo‘q. Hozir OpenCV kavak. EF yo‘q |
| K7 | Fellow o‘qitilgan multimodal model emas | ochiq | `src/tools/fellow_tool.py` | TGMM og‘irligi yo‘q. Hozir Qwen/DeepSeek prompt |
| K8 | Muhokama | **qilindi** (2026-10-13) | `chief_agent._reja_modalitet_toldir`, `_stepwise_yangilash` | — |
| K9 | Vizual panel so‘ralganda | **qilindi** (2026-10-12) | `src/ui/app.py` | «Vizual tekshiruvni ochish». Yangi tahlil panelni yopadi |
| K10 | MIMIC-IV / PTB-XL baholash | ochiq | yo‘q | PhysioNet to‘plami. E’lon qilingan ACC/AUC bu tizimniki emas |

---

## Bilim bazasi

MedlinePlus 11 HTML, NHS 5, ESC 3 HTML, StatPearls 20 `.md`, ESC slayd PDF 2 (HF 2026, valvular). Mayo bo‘sh. Indeks `data/processed/rag_index/` — **900** bo‘lak (StatPearls 128, slayd 382), BioClinicalBERT, bo‘lak 500 belgi / 50 ustma.

```bash
PYTHONPATH=src python src/rag/indeks_qur.py --rebuild
```

Cheklov: Bookshelf HTML captcha — StatPearls da E-utilities xulosasi. Aortic regurgitation faqat Archived. Slayd PDF ko‘p bo‘lak beradi, ba’zi so‘rovda StatPearls dan oldin chiqadi. Xom matn gitda yo‘q.

## Hozirgi qoida

Maqola (arXiv:2508.13256): murakkab holatda reja yuklangan modalitetlar bo‘yicha yuradi va xulosa qadamlar tugagach chiqadi; MDT murakkab holat uchun. Kod: `_reja_modalitet_toldir`, reja tugaguncha CONTINUE, `murakkablik=murakkab` bo‘lsa MDT. Oddiy holatda MDT faqat vosita yiqilsa yoki EKG sifati yomon.

---

## Ishga tushirish

Uyda `git pull`, ishda o‘zgarish → shu fayl → commit → `git push`. Kalit faqat loyiha `.env` da; chatga va gitga yozilmaydi. Kalitni qayta so‘ramang.

```bash
cd cardio-ai-system
./ishga_tushir.sh
```

Conda `(base)` bilan `streamlit` ishlatmang (NumPy/FAISS aralashadi).

- Smoke: `PYTHONPATH=src python3 -c "from agents.chief_agent import ChiefCardiologist"`
- Test: `PYTHONPATH=src python -m unittest tests.test_barqarorlik tests.test_xavfsizlik tests.test_rag_korpus -v`
- 2026-10-12: DeepSeek-R1 va Qwen2.5-VL javob berdi. Fellow rasmi `FELLOW_VISION_*` yoki `QWEN_VL_*` (`qwen/qwen-2.5-vl-72b-instruct`). Reasoner ga rasm ketmaydi.

14 bosqich (1–14) **qilindi**. Kod xaritasi `DASTURIY_YORIQNOMA.md`.
