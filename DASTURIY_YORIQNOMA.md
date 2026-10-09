# Dasturiy yo‘riqnoma — cardio-ai-system

Bu hujjat dasturning **qanday ishlashini** tushuntiradi: papkalar, kutubxonalar, klasslar, funksiyalar va asosiy o‘zgaruvchilar. Klinik foydalanish emas. Tizim shifokor o‘rnini bosmaydi.

Ishga tushirish: loyiha ildizida `./ishga_tushir.sh` → Streamlit `http://127.0.0.1:8501`. Python yo‘li `PYTHONPATH=src`.

---

## 1. Bitta tahlil qanday oqadi

Foydalanuvchi formani to‘ldiradi. `src/ui/app.py` dagi `asosiy()` bemor lug‘atini yig‘adi va `ChiefCardiologist.run_oqim()` ni chaqiradi. Agent LangGraph grafida shu tartibda yuradi:

```
START
  → qabul_qilish
  → murakkablik_baholash
  → cardiac_rag_reja
  → qadam_bajarish → stepwise_yangilash
       CONTINUE → yana qadam_bajarish
       STOP + mdt_kerak → mdt → xulosa_tayyorlash
       STOP → xulosa_tayyorlash
  → END
```

Har tugun `ChiefHolat` lug‘atining bir qismini qaytaradi. LangGraph shu qismlarni umumiy holatga qo‘shib boradi. UI har yangilanishda 6 bosqich qatorini va reja qadamlarini chizadi.

---

## 2. Kutubxonalar

| Paket | Qayerda | Vazifasi |
|---|---|---|
| `streamlit` | `src/ui/app.py` | 3 ustunli sahifa, yuklovchilar, grafik |
| `langgraph` | `src/agents/chief_agent.py` | `StateGraph`: tugunlar va CONTINUE/STOP shartlari |
| `numpy` | EKG, echo, RAG | Signal va kadr massivlari. Talab: `>=1.26.4,<2` |
| `neurokit2` | `ecg_tool.py` | EKG tozalash, R/P/QRS/T nuqtalari |
| `scipy` | `ecg_tool.py` | `find_peaks` — neurokit ishlamasa R cho‘qqi zaxirasi |
| `matplotlib` | `ecg_tool.py`, `app.py` | EKG grafigi va fellow uchun PNG |
| `pydicom` | `echo_yuklash.py` | DICOM kadr va SeriesDescription/Protocol |
| `opencv` (`cv2`) | echo yuklash, segmenter | Video kadr, LV kavak, overlay PNG |
| `scikit-learn` | `echo_view_model.py` | 11 ko‘rinish uchun logistik regressiya (demo) |
| `pypdf` | `lab_tool.py` | Matnli PDF dan analit (skan-OCR yo‘q) |
| `faiss-cpu` | `medical_rag.py` | Bo‘lak vektorlari, ichki ko‘paytma (`IndexFlatIP`) |
| `torch`, `transformers` | `medical_rag.py` | BioClinicalBERT embedding. CUDA yo‘q bo‘lsa CPU |
| `beautifulsoup4` | `ingest.py` | Yuklangan HTML dan matn |
| `python-dotenv` | `llm/client.py`, `app.py` | Loyiha ildizidagi `.env` |
| `urllib` (standart) | `llm/client.py` | OpenAI-mos `POST /v1/chat/completions` |

`.env` gitga kirmaydi. Kalit bo‘lmasa agent to‘xtamaydi: qoida yoki shablon ishlatiladi.

---

## 3. Papkalar

```
cardio-ai-system/
  ishga_tushir.sh          # venv + PYTHONPATH=src + streamlit
  requirements.txt
  .env                     # sirlar; git emas
  .env.example
  HANDOFF.md               # qaysi bosqich ochiq
  DASTURIY_YORIQNOMA.md    # shu fayl
  src/agents/              # bosh agent va MDT
  src/llm/                 # DeepSeek / VLM HTTP
  src/rag/                 # CardiacRAG
  src/tools/               # lab, EKG, echo, fellow
  src/ui/app.py            # Streamlit
  src/xavfsizlik/          # audit va git/PII tekshiruv
  tests/
  data/raw/                # yuklangan HTML; git emas
  data/processed/rag_index/  # FAISS; git emas
  data/audit/              # tahlil JSON; git emas
```

`src/llm/__init__.py` va `src/xavfsizlik/__init__.py` bo‘sh paket belgilari. Ish mantig‘i yo‘q.

---

## 4. Umumiy ma’lumot shakllari

### Bemor lug‘ati

UI va agent shu kalitlarni kutadi. Hammasi shart emas.

| Kalit | Turi | Ma’nosi |
|---|---|---|
| `yosh`, `jins` | son, satr | Demografiya |
| `shikoyatlar`, `anamnez`, `dorilar` | satr | Klinik matn |
| `laboratoriya` | `dict` | Forma analatlari (`troponin_i`, `kaliy`, …) |
| `lab_fayl_bayt`, `lab_fayl_nomi` | bytes, satr | CSV/PDF/TXT |
| `ecg_signal` yoki `ekg` yoki `signal` | `ndarray` | EKG namunalari |
| `sampling_rate` | float | Hz, odatda 500 |
| `echo_fayllar` | ro‘yxat | `{nom, bayt}` juftliklari |
| `echo_bayt`, `echo_fayl`, `echo_path` | bytes yoki yo‘l | Bitta echo fayl |
| `klinik_savol` | satr | RAG so‘rovi uchun qisqa lab matni |

### `ChiefHolat` (`src/agents/chief_agent.py`)

`TypedDict`. Grafning yagona holati. Muhim maydonlar:

| Maydon | Ma’nosi |
|---|---|
| `bemor` | Normallashtirilgan kirish |
| `xom_i` | Bemorning qisqa matni (MDT dagi I) |
| `murakkablik` | `oddiy` yoki `murakkab` |
| `murakkablik_sababi` | Nega shu daraja |
| `rag_dalillar` | CardiacRAG dan 3 tagacha parcha (C) |
| `reja` | Qadamlar: `{id, vosita, tavsif, holat}` |
| `reja_indeks` | Keyingi bajariladigan qadam |
| `lab_natija`, `ecg_natija`, `ecg_technician_natija`, `ep_natija` | Vosita javoblari |
| `echo_natija`, `echo_mask` | Ko‘rinishlar va LV qisqasi (katta massiv grafda qolmaydi) |
| `fellow_natija`, `mdt_natija` | Fellow va munozara |
| `mdt_kerak` | STOP dan keyin MDT ga kirish |
| `oraliq_z` | Vositalar qisqasi (MDT dagi Z) |
| `amal` | `CONTINUE` yoki `STOP` |
| `qadam` | Graf qadami. Chegara `MAX_QADAM = 16` |
| `qadam_tarixi` | Satrlardan iborat jurnal |
| `vizual` | UI paneli: sifat, ko‘rinishlar, LV, MDT raund |
| `xulosa`, `audit` | Yakuniy matn va audit yozuvi |

`Amal` — `Literal["CONTINUE", "STOP"]`.

Chegara o‘zgaruvchilari (agent va EKG da takrorlanadi, tashxis emas): `QRS_KENG_MS=120`, `PR_UZOQ_MS=200`, `QTC_UZOQ_MS` / `QT_UZOQ_MS=460`, `HR_PAST=50`, `HR_YUQori=100`.

---

## 5. `src/agents/chief_agent.py`

### Klass `ChiefCardiologist`

| Atribut | Ma’nosi |
|---|---|
| `self.rag` | `MedicalRAG` yoki `None` |
| `self.max_qadam` | Siklik chegarasi, standart `MAX_QADAM` |
| `self.graf` | `compile()` qilingan LangGraph |

| Metod | Nima qiladi |
|---|---|
| `_graf_qur` | Tugun va qirralarni yig‘adi |
| `_qabul_qilish` | Bemorni tozalaydi, bo‘sh natijalarni qo‘yadi, `xom_i` yozadi |
| `_murakkablik_baholash` | `llm_json` dan `oddiy`/`murakkab`. API yo‘q bo‘lsa qoida: troponin ≥ 34, NT-proBNP ≥ 300, echo bor, yoki EKG + boshqa modalitet |
| `_cardiac_rag_reja` | Murakkab va RAG bor bo‘lsa `retrieve` + `reja_tuz`. Aks holda `_oddiy_reja` |
| `_qadam_bajarish` | `reja[reja_indeks].vosita` bo‘yicha bitta vosita |
| `_stepwise_yangilash` | `oraliq_z` ni yangilaydi, `llm_json` dan `CONTINUE`/`STOP` va `mdt` |
| `_mdt` | `mdt_munozara` ni chaqiradi |
| `_xulosa_tayyorlash` | Matn yig‘adi, `audit_yig` + `audit_saqla` |
| `_keyingi_stepwise` | CONTINUE → vosita; STOP+mdt → MDT; aks holda xulosa |
| `run` | `graf.invoke` — bitta yakuniy holat |
| `run_oqim` | `graf.stream` — UI uchun har tugundan keyin holat |

Yordamchi funksiya `chief_cardiologist_yarat(rag=None)` RAG ni urinib ko‘radi va agent qaytaradi.

### Vosita nomlari (`_qadam_bajarish`)

| `vosita` | Chaqiruv |
|---|---|
| `lab_technician` | `process_lab` |
| `ecg_technician` | `ecg_technician_tahlil` |
| `electrophysiologist` | `electrophysiologist_tahlil` |
| `echo_technician` | `classify_echo_views` |
| `echo_segmenter` | `segment_lv` |
| `cardiology_fellow` | `dastlabki_tashxis` |

Noma’lum nom jurnalga yoziladi, dastur yiqilmaydi. Katta echo/EKG massivlari holatga `_echo_qisqacha`, `_lv_qisqacha`, `_ekg_qisqacha` orqali qisqartirib qo‘yiladi.

`_oddiy_reja` bemorda nima borligiga qarab shu oltita qadamdan keraklisini qo‘shadi. Signal yo‘q bo‘lsa EKG/EP tushadi. Echo yo‘q bo‘lsa echo qadamlari tushadi. Fellow har doim oxirida.

---

## 6. `src/agents/mdt.py`

Bu reja vositasi emas. Graf tuguni. Ikki rol: MedGemma (still tasvir) va Qwen2.5-VL (video kadrlar).

| O‘zgaruvchi | Qiymat | Ma’nosi |
|---|---|---|
| `MAX_RAUND` | 3 | Munozara chegarasi |
| `MAX_VIDEO_KADR` | 6 | Qwen ga ketadigan kadrlar |
| `VLM_TIMEOUT` | 180 | Soniyada HTTP kutish |

| Funksiya | Vazifasi |
|---|---|
| `mdt_vizual_yig` | EKG PNG, echo still, LV overlay, video kadrlarni yig‘adi |
| `mdt_munozara` | Raundlar, konsensus, umumlashtirish |
| `_rol_javobi` | `vlm_sozlama(rol)` → rasmli `llm_chat`, bo‘lmasa matn, bo‘lmasa `_shablon` |
| `_konsensus_bormi` | Avval `llm_json`, bo‘lmasa `_konsensus_qoida` |
| `_data_url`, `_kadr_png`, `_kadr_tanla` | PNG ni base64 data-URL qiladi |

`manba` qiymatlari: `vlm`, `llm`, `shablon`. MedGemma hosti ochilmasa yoki javob bo‘lmasa shu rol `shablon` bo‘ladi. Bu fellow chaqiruvidan alohida.

---

## 7. `src/llm/client.py`

Barcha model chaqiruvlari shu fayldan o‘tadi. Og‘irlik diskka yuklanmaydi. `load_dotenv` loyiha `.env` ini o‘qiydi.

| Funksiya | Qaytaradi |
|---|---|
| `_sozlama` | `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`, `FELLOW_VISION_MODEL` |
| `llm_mavjud` | DeepSeek kaliti bor-yo‘qligi |
| `llm_chat` | `POST {baza}/v1/chat/completions`. Xatoda `None` |
| `llm_json` | Matndan `{...}` ajratadi. Bo‘lmasa `zaxira` lug‘at |
| `vlm_sozlama(rol)` | `medgemma` yoki `qwen_vl` uchun URL, kalit, model |
| `fellow_vision_sozlama` | Fellow rasmlari uchun URL, kalit, model |
| `llm_vision_model` | Faqat vision model nomi |
| `model_qisqacha` | UI/audit: `chat`, `fellow_vision`, `medgemma`, `qwen_vl` |

`.env` maydonlari:

| Maydon | Kim o‘qiydi |
|---|---|
| `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL` | Murakkablik, reja, stepwise, fellow **matn** |
| `FELLOW_VISION_MODEL`, `FELLOW_VISION_BASE_URL`, `FELLOW_VISION_API_KEY` | Fellow **rasm** |
| `QWEN_VL_BASE_URL`, `QWEN_VL_API_KEY`, `QWEN_VL_MODEL` | MDT Qwen. Fellow kaliti bo‘sh bo‘lsa vision ham shu kalitni oladi |
| `MEDGEMMA_BASE_URL`, `MEDGEMMA_API_KEY`, `MEDGEMMA_MODEL` | Faqat MDT MedGemma |
| `USE_BIOCLINICAL_BERT` | `1` bo‘lsa RAG BERT |
| `RAG_INDEX_DIR`, `AUDIT_YOZ`, `AUDIT_DIR` | Indeks va audit yo‘li |

`deepseek-reasoner` tasvir qabul qilmaydi. `fellow_vision_sozlama` model nomida `reasoner` bo‘lsa `None` qaytaradi. Fellow rasmni DeepSeek ga yubormaydi.

Fellow va MDT alohida. MedGemma kaliti fellow matnini o‘zgartirmaydi.

---

## 8. Vositalar — `src/tools/`

### `lab_tool.py` — Laboratory technician

Model yo‘q. Qoida va sinonim.

| O‘zgaruvchi | Ma’nosi |
|---|---|
| `LAB_IZOH` | Analit → `(birlik, qisqa izoh)` |
| `LAB_SINONIM` | Hisobot nomi → ichki kalit (`Potassium` → `kaliy`) |

| Funksiya | Vazifasi |
|---|---|
| `csv_dan_lab`, `matndan_lab`, `pdf_dan_lab`, `fayldan_lab` | Baytdan `{kalit: son}` |
| `dorilarni_ajrat` | `nom \| doza \| chastota` |
| `process_lab` | Forma + fayl + dori → `rag_satr`, `tokenlar`, `qiymatlar` |

`rag_satr` namunasi: `LAB kaliy=3.2 troponin_i=88.0 | MEDS aspirin|75 mg|once daily`. Fayl qiymati forma ustiga yoziladi.

### `ekg_yuklash.py`

| O‘zgaruvchi | Ma’nosi |
|---|---|
| `VAQT_SARLAVHA` | `time`, `sec` kabi ustunlar signal emas |

| Funksiya | Vazifasi |
|---|---|
| `csv_dan_ekg` | `# sampling_rate=500` va I..V6 yoki 12 ustun |
| `wfdb_dan_ekg` | `.hea` + `.dat` (format 16 yoki 212) |
| `ekg_fayllardan_oqish` | Qaysi format ekanini tanlaydi |

PNG/JPG qog‘oz EKG qabul qilinmaydi. Raqamli signal kerak.

### `ecg_tool.py` — ECG technician va Electrophysiologist

| O‘zgaruvchi | Ma’nosi |
|---|---|
| `TASMA_NOMLARI` | I, II, III, aVR, aVL, aVF, V1…V6 |
| `ASOSIY_TASMA_TARTIBI` | Interval uchun avval II, keyin I, V5, V2, aVF |
| `MIN_URINISH_SONIYASI` | 2.0 — bundan qisqa yozuv tahlil qilinmaydi |

| Funksiya | Vazifasi |
|---|---|
| `ecg_technician_tahlil` | Tozalash, sifat, R cho‘qqilar, HR |
| `electrophysiologist_tahlil` | P/QRS/T, PR, QRS, QT, QTc Bazett, SDNN, RMSSD |
| `process_ecg_signal` | Ikkalasini birga chaqiradi |
| `ekg_qisqa_png` | Bitta tasma grafigi (fellow/MDT uchun) |
| `namuna_12_tasma_ekg` | Sintetik signal |

`nk` (`neurokit2`) yoki `find_peaks` import bo‘lmasa chaqiruv tushunarli `ok=False` qaytaradi.

### `echo_yuklash.py`

| Funksiya | Kirish |
|---|---|
| `dicom_dan_kadrlar` | `.dcm` |
| `video_dan_kadrlar` | mp4/avi/mov/mkv — eng ko‘pi 8 kadr |
| `rasm_dan_kadr` | png/jpg |
| `echo_fayldan_kadrlar` | Kengaytma bo‘yicha tanlov |
| `echo_yollardan_oqish` | Bir nechta `(nom, bayt)` |
| `_kulrang` | Kadrni uint8 kulrang qiladi |
| `_dicom_meta` | `series_description`, `protocol_name`, `view_name` |

### `echo_view_model.py` va `echo_tool.py` — Echocardiography technician

| O‘zgaruvchi | Ma’nosi |
|---|---|
| `ECHO_KORINISHLAR` | 11 yorliq: A2C, A4C, A3C, PLAX, PSAX-AV/MV/PM/APEX, SUBCOSTAL-4C, SUBCOSTAL-IVC, SSN |
| `_TEG_KALIT` | DICOM matni → yorliq |
| `_MODEL_W`, `_MODEL_B` | Birinchi chaqiruvda hisoblangan regressiya og‘irligi |

| Funksiya | Vazifasi |
|---|---|
| `tegdan_korinish` | DICOM tegidan yorliq |
| `yasama_echo_kadr` | O‘qitish uchun sintetik kadr |
| `kadr_belgilari` | 9 ta geometrik belgi |
| `_modelni_oqit` | `LogisticRegression` yoki markazga masofa |
| `kadrlar_tasnif` | 6 tagacha kadr ehtimolini o‘rtacha qiladi |
| `classify_echo_views` | Avval teg, bo‘lmasa geometrik model |
| `echo_bormi` | Reja uchun: echo fayl bormi |

`manba`: `dicom_teg`, `geometrik_model`, `dicom_teg+geometrik_model`, `aniqlanmadi`. Klinik CNN emas.

### `echo_segmenter.py` — Echocardiography segmenter

| O‘zgaruvchi | Ma’nosi |
|---|---|
| `APICAL` | A4C, A2C, A3C — kavakni pastki-o‘ngga yaqin qidiradi |
| `SAX` | PSAX-* — dumaloq kavakni afzal ko‘radi |

| Funksiya | Vazifasi |
|---|---|
| `segment_lv` | Kadrdan taxminiy LV maska |
| `_kavak_maska` | Sektordagi qorong‘i bog‘langan komponent |
| `overlay_qur` | Cyan maska + sariq kontur |
| `_png_bayt` | Overlay ni PNG bytes |

Muvaffaqiyatda `model="opencv-kavak"`. U-Net yo‘q. EF hisoblanmaydi. Kadr yoki kavak yo‘q bo‘lsa `ok=False`, yolg‘on maska yo‘q.

### `fellow_tool.py` — Cardiology fellow

| Funksiya | Vazifasi |
|---|---|
| `dastlabki_tashxis` | Lab + EKG + echo + LV + RAG ni bitta xulosaga yig‘adi |
| `_dalil_matn` | Bor va yo‘q modalitetlar. Yo‘q narsa o‘ylab topilmaydi |
| `_rasmlarni_yig` | `ekg_qisqa_png` va echo/LV PNG |
| `_echo_kadr_png` | Overlay bo‘lsa u, bo‘lmasa birinchi kadr |

`manba` zanjiri:

1. Rasm bor va `fellow_vision_sozlama()` bor → `llm_multimodal` (`FELLOW_VISION_MODEL`, hozir Qwen).
2. Javob `None` → matn `llm_chat` DeepSeek ga (`llm`).
3. U ham `None` → `shablon` (dalillar ro‘yxati, model chaqiruvi emas).

`model` maydoni: multimodal da vision nomi, matnda chat nomi, aks holda `shablon`.

---

## 9. `src/rag/` — CardiacRAG

### `medical_rag.py` — klass `MedicalRAG`

| O‘zgaruvchi | Qiymat | Ma’nosi |
|---|---|---|
| `MODEL_NOMI` | `emilyalsentzer/Bio_ClinicalBERT` | Embedding modeli |
| `EMBEDDING_OLCHAMI` | 768 | FAISS vektor uzunligi |
| `N_KONTEKST` | 3 | Odatdagi topilma soni |
| `BOSQICH1_SONI` | 9 | Birinchi navbatdagi nomzodlar |
| `BOSQICH2_SONI` | 3 | Qayta tartibdan keyin qoladiganlar |
| `POZITSION_BONUS` | 1.2 | Parcha boshidagi kalit uchun |
| `BOSH_ULUSH` | 0.30 | «Bosh» hisoblangan ulush |
| `MW_KOEF` | 1.5 | Tibbiy lug‘at mosligi koeffitsienti |
| `TIBBIY_LUGAT` | dict | AF, ACS, troponin kabi sinonimlar |
| `MAX_TOKEN` | 256 | BERT ga beriladigan uzunlik |
| `INDEKS_FAYL`, `BOLAK_FAYL`, `META_FAYL` | `index.faiss`, `chunks.json`, `meta.json` | Diskdagi indeks |

| Metod | Vazifasi |
|---|---|
| `_embedding_yarat` | BERT o‘rtacha pul, bo‘lmasa hashing vektor |
| `index_documents` | Bo‘laklarni FAISS ga qo‘shadi |
| `indeksni_saqla` / `indeksni_yukla` | Disk kesh. Imzo seed va `data/raw` hashiga bog‘liq |
| `retrieve` | Gibrid ball: vektor + TF-IDF + tibbiy so‘z + pozitsiya |
| `urug_indeks` | Seed va raw dan indeks. O‘zgarmagan bo‘lsa keshdan |
| `reja_tuz` | Dalillarga qarab vosita qadamlari |

`_bert_yoqilgan()` `USE_BIOCLINICAL_BERT=1` ni tekshiradi.

### Boshqa RAG fayllari

| Fayl | Funksiya | Vazifasi |
|---|---|---|
| `ingest.py` | `html_dan_matn`, `pdf_dan_matn`, `chunklarga_ajrat`, `korpus_bolaklari` | Matnni ~180 so‘zlik, 40 so‘z ustma-ust parchaga ajratadi. Prefiks `[jild/fayl]` |
| `yuklab_ol.py` | `barcha_sahifalarni_yukla` | `manbalar.json` dagi sahifalarni `data/raw/` ga yozadi |
| `indeks_qur.py` | `asosiy` | `PYTHONPATH=src python src/rag/indeks_qur.py --qayta` |

O‘zgaruvchilar: `CHUNK_SOZ=180`, `CHUNK_USTMA_UST=40`, `RAW_DIR`, `MANBALAR_JSON`.

---

## 10. `src/ui/app.py`

Klass yo‘q. `asosiy()` sahifani yig‘adi.

| O‘zgaruvchi | Ma’nosi |
|---|---|
| `ODATIY_HZ` | 500 |
| `NAMUNA_SONIYA` | 8 — sintetik EKG uzunligi |
| `LAB_MAYDONLARI` | Formadagi 8 analit: nom, izoh, min, max, odatiy |

| Funksiya | Vazifasi |
|---|---|
| `_conda_numpy_aralashmasin` | conda `(base)` NumPy aralashsa darhol xato |
| `_ustun1_forma` | Yosh, shikoyat, echo/lab yuklash, «Tahlil qilish» |
| `_ustun2_ekg` | Tozalangan 12 tasma |
| `_ustun3_xulosa` | Xulosa, fellow, echo, LV, audit |
| `_olti_bosqich_chiz` | Murakkablik → RAG → vosita → stepwise → MDT → xulosa |
| `_reja_jonli` | `reja` dagi `pending`/`bajarildi` |
| `_vizual_tekshirish_paneli` | EKG, 11 echo slot, LV, MDT yonma-yon |
| `_agent_ol` | Keshlangan `ChiefCardiologist` |
| `_rag_ol` | Keshlangan `MedicalRAG` |

Fellow expanderidagi ogohlantirish shu shart bilan chiqadi: fellow `manba=="shablon"` **yoki** MDT MedGemma `manba=="shablon"`. Qwen ishlasa ham MedGemma yiqilsa shu yozuv ko‘rinadi.

---

## 11. `src/xavfsizlik/`

### `audit.py`

| O‘zgaruvchi | Ma’nosi |
|---|---|
| `KLINIK_OGOHLANTIRISH` | Har xulosa oxiridagi ogohlantirish |
| `KLINIK_PRODUCTION` | `False` — demo |
| `ODATIY_AUDIT_DIR` | `data/audit` |

| Funksiya | Vazifasi |
|---|---|
| `audit_yig` | Model nomlari, C manba id, fellow/MDT `manba`. Signal, ism, kalit yo‘q |
| `c_manbalar` | `[medlineplus/...]` dan id |
| `audit_saqla` | `AUDIT_YOZ=0` bo‘lsa yozmaydi |
| `audit_yozilsinmi` | Muhitdan yozish-yozmaslik |

### `himoya.py`

| O‘zgaruvchi | Ma’nosi |
|---|---|
| `TAQIQLANGAN_GIT` | `.env`, `data/raw`, `data/processed`, `data/audit`, `data/uploads` |
| `TAQIQLANGAN_PREFIKS` | Kalit naqshlari |

| Funksiya | Vazifasi |
|---|---|
| `gitda_taqiqlanganlar` | Git indeksida sirli yo‘l bormi |
| `pii_izlari`, `kalit_maydonlarini_ol`, `matndan_sirni_kes` | Matn va lug‘atdan sir izlash |
| `git_fayllarini_sirga_tekshir` | Track qilingan fayllarda kalit naqshi |

---

## 12. Testlar

`tests/test_barqarorlik.py` kalitlarni o‘chirib shablon yo‘lini tekshiradi: lab, EKG, echo, bo‘sh bemor, LangGraph chegarasi, hashing RAG, MDT shablon.

`tests/test_xavfsizlik.py` gitda `.env` yo‘qligini, auditda signal/kalit yo‘qligini va `KLINIK_PRODUCTION is False` ni tekshiradi.

```bash
source venv/bin/activate
PYTHONPATH=src python -m unittest tests.test_barqarorlik tests.test_xavfsizlik -v
```

---

## 13. Nima model emas

| Qism | Haqiqiy mexanizm |
|---|---|
| Lab | Sinonim lug‘ati va regex |
| EKG | neurokit2 / scipy |
| Echo ko‘rinish | DICOM teg + sintetik kadrda o‘qitilgan logistik regressiya |
| LV | OpenCV qorong‘i kavak |
| Fellow matn | DeepSeek HTTP |
| Fellow rasm | `FELLOW_VISION_MODEL` HTTP (Qwen), MedGemma emas |
| MDT | MedGemma va Qwen HTTP. Ulanmasa shablon matn |
| RAG | BioClinicalBERT embedding + FAISS, generativ model emas |
