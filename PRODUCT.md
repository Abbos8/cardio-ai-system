# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Mavjud ilova: Python, Streamlit (`src/ui/app.py`), brauzerda `http://127.0.0.1:8501`. Kiritish olti qadam, natija alohida ekran. HTML prototip `design/workflow/index.html` dizayn manbasi.

## Users

Asosiy foydalanuvchi — kardiolog, ish joyidagi stantsiyada. U bitta holatni yig‘adi va tahlilni ko‘rib chiqadi. Tizim shifokor o‘rnini bosmaydi; yakuniy qaror shifokorga tegishli.

## Product Purpose

Kardiologik AI agent bitta bemor holatini yig‘ib, klinik qaror qo‘llab-quvvatlash uchun ehtiyotkor xulosa beradi. Muvaffaqiyat: kardiolog kerakli ma’lumotni bosqichma-bosqich kiritadi, nima yetishayotganini ko‘radi, tahlildan keyin xulosani alohida ekranda o‘qiydi. Tashxis emas. Klinik production emas.

## Positioning

Holat bitta uzun formada emas, olti qadamda yig‘iladi, so‘ng alohida natija ekraniga o‘tadi. Tahlil oqimi qo‘shni oddiy forma-CDSS da yo‘q: murakkablik → CardiacRAG reja → ekspert vositalar → CONTINUE/STOP → MDT → xulosa. Vizual tekshiruv shifokor so‘raganda ochiladi.

## Operating Context

Ish stantsiyasi, keng brauzer oynasi. Kiritish va natija bir oynada aralashmasin. Qadamlar, tasdiqlangan tartibda: holatni boshlash → bemor → laboratoriya → EKG → echo → tekshiruv va tahlil, keyin alohida natija. Sintetik holatlar `data/sintetik` dan (sodda, murakkab, juda murakkab) solishtirish uchun; haqiqiy yozuv emas. Texnik holat (RAG, LLM, audit) klinik kiritishdan chetda.

## Capabilities and Constraints

- Interfeys tili: o‘zbekcha. Tibbiy ogohlantirish har ekranda: tashxis emas, yakuniy qaror shifokorga tegishli.
- Bemor: yosh, jins, shikoyat, anamnez, dorilar.
- Laboratoriya: troponin I, NT-proBNP, kreatinin, kaliy, glyukoza, LDL, HDL, HbA1c. Raqamlar orientir; tashxis chegarasi emas. CSV/PDF hisobot maydonlar ustidan yozilishi mumkin.
- EKG: CSV yoki WFDB (.hea + .dat), sampling rate, 12 tasma. Bo‘sh bo‘lsa tanlangan sintetik signal.
- Echo: DICOM, video yoki rasm, bir nechta fayl. EchoPrime 11 ko‘rinish.
- Holat so‘zlari bilan: tayyor, kutilmoqda, ogohlantirish, hali yo‘q, xato. Faqat rang yetarli emas.
- Agent oqimi Streamlitda ishlaydi: Tekshiruv qadamidagi «Tahlil qilish». Oqim o‘zgarmagan.

## Brand Commitments

Nomi: Kardiologik AI agent (`cardio-ai-system`). Ovoz: ehtiyotkor, aniq, tashxis e’lon qilmaydi. Sintetik ma’lumot har doim shunday yoziladi.

## Evidence on Hand

- Joriy oynaning manbasi: `src/ui/app.py`, jonli jarayon `src/ui/jonli_tahlil.py`.
- Hozirgi ko‘rinish (kodda, yangi vizual dunyo emas): fon `#ECFEFF`, aksent `#0891B2`, matn `#164E63`, Atkinson Hyperlegible, ogohlantirish `#FEF3C7` / `#D97706`.
- Sintetik paketlar: `data/sintetik/`. Bilim bazasi haqiqiyligi `HANDOFF.md` da; yangi dalil, bemor yoki aniqlik raqami o‘ylab topilmaydi. E’lon qilingan ACC/AUC bu tizimniki emas.

## Product Principles

- Bitta vazifa, bitta ekran: kiritish va xulosa aralashmaydi.
- Kardiolog har qadamda holatning to‘liqligini ko‘radi va orqaga qaytadi.
- Yetishmovchilik yashirinmaydi; tahlil to‘liq bo‘lmagan holatda ham nima yetishmasligini aytadi.
- Sintetik va haqiqiy yozuv vizual jihatdan ajraladi.
- Texnik iz (model, RAG, audit) klinik matnni siqib chiqarmaydi.

## Accessibility & Inclusion

Holat rang bilan birga so‘z bilan yoziladi. Asosiy tugmalar barmoq bilan bosiladigan o‘lchamda. Matn zich klinik ma’lumotni o‘qish uchun yetarli kontrast va qator balandligida.
