"""Jonli tahlil: fon oqimida agent ishi va interfeys uchun bosqich xabarlari.

Streamlit ga bog‘liq emas, shuning uchun alohida sinab ko‘riladi.
Natija tashxis emas; yakuniy qaror shifokorga tegishli.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

# LangGraph ning 6 bosqichi (UI chiplari uchun)
BOSQICH_NOMLARI: List[str] = [
    "1 · Qabul",
    "2 · Murakkablik",
    "3 · RAG reja",
    "4 · Vositalar",
    "5 · MDT",
    "6 · Xulosa",
]

ODATIY_HZ = 500.0
TUGADI_INDEKS = 6


def md_toza(matn: Any, uzunlik: int = 160) -> str:
    """Markdown belgilarini zararsizlantirib, matnni qisqartiradi.

    Args:
        matn: LLM yoki vosita matni (ichida *, _, $ bo‘lishi mumkin).
        uzunlik: Eng ko‘p belgi soni.

    Returns:
        Bir qatorli xavfsiz matn. Ko‘rinish uchun, tibbiy mazmunni o‘zgartirmaydi.
    """
    s = " ".join(str(matn if matn is not None else "").split())
    if len(s) > uzunlik:
        s = s[: uzunlik - 1] + "…"
    return re.sub(r"([\\`*_{}\[\]()#+!$|<>~])", r"\\\1", s)


def bosqich_holati(holat: Optional[Dict[str, Any]]) -> Tuple[int, str]:
    """Graf holatidan joriy bosqich indeksini va «hozir nima bo‘layapti» matnini topadi.

    Har tugundan keyin holat yangilanadi, shuning uchun keyingi tugun oldindan
    ma’lum. Foydalanuvchi uzoq LLM kutish paytida ham nima bajarilayotganini ko‘radi.

    Args:
        holat: LangGraph holati yoki None (hali birinchi tugun tugamagan).

    Returns:
        (indeks, matn). indeks 0..5 — joriy bosqich, 6 — tahlil tugagan.
    """
    if not holat:
        return 0, "Bemor ma’lumotlari qabul qilinmoqda"
    if holat.get("xulosa"):
        return TUGADI_INDEKS, "Tahlil tugadi"
    tarix = [str(t) for t in (holat.get("qadam_tarixi") or [])]
    oxirgi = tarix[-1] if tarix else ""
    reja = holat.get("reja") or []
    indeks = int(holat.get("reja_indeks") or 0)

    def keyingi_vosita() -> str:
        """Rejadagi navbatdagi vosita nomini matnga aylantiradi."""
        if indeks < len(reja):
            vosita = reja[indeks].get("vosita")
            return f"Qadam {indeks + 1}/{len(reja)}: {vosita} bajarilmoqda"
        return "Reja tugadi, CONTINUE/STOP baholanmoqda"

    if oxirgi.startswith("qabul_qilish"):
        return 1, "Murakkablik baholanmoqda (LLM javobi kutilmoqda)"
    if oxirgi.startswith("murakkablik"):
        return 2, "CardiacRAG: dalillar qidirilmoqda, klinik reja tuzilmoqda"
    if oxirgi.startswith("cardiac_rag_reja"):
        return 3, keyingi_vosita()
    if oxirgi.startswith("stepwise"):
        if holat.get("amal") == "CONTINUE":
            return 3, keyingi_vosita()
        if holat.get("mdt_kerak"):
            return 4, "MDT muhokamasi (MedGemma, Qwen, DeepSeek javoblari kutilmoqda)"
        return 5, "Yakuniy xulosa tayyorlanmoqda"
    if oxirgi.startswith("mdt"):
        return 5, "Yakuniy xulosa tayyorlanmoqda"
    return 3, "Stepwise: CONTINUE/STOP baholanmoqda (LLM javobi kutilmoqda)"


def oqim_bosqichlari(holat: Optional[Dict[str, Any]]) -> List[str]:
    """6 bosqich chiplari uchun holatlar: done, current, pending, skip.

    Args:
        holat: Joriy yoki yakuniy graf holati.

    Returns:
        6 ta qiymat. Murakkab bo‘lmagan holatda MDT «skip» bo‘ladi.
    """
    indeks, _matn = bosqich_holati(holat)
    nomlar: List[str] = []
    for i in range(6):
        if indeks >= TUGADI_INDEKS or i < indeks:
            nomlar.append("done")
        elif i == indeks:
            nomlar.append("current")
        else:
            nomlar.append("pending")
    if indeks >= 5 and not (holat or {}).get("mdt_natija"):
        nomlar[4] = "skip"
    return nomlar


def paket_xulosasi(paket: Optional[Dict[str, Any]]) -> str:
    """Yuklangan bemor paketini bir qatorda tavsiflaydi.

    Args:
        paket: Agentga ketadigan bemor lug‘ati (yosh, lab, EKG, echo).

    Returns:
        «yosh 45, ayol; lab 8 ko‘rsatkich; EKG 12 tasma, 8.0 s @ 500 Hz; echo 2 fayl».
    """
    p = paket or {}
    lab = p.get("laboratoriya") or {}
    signal = p.get("ecg_signal")
    ekg = "EKG yo‘q"
    if signal is not None:
        try:
            n, tasma = int(signal.shape[0]), int(signal.shape[1])
            hz = float(p.get("sampling_rate") or ODATIY_HZ)
            ekg = f"EKG {tasma} tasma, {n / hz:.1f} s @ {hz:g} Hz"
        except Exception:
            ekg = "EKG bor"
    echo = len(p.get("echo_fayllar") or [])
    return (
        f"yosh {p.get('yosh')}, {p.get('jins')}; lab {len(lab)} ko‘rsatkich; "
        f"{ekg}; echo {echo} fayl"
    )


def vosita_natijasi(vosita: str, holat: Dict[str, Any]) -> str:
    """Bajarilgan vositaning qisqa natijasini beradi.

    Args:
        vosita: Reja qadamidagi vosita nomi.
        holat: Graf holati (vosita natija maydonlari bilan).

    Returns:
        Qisqa matn (HR, ko‘rinishlar soni va h.k.). Tashxis emas.
    """
    if vosita == "lab_technician":
        lab = holat.get("lab_natija") or {}
        return f"{len(lab.get('qiymatlar') or {})} ko‘rsatkich, {len(lab.get('tokenlar') or [])} token"
    if vosita == "ecg_technician":
        t = holat.get("ecg_technician_natija") or {}
        return (
            f"sifat={t.get('sifat')}, HR≈{t.get('yurak_chastotasi_bpm')}, "
            f"R={t.get('urishlar_soni')}"
        )
    if vosita == "electrophysiologist":
        e = holat.get("ep_natija") or {}
        return (
            f"HR {e.get('yurak_chastotasi_bpm')}, PR {e.get('pr_ms')} ms, "
            f"QRS {e.get('qrs_ms')} ms, QTc {e.get('qtc_bazett_ms')} ms"
        )
    if vosita == "echo_technician":
        e = holat.get("echo_natija") or {}
        k = e.get("korinishlar") or []
        return f"{len(k)} ko‘rinish: " + ", ".join(str(x) for x in k)
    if vosita == "echo_segmenter":
        m = holat.get("echo_mask") or {}
        if m.get("ok"):
            return f"LV ulush={m.get('ulush')}"
        return str(m.get("xabar") or "maska yo‘q")
    if vosita == "cardiology_fellow":
        f = holat.get("fellow_natija") or {}
        return f"manba={f.get('manba')}"
    return ""


def jonli_xabarlar(
    paket: Optional[Dict[str, Any]],
    holat: Optional[Dict[str, Any]],
    indeks: int,
) -> List[Dict[str, Any]]:
    """Jonli panel uchun bosqich xabarlarini tuzadi (belgi, matn, ichki).

    Birinchi qator tahlil boshlanishi bilan darhol chiqadi: bemor ma’lumotlari
    yuklangani. Keyin murakkablik, reja (qadamlari bilan), vositalar, stepwise,
    MDT va xulosa.

    Args:
        paket: Agentga ketgan bemor lug‘ati.
        holat: Hozirgi graf holati yoki None.
        indeks: bosqich_holati() dagi joriy bosqich.

    Returns:
        [{belgi, matn, ichki}]. «matn» da markdown qalin belgi bor, dinamik qism toza.
    """
    h = holat or {}
    qator: List[Dict[str, Any]] = [
        {
            "belgi": "✅",
            "matn": "**Bemor ma’lumotlari yuklandi** — " + md_toza(paket_xulosasi(paket), 200),
            "ichki": False,
        }
    ]

    if h.get("murakkablik"):
        qator.append(
            {
                "belgi": "✅",
                "matn": f"**Murakkablik baholandi: {md_toza(h['murakkablik'])}** — "
                + md_toza(h.get("murakkablik_sababi")),
                "ichki": False,
            }
        )
    else:
        qator.append(
            {
                "belgi": "▶" if indeks <= 1 else "○",
                "matn": "**Murakkablik baholash**" + (" — LLM javobi kutilmoqda…" if indeks <= 1 else ""),
                "ichki": False,
            }
        )

    reja = h.get("reja") or []
    reja_indeks = int(h.get("reja_indeks") or 0)
    tarix = [str(x) for x in (h.get("qadam_tarixi") or [])]
    oxirgi = tarix[-1] if tarix else ""
    # Vosita faqat reja tuzilgach yoki stepwise CONTINUE dan keyin ishlaydi;
    # vosita tugagach navbat stepwise baholashda (keyingi qadam hali kutmoqda).
    vosita_ishlayapti = indeks == 3 and (
        oxirgi.startswith("cardiac_rag_reja")
        or (oxirgi.startswith("stepwise") and h.get("amal") == "CONTINUE")
    )
    if reja:
        qator.append(
            {
                "belgi": "✅",
                "matn": f"**Klinik reja tuzildi** — {len(reja)} qadam, "
                f"CardiacRAG dalillari: {len(h.get('rag_dalillar') or [])}",
                "ichki": False,
            }
        )
        for i, q in enumerate(reja):
            hol = str(q.get("holat") or "pending")
            vosita = str(q.get("vosita") or "")
            if hol == "done":
                belgi, oxiri = "✅", vosita_natijasi(vosita, h)
            elif hol == "xato":
                belgi, oxiri = "❌", "xato (tarixga qarang)"
            elif i == reja_indeks and vosita_ishlayapti:
                belgi, oxiri = "▶", "bajarilmoqda…"
            else:
                belgi, oxiri = "○", md_toza(q.get("tavsif"), 80)
            qator.append(
                {
                    "belgi": belgi,
                    "matn": f"{md_toza(q.get('id'), 12)} · **{md_toza(vosita, 30)}** — {md_toza(oxiri)}",
                    "ichki": True,
                }
            )
    else:
        qator.append(
            {
                "belgi": "▶" if indeks == 2 else "○",
                "matn": "**CardiacRAG reja tuzish**" + (" — dalillar qidirilmoqda…" if indeks == 2 else ""),
                "ichki": False,
            }
        )

    if h.get("baho_sababi"):
        qator.append(
            {
                "belgi": "✅",
                "matn": f"**Stepwise: {md_toza(h.get('amal'))}** — {md_toza(h.get('baho_sababi'))}",
                "ichki": False,
            }
        )

    mdt = h.get("mdt_natija")
    if mdt:
        qator.append(
            {
                "belgi": "✅",
                "matn": f"**MDT muhokamasi** — {md_toza(mdt.get('raund_soni'))} raund, "
                f"konsensus={md_toza(mdt.get('konsensus'))}",
                "ichki": False,
            }
        )
    elif indeks == 4:
        qator.append(
            {"belgi": "▶", "matn": "**MDT muhokamasi** — modellar javobi kutilmoqda…", "ichki": False}
        )

    if h.get("xulosa"):
        qator.append({"belgi": "✅", "matn": "**Yakuniy xulosa tayyor**", "ichki": False})
    else:
        qator.append(
            {
                "belgi": "▶" if indeks == 5 else "○",
                "matn": "**Yakuniy xulosa**",
                "ichki": False,
            }
        )
    return qator


def _ishchi(agent: Any, paket: Dict[str, Any], job: Dict[str, Any]) -> None:
    """Agentni fon oqimida yuritib, har tugundan keyingi holatni job ga yozadi.

    Args:
        agent: ChiefCardiologist (run_oqim va run bilan).
        paket: Bemor lug‘ati.
        job: Umumiy ish lug‘ati (holat, xato, tugadi).

    Returns:
        None. Xato job['xato'] ga yoziladi; Streamlit chaqiruvlari yo‘q.
    """
    try:
        oxirgi: Optional[Dict[str, Any]] = None
        for holat in agent.run_oqim(paket):
            if job.get("bekor"):
                break
            oxirgi = holat
            job["holat"] = holat
        if oxirgi is None and not job.get("bekor"):
            job["holat"] = agent.run(paket)
    except Exception as exc:  # tahlil yiqilsa ham interfeys to‘xtamasin
        job["xato"] = str(exc)
    finally:
        job["tugagan"] = time.time()
        job["tugadi"] = True


class TahlilOmbori:
    """Bitta tahlil ishini fon oqimida saqlaydi (sahifa yangilansa ham yo‘qolmaydi).

    Bir vaqtda bitta ish. Mahalliy bitta foydalanuvchi uchun mo‘ljallangan.
    """

    def __init__(self) -> None:
        """Bo‘sh ombor yaratadi."""
        self.lock = threading.Lock()
        self.job: Optional[Dict[str, Any]] = None
        self._hisob = 0

    def joriy(self) -> Optional[Dict[str, Any]]:
        """Oxirgi ishni qaytaradi.

        Returns:
            Ish lug‘ati yoki None.
        """
        return self.job

    def ishlayapti(self) -> bool:
        """Ish hali tugamaganini bildiradi.

        Returns:
            True — fon oqimi ishlayapti.
        """
        j = self.job
        return bool(j) and not j["tugadi"]

    def boshlash(
        self,
        agent: Any,
        paket: Dict[str, Any],
        ekg_xabar: str = "",
        ishchi: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """Yangi tahlilni fon oqimida boshlaydi.

        Args:
            agent: ChiefCardiologist.
            paket: Bemor lug‘ati.
            ekg_xabar: EKG izohi (interfeysda ko‘rsatish uchun).
            ishchi: Test uchun almashtiriladigan ishchi funksiya.

        Returns:
            Yangi ish lug‘ati; oldingi ish hali tugamagan bo‘lsa None.
        """
        with self.lock:
            if self.ishlayapti():
                return None
            self._hisob += 1
            job: Dict[str, Any] = {
                "id": self._hisob,
                "paket": paket,
                "holat": None,
                "xato": None,
                "tugadi": False,
                "bekor": False,
                "boshlandi": time.time(),
                "tugagan": None,
                "ekg_xabar": ekg_xabar,
            }
            self.job = job
        threading.Thread(
            target=ishchi or _ishchi,
            args=(agent, paket, job),
            daemon=True,
            name="tahlil-ishchi",
        ).start()
        return job

    def bekor_qilish(self) -> None:
        """Natijani kutmaslik: ish tugagan deb belgilanadi, fon oqimi o‘zi tugaydi.

        Returns:
            None. Qisman natija saqlanadi, xulosa bo‘lmasligi mumkin.
        """
        j = self.job
        if j and not j["tugadi"]:
            j["bekor"] = True
            j["tugadi"] = True
            j["tugagan"] = time.time()


def ish_holati(job: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Tugagan ishdan interfeys uchun yakuniy agent holatini yasaydi.

    Args:
        job: Ish lug‘ati.

    Returns:
        Agent holati. Xato yoki bekor qilishda tushunarli ehtiyotkor xulosali holat.
    """
    if not job:
        return None
    holat = job.get("holat")
    if job.get("xato") and not (holat or {}).get("xulosa"):
        return {
            "amal": "STOP",
            "xulosa": (
                f"Tahlil bajarilmadi: {job['xato']}. "
                "Bu tashxis emas; shifokor bilan ko‘rib chiqing."
            ),
            "qadam_tarixi": [str(job["xato"])],
            "ecg_natija": None,
            "rag_dalillar": [],
        }
    if job.get("bekor") and not (holat or {}).get("xulosa"):
        qisman = dict(holat or {})
        qisman.setdefault("qadam_tarixi", [])
        qisman["amal"] = "STOP"
        qisman["xulosa"] = (
            "Tahlil kutilmasdan to‘xtatildi, yakuniy xulosa tayyor emas. "
            "Qisman natijalar bo‘limlarda. Bu tashxis emas; shifokor bilan ko‘rib chiqing."
        )
        return qisman
    return holat


def ketma_ket(holatlar: Iterable[Dict[str, Any]]) -> List[Tuple[int, str]]:
    """Sinov uchun: holatlar ketma-ketligidagi bosqich indekslari va matnlari.

    Args:
        holatlar: Graf holatlari.

    Returns:
        [(indeks, matn)] ro‘yxati.
    """
    return [bosqich_holati(h) for h in holatlar]
