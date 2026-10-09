"""Ko‘p tarmoqli munozara (MDT): Qwen, MedGemma va DeepSeek-R1.

Ketma-ketlik:
t=1: Qwen(I), MedGemma(I), D1=DeepSeek(Q1,P1);
t=2: ikkalasi (D1, Z), D2=DeepSeek(Q2,P2);
t>=3 toq: (D, I), juft: (D, Z); D_t=DeepSeek(Q_t,P_t).
To‘xtash (t>=3): agree(Q_t, D_{t-1}) va agree(P_t, D_{t-1}), yoki t=T.
Tasvirlar rolga yopishgan: I raundida MedGemma still, Qwen video. Tashxis emas.
"""

from __future__ import annotations

import base64
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llm.client import llm_chat, llm_chat_natija, llm_json, model_qisqacha, vlm_sozlama
from tools.ecg_tool import ekg_qisqa_png

MAX_RAUND = 3
MAX_VIDEO_KADR = 6
VLM_TIMEOUT = 180.0
# Shuncha ketma-ket sondan keyin matn EKG/echo namunasi hisoblanadi, xulosa emas
_MATRITSA_SONI = 24
_RAQAM_MATRITSA = re.compile(
    r"(?:[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?[\s,;\[\]\(\)]+){"
    + str(_MATRITSA_SONI)
    + r",}"
)


def _data_url(png: bytes) -> str:
    """PNG ni OpenAI-mos image_url data URI ga aylantiradi.

    Args:
        png: Rasm baytlari.

    Returns:
        data:image/png;base64,...
    """
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _kadr_png(kadr: Any, max_tomon: int = 384) -> Optional[bytes]:
    """Kulrang/RGB kaderni kichik PNG qiladi (VLM yukini cheklash).

    Args:
        kadr: 2D yoki 3D massiv.
        max_tomon: Eng uzun tomon (piksel).

    Returns:
        PNG yoki None.
    """
    try:
        import numpy as np

        try:
            import cv2
        except ImportError:
            return None
        arr = np.asarray(kadr)
        if arr.ndim == 2:
            rgb = np.stack([arr, arr, arr], axis=-1)
        else:
            rgb = arr[..., :3] if arr.shape[-1] >= 3 else arr
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        if max(h, w) > max_tomon and h > 0 and w > 0:
            scale = max_tomon / float(max(h, w))
            rgb = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))))
        ok, buf = cv2.imencode(".png", rgb)
        return bytes(buf) if ok else None
    except Exception:
        return None


def _kadr_tanla(kadrlar: Sequence[Any], max_n: int = MAX_VIDEO_KADR) -> List[Any]:
    """Cine/video kadrlarini teng oraliqda tanlaydi.

    Args:
        kadrlar: To‘liq ketma-ketlik.
        max_n: Yuboriladigan yuqori chegara.

    Returns:
        Tanlangan kadrlar.
    """
    if not kadrlar:
        return []
    n = len(kadrlar)
    if n <= max_n:
        return list(kadrlar)
    return [kadrlar[int(round(i * (n - 1) / (max_n - 1)))] for i in range(max_n)]


def _echo_yozuvlar(bemor: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Bemor echo fayllaridan kadr yozuvlarini o‘qiydi.

    Args:
        bemor: echo_fayllar / echo_bayt.

    Returns:
        [{nom, kadrlar, video}].
    """
    from tools.echo_yuklash import echo_yollardan_oqish

    juft = []
    roy = bemor.get("echo_fayllar") or []
    if isinstance(roy, (list, tuple)):
        for element in roy:
            if isinstance(element, dict) and element.get("bayt"):
                juft.append((str(element.get("nom") or "echo.dcm"), bytes(element["bayt"])))
    if bemor.get("echo_bayt"):
        juft.append((str(bemor.get("echo_fayl_nomi") or "echo.dcm"), bytes(bemor["echo_bayt"])))
    if not juft:
        return []
    natija: List[Dict[str, Any]] = []
    for yoz in (echo_yollardan_oqish(juft).get("yozuvlar") or []):
        nom = str(yoz.get("nom") or "")
        kadrlar = yoz.get("kadrlar") or []
        if not kadrlar:
            continue
        past = nom.lower()
        video = past.endswith((".mp4", ".avi", ".mov", ".mpg", ".mpeg", ".mkv")) or len(kadrlar) >= 2
        natija.append({"nom": nom, "kadrlar": kadrlar, "video": video})
    return natija


def mdt_vizual_yig(
    bemor: Optional[Dict[str, Any]] = None,
    ecg: Optional[Dict[str, Any]] = None,
    segment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """MedGemma uchun still rasmlar va Qwen uchun video kadrlar.

    Args:
        bemor: EKG signal va echo fayllar.
        ecg: EP o‘lchovlari (tasma).
        segment: LV overlay PNG.

    Returns:
        still [{id,png}], video [{id,png}], yoq_dalillar. Tashxis emas.
    """
    bemor = bemor or {}
    still: List[Dict[str, Any]] = []
    video: List[Dict[str, Any]] = []
    yoq: List[str] = []

    sig = None
    for kalit in ("ecg_signal", "ekg", "signal"):
        if bemor.get(kalit) is not None:
            sig = bemor[kalit]
            break
    if sig is not None:
        tasma = (ecg or {}).get("tasma") or "II"
        sr = float(bemor.get("sampling_rate") or 500.0)
        png = ekg_qisqa_png(sig, sampling_rate=sr, tasma=str(tasma))
        if png:
            still.append({"id": "ecg_grafik", "png": png})
    else:
        yoq.append("ecg_grafik")

    if segment and segment.get("overlay_png"):
        still.append({"id": "lv_overlay", "png": bytes(segment["overlay_png"])})
    else:
        yoq.append("lv_overlay")

    yozuvlar = _echo_yozuvlar(bemor)
    if not yozuvlar:
        yoq.append("echo")
        yoq.append("echo_video")
    else:
        cine_bor = False
        for i, yoz in enumerate(yozuvlar):
            kadrlar = yoz.get("kadrlar") or []
            bir = _kadr_png(kadrlar[0]) if kadrlar else None
            if bir:
                still.append({"id": f"echo_still_{i}", "png": bir})
            if yoz.get("video"):
                cine_bor = True
                for j, kadr in enumerate(_kadr_tanla(kadrlar)):
                    png = _kadr_png(kadr)
                    if png:
                        video.append({"id": f"echo_video_{i}_{j}", "png": png})
        if not cine_bor:
            yoq.append("echo_video")
            if still:
                # still kadrni Qwen ham ko‘rsin, lekin cine emasligini bilsin
                for r in still:
                    if str(r.get("id") or "").startswith("echo_still_"):
                        video.append({"id": r["id"] + "_qwen", "png": r["png"]})

    return {
        "still": still,
        "video": video,
        "yoq_dalillar": yoq,
        "still_id": [r["id"] for r in still],
        "video_id": [r["id"] for r in video],
    }


def _multimodal_qism(matn: str, rasmlar: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Matn + rasmlarni chat content ro‘yxatiga yig‘adi.

    Args:
        matn: I/Z va ko‘rsatma.
        rasmlar: [{png}].

    Returns:
        OpenAI-mos content qismlari.
    """
    qismlar: List[Dict[str, Any]] = [{"type": "text", "text": matn}]
    for r in rasmlar:
        png = r.get("png")
        if png:
            qismlar.append({"type": "image_url", "image_url": {"url": _data_url(bytes(png))}})
    return qismlar


def raqam_matritsasi_bormi(matn: str) -> bool:
    """Matnda EKG/echo namunasi kabi uzun raqam ketma-ketligi borligini tekshiradi.

    Args:
        matn: I yoki Z.

    Returns:
        True — MedGemma ga matritsa emas, sub-agent xulosasi ketadi.
    """
    if not matn:
        return False
    return _RAQAM_MATRITSA.search(matn) is not None


def _xulosa_satri(sarlavha: str, obyekt: Any) -> str:
    """Vosita javobidan faqat tayyor matn xulosasini oladi.

    Args:
        sarlavha: LAB, EKG, ECHO, LV.
        obyekt: Sub-agent lug‘ati.

    Returns:
        «SARLAVHA: xabar» yoki bo‘sh. Raqam matritsasi kirmaydi.
    """
    if not isinstance(obyekt, dict):
        return ""
    for kalit in ("xabar", "rag_satr", "matn"):
        matn = obyekt.get(kalit)
        if isinstance(matn, str) and matn.strip() and not raqam_matritsasi_bormi(matn):
            return f"{sarlavha}: {matn.strip()}"
    return ""


def medgemma_matnlari(
    xom_i: str,
    oraliq_z: str,
    lab: Optional[Dict[str, Any]] = None,
    ecg: Optional[Dict[str, Any]] = None,
    echo: Optional[Dict[str, Any]] = None,
    segment: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Uzun EKG/echo matritsasi bo‘lsa MedGemma ga faqat matn xulosalarini beradi.

    Args:
        xom_i: Bemor qisqasi.
        oraliq_z: Vositalar matni (ichida signal bo‘lishi mumkin).
        lab, ecg, echo, segment: Sub-agent javoblari.

    Returns:
        (I, Z). Matritsa yo‘q bo‘lsa kirish o‘zgarmaydi. Tashxis emas.
    """
    if not (raqam_matritsasi_bormi(xom_i) or raqam_matritsasi_bormi(oraliq_z)):
        return xom_i, oraliq_z
    i_qator = [q for q in (xom_i or "").splitlines() if not raqam_matritsasi_bormi(q)]
    i_toza = "\n".join(i_qator).strip()
    if not i_toza or raqam_matritsasi_bormi(i_toza):
        i_toza = "Bemor qisqasi: EKG/echo raqam matritsasi yuborilmadi."
    xulosalar = [
        _xulosa_satri("LAB", lab),
        _xulosa_satri("EKG", ecg),
        _xulosa_satri("ECG_TECH", (ecg or {}).get("technician") if isinstance(ecg, dict) else None),
        _xulosa_satri("ECG_EP", (ecg or {}).get("ep") if isinstance(ecg, dict) else None),
        _xulosa_satri("ECHO", echo),
        _xulosa_satri("LV", segment),
    ]
    for qator in (oraliq_z or "").splitlines():
        if qator.strip().upper().startswith("FELLOW:") and not raqam_matritsasi_bormi(qator):
            xulosalar.append(qator.strip())
    z_toza = "\n".join(q for q in xulosalar if q)
    if not z_toza:
        z_toza = "Sub-agent matn xulosasi yo‘q. Raqam matritsasi yuborilmadi."
    return i_toza, z_toza


# MedGemma thought izidan keyin haqiqiy izoh shu belgilardan so‘ng keladi
_YAKUN_BELGILARI = (
    "**Final Output:**",
    "Final Output:",
    "Looks good.",
    "**Drafting (Uzbek):**",
    "**Drafting:**",
)


def _takror_qisqart(matn: str) -> str:
    """Bir xil mulohaza ikki marta yopishtirilgan bo‘lsa, bittasini qoldiradi.

    Args:
        matn: Model javobining yakuniy qismi.

    Returns:
        Takrorsiz matn.
    """
    t = matn.strip()
    if len(t) >= 80:
        bosh = t[:60]
        ikkinchi = t.find(bosh, 40)
        if ikkinchi != -1 and ikkinchi < int(len(t) * 0.75):
            t = t[:ikkinchi].strip()
    bloklar = []
    korilgan = set()
    for blok in t.split("\n\n"):
        kalit = " ".join(blok.split())
        if len(kalit) < 8 or kalit in korilgan:
            continue
        korilgan.add(kalit)
        bloklar.append(blok.strip())
    return "\n\n".join(bloklar).strip()


def _mulohazani_ajrat(matn: str) -> str:
    """Thought izidan klinik mulohazani ajratadi.

    MedGemma ba’zan javobni ``thought`` bilan boshlab bemor qisqasi va
    oraliq natijani qayta yozadi. Yakuniy jumlalar iz oxirida qoladi.
    Ekranda shu iz mulohaza o‘rnini bosmasin.

    Args:
        matn: Modelning xom javobi.

    Returns:
        Shifokorga ko‘rsatiladigan mulohaza. Bo‘lmasa qisqa ogohlantirish.
        Tashxis emas.
    """
    if not matn or not str(matn).strip():
        return ""
    t = str(matn).strip()
    past = t.lower()
    if past.startswith("thought") or past.startswith("**thought"):
        topildi = ""
        for belgi in _YAKUN_BELGILARI:
            idx = t.rfind(belgi)
            if idx == -1:
                continue
            qolgan = t[idx + len(belgi) :].strip()
            review = qolgan.find("**Review")
            if review != -1:
                qolgan = qolgan[:review].strip()
            if len(qolgan) >= 40:
                topildi = qolgan
                break
        if not topildi:
            return (
                "Model ichki reja izini qaytardi, yakuniy izoh ajralmadi. "
                "Shifokor berilgan ma’lumotni o‘zi ko‘rsin. Tashxis emas."
            )
        t = topildi
    t = _takror_qisqart(t)
    for belgi in (
        "**Klinik mulohaza:**",
        "Klinik mulohaza:",
        "**Ehtiyotkor Izoh:**",
        "**Ehtiyotkor izoh:**",
    ):
        if t.startswith(belgi):
            t = t[len(belgi) :].strip()
    return t.strip()


def raund_kirish_turi(t: int) -> str:
    """t-raundda Qwen va MedGemma qaysi matnni olishini belgilaydi.

    Args:
        t: 1 dan boshlanadigan raund.

    Returns:
        ``I`` (faqat xom kirish), ``DZ`` (oldingi D va Z), ``DI`` (oldingi D va I).
    """
    if t <= 1:
        return "I"
    if t % 2 == 0:
        return "DZ"
    return "DI"


def toxtash_sharti(t: int, chegara: int, qwen_agree: bool, med_agree: bool) -> bool:
    """Formula bo‘yicha iteratsiyani to‘xtatish.

    t=1 va t=2 har doim bajariladi. t>=3 da ikkala rol oldingi D bilan
    kelishsa yoki t=T bo‘lsa to‘xtaydi.

    Args:
        t: Joriy raund.
        chegara: T, maksimal raund.
        qwen_agree: agree(Q_t, D_{t-1}).
        med_agree: agree(P_t, D_{t-1}).

    Returns:
        True — sikl shu raundda tugaydi.
    """
    if t < 3:
        return False
    return bool(qwen_agree and med_agree) or t >= chegara


def _foydalanuvchi_matn(
    bemor_qisqa: str,
    oraliq: str,
    oldingi_d: str,
    rasmlar_id: Sequence[str],
    yoq: Sequence[str],
    video: bool,
) -> str:
    """Shu raundning formuladagi kirishini yozadi; bo‘sh qism tushadi.

    Args:
        bemor_qisqa: I. Bo‘sh bo‘lsa bu raundda yuborilmaydi.
        oraliq: Z. Bo‘sh bo‘lsa yuborilmaydi.
        oldingi_d: D_{t-1}. Bo‘sh bo‘lsa yuborilmaydi.
        rasmlar_id: Yuborilgan rasm identifikatorlari.
        yoq: Yo‘q modalitetlar.
        video: Qwen cine/kadr ketma-ketligi.

    Returns:
        Prompt matni. Model klinik izoh yozishi kerak, tashxis emas.
    """
    tur = "video yoki kadrlar" if video else "tasvirlar"
    qatorlar = ["Quyidagi ma’lumotni o‘qing. Javobda ro‘yxatni qayta ko‘chirmang.", ""]
    if bemor_qisqa:
        qatorlar.append(f"Bemor qisqasi:\n{bemor_qisqa}\n")
    if oraliq:
        qatorlar.append(f"Oraliq natijalar:\n{oraliq}\n")
    if oldingi_d:
        qatorlar.append(f"Oldingi umumlashtirish:\n{oldingi_d}\n")
    qatorlar.append(f"Yuborilgan {tur}: {', '.join(rasmlar_id) or 'yo‘q'}.")
    qatorlar.append(f"Yo‘q dalillar (o‘ylab topmang): {', '.join(yoq) or '—'}.")
    qatorlar.append("")
    qatorlar.append("Ehtiyotkor klinik izoh (tashxis emas):")
    return "\n".join(qatorlar)


def _shablon(rol: str, xom_i: str, oraliq_z: str, rasmlar_id: Sequence[str], yoq: Sequence[str]) -> str:
    """VLM yo‘qida mulohaza o‘rniga I/Z ni chop etmaydigan ehtiyotkor matn.

    Args:
        rol: MedGemma yoki Qwen tavsifi.
        xom_i, oraliq_z: Kontekst bor-yo‘qligini bilish uchun (matni chiqarilmaydi).
        rasmlar_id, yoq: Vizual dalil holati.

    Returns:
        Shablon yozuv. Tashxis emas.
    """
    bosh = "" if (xom_i or oraliq_z) else " Bemor qisqasi va oraliq natija bo‘sh."
    return (
        f"{rol}: model javobi yo‘q, shablon. Klinik mulohaza shakllanmadi.{bosh} "
        f"Rasmlar: {', '.join(rasmlar_id) or 'yo‘q'}; yo‘q dalil: {', '.join(yoq) or '—'}. "
        "I va Z alohida ko‘rsatiladi — ular mulohaza emas. "
        "Tashxis emas; shifokor ko‘rigi shart."
    )


def _rol_javobi(
    rol_nom: str,
    tizim: str,
    bemor_qisqa: str,
    oraliq: str,
    oldingi_d: str,
    rasmlar: Sequence[Dict[str, Any]],
    yoq: Sequence[str],
    video: bool,
) -> Tuple[str, str]:
    """Bitta VLM/LLM rolida shu raundning izohini oladi.

    Qwen va MedGemma bir-birining joriy matnini ko‘rmaydi: ikkalasi ham
    faqat formuladagi kirishni (I, yoki D va Z, yoki D va I) oladi.

    Args:
        rol_nom: medgemma yoki qwen_vl (ulanish).
        tizim: System prompt.
        bemor_qisqa: I yoki bo‘sh.
        oraliq: Z yoki bo‘sh.
        oldingi_d: D_{t-1} yoki bo‘sh.
        rasmlar: PNG lar. Z raundida bo‘sh.
        yoq: Yo‘q dalillar.
        video: True — Qwen cine.

    Returns:
        (matn, manba=vlm|llm|shablon). Tashxis emas.
    """
    ids = [str(r.get("id") or "") for r in rasmlar]
    user = _foydalanuvchi_matn(bemor_qisqa, oraliq, oldingi_d, ids, yoq, video)
    soz = vlm_sozlama(rol_nom)
    if soz and rasmlar:
        javob = llm_chat(
            [
                {"role": "system", "content": tizim},
                {"role": "user", "content": _multimodal_qism(user, rasmlar)},
            ],
            model=soz["model"],
            baza_url=soz["base_url"],
            api_kalit=soz["api_key"],
            timeout=VLM_TIMEOUT,
            max_token=900,
        )
        if javob:
            return _mulohazani_ajrat(javob), "vlm"
    if soz:
        javob = llm_chat(
            [
                {"role": "system", "content": tizim},
                {"role": "user", "content": user},
            ],
            model=soz["model"],
            baza_url=soz["base_url"],
            api_kalit=soz["api_key"],
            timeout=VLM_TIMEOUT,
            max_token=900,
        )
        if javob:
            return _mulohazani_ajrat(javob), "llm"
    rol_matn = "MedGemma (tasvir)" if rol_nom == "medgemma" else "Qwen2.5-VL (video)"
    return _shablon(rol_matn, bemor_qisqa, oraliq, ids, yoq), "shablon"


def _raund_qismlari(
    kirish: str,
    xom_i: str,
    oraliq_z: str,
    med_i: str,
    med_z: str,
    matritsa: bool,
    oldingi_d: str,
) -> Tuple[str, str, str]:
    """Formuladagi kirishni bemor / oraliq / D qismlariga ajratadi.

    Matritsa bo‘lsa MedGemma va Qwen tozalangan xulosani oladi, namuna emas.

    Args:
        kirish: ``I``, ``DZ`` yoki ``DI``.
        xom_i, oraliq_z: Asl I va Z.
        med_i, med_z: Matritsasiz nusxa.
        matritsa: I yoki Z da uzun raqam ketma-ketligi bor.
        oldingi_d: D_{t-1}.

    Returns:
        (bemor_qisqa, oraliq, d). Bo‘sh qism shu raundda yuborilmaydi.
    """
    i_matn = med_i if matritsa else xom_i
    z_matn = med_z if matritsa else oraliq_z
    if kirish == "DZ":
        return "", z_matn, oldingi_d
    if kirish == "DI":
        return i_matn, "", oldingi_d
    return i_matn, "", ""


def _raund_rasmlari(
    rol: str,
    kirish: str,
    matritsa: bool,
    viz: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """I raundida rolga mos tasvirni beradi; Z raundida faqat matn qoladi.

    Args:
        rol: medgemma yoki qwen_vl.
        kirish: ``I``, ``DZ``, ``DI``.
        matritsa: True bo‘lsa MedGemma kadr olmaydi.
        viz: mdt_vizual_yig natijasi.

    Returns:
        PNG ro‘yxati. Tashxis emas.
    """
    if kirish == "DZ":
        return []
    if rol == "medgemma":
        if matritsa:
            return []
        return list(viz.get("still") or [])
    return list(viz.get("video") or [])


def _sozlar(matn: str) -> set:
    """Kelishuv uchun qisqa so‘zlarsiz to‘plam.

    Args:
        matn: Izoh.

    Returns:
        Kamida 4 harfli so‘zlar. Shablon iboralar tushirilgan.
    """
    shablon_soz = {"tashxis", "emas", "shifokor", "klinik", "javobi", "model", "shablon"}
    topilgan = set(re.findall(r"[0-9A-Za-zʻʼ'’]{4,}", (matn or "").lower()))
    return {s for s in topilgan if s not in shablon_soz}


def _matn_kelishuvi(a: str, b: str) -> bool:
    """API yo‘qida ikki matn bir xil klinik faktlarni aytayotganini taxmin qiladi.

    Args:
        a: Joriy rol izohi (Q yoki P).
        b: Oldingi D.

    Returns:
        True — sezilarli so‘z kesishmasi. Tashxis emas.
    """
    ta, tb = _sozlar(a), _sozlar(b)
    if len(ta) < 4 or len(tb) < 4:
        return False
    umumiy = ta & tb
    if len(umumiy) < 4:
        return False
    return len(umumiy) / float(len(ta | tb)) >= 0.35


def _deepseek_d(qwen: str, med: str) -> Tuple[str, str]:
    """D_t = DeepSeek-R1(Q_t, P_t).

    Args:
        qwen: Shu raunddagi Qwen izohi.
        med: Shu raunddagi MedGemma izohi.

    Returns:
        (D matni, manba=llm|shablon). Yangi o‘lchov qo‘shilmaydi. Tashxis emas.
    """
    # Reasoner mulohazasi max_tokens dan ketadi; 900 da content bo‘sh qoladi
    javob, xato = llm_chat_natija(
        [
            {
                "role": "system",
                "content": (
                    "Siz DeepSeek-R1. Qwen va MedGemma izohlaridan bitta D sintez yozing. "
                    "Tashxis qo‘ymang. Yangi o‘lchov va kasallik nomi qo‘shmang. "
                    "Ikkalasi mos kelmasa, farqni yozing. 5–8 jumla. "
                    "Yakuniy javobni content da qoldiring."
                ),
            },
            {
                "role": "user",
                "content": f"Qwen:\n{qwen}\n\nMedGemma:\n{med}\n\nD:",
            },
        ],
        max_token=4096,
        timeout=180.0,
    )
    if javob:
        return _mulohazani_ajrat(javob), "llm"
    sabab = xato or "javob bo‘sh"
    return (
        f"DeepSeek-R1 sintez qilmadi ({sabab}). "
        "Quyidagi qator model sintezi emas. "
        f"Qwen: {qwen[:280]} MedGemma: {med[:280]} "
        "Hisob to‘ldirilgach D qayta hisoblanadi. Tashxis emas; shifokor ko‘rigi shart."
    ), "shablon"


def _oldingi_d_bilan_kelishuv(qwen: str, med: str, oldingi_d: str) -> Tuple[bool, bool, str]:
    """agree(Q_t, D_{t-1}) va agree(P_t, D_{t-1}).

    Args:
        qwen: Joriy Qwen izohi.
        med: Joriy MedGemma izohi.
        oldingi_d: O‘tgan raunddagi DeepSeek D.

    Returns:
        (qwen_agree, medgemma_agree, sabab). Tashxis emas.
    """
    zaxira = {
        "qwen_agree": _matn_kelishuvi(qwen, oldingi_d),
        "medgemma_agree": _matn_kelishuvi(med, oldingi_d),
        "sabab": "Qoida: oldingi D bilan so‘z kesishmasi.",
    }
    natija = llm_json(
        tizim=(
            "MDT kelishuvini baholang. JSON: "
            '{"qwen_agree":true|false,"medgemma_agree":true|false,"sabab":"qisqa"}. '
            "qwen_agree — Qwen izohi oldingi D bilan klinik faktlarda mos. "
            "medgemma_agree — MedGemma izohi oldingi D bilan mos. "
            "Yangi tashxis yoki zid o‘lchov bo‘lsa false. Tashxis qo‘ymang."
        ),
        foydalanuvchi=(
            f"Oldingi D:\n{oldingi_d}\n\nQwen:\n{qwen}\n\nMedGemma:\n{med}"
        ),
        zaxira=zaxira,
    )
    return (
        bool(natija.get("qwen_agree")),
        bool(natija.get("medgemma_agree")),
        str(natija.get("sabab") or zaxira["sabab"]),
    )


def mdt_munozara(
    xom_i: str,
    oraliq_z: str,
    max_raund: int = MAX_RAUND,
    bemor: Optional[Dict[str, Any]] = None,
    lab: Optional[Dict[str, Any]] = None,
    ecg: Optional[Dict[str, Any]] = None,
    echo: Optional[Dict[str, Any]] = None,
    segment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Qwen, MedGemma va DeepSeek-R1 ni berilgan ketma-ketlikda yurgizadi.

    t=1: ikkala VLM faqat I. t=2: ikkalasi D1 va Z. t>=3 toq: D va I,
    juft: D va Z. Har raunddan keyin D_t = DeepSeek(Q_t, P_t).
    Yakuniy matn oxirgi D. Tashxis emas.

    Args:
        xom_i: Bemor va xom signallar qisqasi (I).
        oraliq_z: Lab, EKG, echo, fellow (Z).
        max_raund: T, maksimal raund.
        bemor: Vizual fayllar (EKG/echo).
        lab, ecg, echo, segment: Oraliq vosita chiqishlari (Z ga qo‘shimcha).

    Returns:
        raundlar, oxirgi D, kelishuv, manba. Tashxis emas.
    """
    viz = mdt_vizual_yig(bemor, ecg, segment)
    z_qosh = oraliq_z or ""
    if lab and (lab.get("rag_satr") or lab.get("xabar")):
        if "LAB:" not in z_qosh:
            z_qosh += "\nLAB: " + str(lab.get("rag_satr") or lab.get("xabar"))
    if echo and echo.get("xabar") and "ECHO:" not in z_qosh:
        z_qosh += "\nECHO: " + str(echo.get("xabar"))
    med_i, med_z = medgemma_matnlari(xom_i, z_qosh, lab, ecg, echo, segment)
    matritsa = med_i != xom_i or med_z != z_qosh
    yoq = list(viz.get("yoq_dalillar") or [])

    med_tizim = (
        "Siz tibbiy tasvirlar (EKG grafik, echo still, LV overlay) bo‘yicha yordamchisiz. "
        "Faqat berilgan faktlar asosida 5–8 jumlalik ehtiyotkor klinik izoh yozing: "
        "nima qayd etilgan, nima yetishmaydi, shifokor nimani o‘zi tekshirsin. "
        "Kirish matnini qayta ko‘chirmang. Ichki reja yozmang. "
        "Javob darhol izohdan boshlansin. "
        "Berilgan son va ko‘rinish nomlarini boshqacha talqin qilmang. "
        "Yangi o‘lchov va kasallik nomi qo‘shmang. "
        "O‘zbek yoki ingliz. Yakuniy qaror shifokorga tegishli."
    )
    qwen_tizim = (
        "Siz echo video va kadr ketma-ketligi bo‘yicha yordamchisiz. "
        "Kadrlar ketma-ketligini harakat sifatida ko‘ring; still bo‘lsa shunday yozing. "
        "Faqat berilgan faktlar asosida 5–8 jumlalik ehtiyotkor klinik izoh yozing. "
        "Kirish matnini qayta ko‘chirmang. Ichki reja yozmang. "
        "Yo‘q dalilni o‘ylab topmang. Yangi o‘lchov va kasallik nomi qo‘shmang. "
        "O‘zbek yoki ingliz. Yakuniy qaror shifokorga tegishli."
    )

    raundlar: List[Dict[str, Any]] = []
    med = ""
    qwen = ""
    med_manba = "shablon"
    qwen_manba = "shablon"
    d_matn = ""
    d_manba = "shablon"
    konsensus = False
    konsensus_sabab = ""
    chegara = max(1, max_raund)

    for t in range(1, chegara + 1):
        kirish = raund_kirish_turi(t)
        bemor_qisqa, oraliq, oldingi_d = _raund_qismlari(
            kirish, xom_i, z_qosh, med_i, med_z, matritsa, d_matn
        )
        # Q va P bir-birini ko‘rmaydi: ikkalasi ham shu raundning D/I/Z ini oladi
        qwen, qwen_manba = _rol_javobi(
            "qwen_vl",
            qwen_tizim,
            bemor_qisqa,
            oraliq,
            oldingi_d,
            _raund_rasmlari("qwen_vl", kirish, matritsa, viz),
            yoq,
            video=True,
        )
        med, med_manba = _rol_javobi(
            "medgemma",
            med_tizim,
            bemor_qisqa,
            oraliq,
            oldingi_d,
            _raund_rasmlari("medgemma", kirish, matritsa, viz),
            yoq,
            video=False,
        )
        qwen_agree, med_agree, kelish_sabab = (False, False, "")
        if t >= 3 and oldingi_d:
            qwen_agree, med_agree, kelish_sabab = _oldingi_d_bilan_kelishuv(qwen, med, oldingi_d)
        d_matn, d_manba = _deepseek_d(qwen, med)
        kelishdi = bool(qwen_agree and med_agree)
        if kelishdi:
            konsensus = True
            konsensus_sabab = kelish_sabab or "Qwen va MedGemma oldingi D bilan kelishdi."
        elif t >= chegara:
            konsensus = False
            konsensus_sabab = kelish_sabab or f"t=T ({chegara}), to‘liq kelishuv yo‘q."
        raundlar.append(
            {
                "raund": str(t),
                "kirish": kirish,
                "medgemma": med,
                "qwen": qwen,
                "deepseek": d_matn,
                "medgemma_manba": med_manba,
                "qwen_manba": qwen_manba,
                "deepseek_manba": d_manba,
                "qwen_agree": qwen_agree,
                "medgemma_agree": med_agree,
                "konsensus": kelishdi,
                "konsensus_sabab": kelish_sabab,
            }
        )
        if toxtash_sharti(t, chegara, qwen_agree, med_agree):
            break

    model = model_qisqacha()
    return {
        "ok": True,
        "raundlar": raundlar,
        "umumlashtirish": (d_matn or "").strip(),
        "raund_soni": len(raundlar),
        "konsensus": konsensus,
        "konsensus_sabab": konsensus_sabab,
        "medgemma_manba": med_manba,
        "qwen_manba": qwen_manba,
        "deepseek_manba": d_manba,
        "medgemma_model": model.get("medgemma") if med_manba != "shablon" else "shablon",
        "qwen_model": model.get("qwen_vl") if qwen_manba != "shablon" else "shablon",
        "deepseek_model": model.get("chat") if d_manba != "shablon" else "shablon",
        "still_id": viz.get("still_id") or [],
        "video_id": viz.get("video_id") or [],
        "yoq_dalillar": yoq,
        "xom_i": xom_i,
        "oraliq_z": z_qosh[:2000],
        "xabar": "MDT yakunlandi. Shifokor o‘rnini bosmaydi.",
    }
