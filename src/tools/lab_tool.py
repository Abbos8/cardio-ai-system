"""Lab technician: CSV/PDF/forma qiymatlari va dorilarni barqaror matn/token formatiga keltiradi.

Natija klinik tashxis o‘rnini bosmaydi.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Any, Dict, List, Optional, Tuple

# Ko‘rsatkich → (birlik, qisqa klinik izoh — tashxis emas)
LAB_IZOH: Dict[str, Tuple[str, str]] = {
    "troponin_i": ("ng/L", "yurak shikastlanishi biomarkeri"),
    "troponin": ("ng/L", "yurak shikastlanishi biomarkeri"),
    "nt_probnp": ("pg/mL", "yurak yetishmovchiligi markeri"),
    "bnp": ("pg/mL", "natriuretik peptid"),
    "kreatinin": ("µmol/L", "buyrak funksiyasi"),
    "kaliy": ("mmol/L", "elektrolit, aritmiya xavfi bilan bog‘liq"),
    "natriy": ("mmol/L", "elektrolit"),
    "glyukoza": ("mmol/L", "qon shakari"),
    "ldl": ("mmol/L", "lipid profili"),
    "hdl": ("mmol/L", "lipid profili"),
    "hba1c": ("%", "uzoq muddatli glyukoza"),
    "gemoglobin": ("g/L", "qon"),
    "crp": ("mg/L", "yallig‘lanish"),
}

# Hisobotdagi nomlar → ichki kalit
LAB_SINONIM: Dict[str, str] = {
    "troponin_i": "troponin_i",
    "troponini": "troponin_i",
    "troponin i": "troponin_i",
    "hs troponin i": "troponin_i",
    "hstni": "troponin_i",
    "tni": "troponin_i",
    "tn i": "troponin_i",
    "troponin": "troponin",
    "nt_probnp": "nt_probnp",
    "nt-probnp": "nt_probnp",
    "ntprobnp": "nt_probnp",
    "n-terminal pro-bnp": "nt_probnp",
    "bnp": "bnp",
    "kreatinin": "kreatinin",
    "creatinine": "kreatinin",
    "creat": "kreatinin",
    "kaliy": "kaliy",
    "potassium": "kaliy",
    "k+": "kaliy",
    "k": "kaliy",
    "natriy": "natriy",
    "sodium": "natriy",
    "na+": "natriy",
    "na": "natriy",
    "glyukoza": "glyukoza",
    "glucose": "glyukoza",
    "ldl": "ldl",
    "hdl": "hdl",
    "hba1c": "hba1c",
    "hb a1c": "hba1c",
    "gemoglobin": "gemoglobin",
    "hemoglobin": "gemoglobin",
    "hb": "gemoglobin",
    "crp": "crp",
}


def _nomni_tozala(nom: str) -> str:
    """Analit nomini sinonim lug‘atiga mos kalitga keltiradi.

    Args:
        nom: CSV/PDF/forma sarlavhasi.

    Returns:
        Ichki kalit yoki soddalashtirilgan satr.
    """
    s = re.sub(r"\s+", " ", str(nom).strip().lower())
    s = s.replace("_", " ")
    s = re.sub(r"[\(\)\[\]:,]", "", s)
    s = s.replace("μ", "u").replace("µ", "u")
    if s in LAB_SINONIM:
        return LAB_SINONIM[s]
    s2 = s.replace(" ", "_")
    if s2 in LAB_SINONIM:
        return LAB_SINONIM[s2]
    if s2 in LAB_IZOH:
        return s2
    return s2 or s


def _son(qiymat: Any) -> Optional[float]:
    """Laboratoriya qiymatini float ga o‘giradi.

    Args:
        qiymat: Forma, CSV yoki matndagi son.

    Returns:
        Float yoki None.
    """
    if qiymat is None or qiymat == "":
        return None
    if isinstance(qiymat, (int, float)) and not isinstance(qiymat, bool):
        return float(qiymat)
    matn = str(qiymat).strip().replace(",", ".")
    matn = re.sub(r"[^0-9.\-eE]", "", matn)
    if matn in {"", ".", "-", "-."}:
        return None
    try:
        return float(matn)
    except ValueError:
        return None


def _tokenlar(matn: str) -> List[str]:
    """Matndan qisqa tibbiy tokenlar ajratadi.

    Args:
        matn: Lab yoki dori satri.

    Returns:
        Kichik harfli tokenlar. RAG/fellow so‘rovi uchun.
    """
    qismlar = re.findall(r"[A-Za-zА-Яа-яЁёўғҳқ0-9_\-]+", matn.lower())
    return [t for t in qismlar if len(t) > 1]


def csv_dan_lab(bayt: bytes) -> Dict[str, float]:
    """CSV dan analit→qiymat lug‘atini o‘qiydi.

    Args:
        bayt: Fayl tarkibi. Ustunlar: name/analyte/test va value/result, yoki 2 ustun.

    Returns:
        Ichki kalit → son. Tashxis emas.
    """
    try:
        matn = bayt.decode("utf-8-sig")
    except UnicodeDecodeError:
        matn = bayt.decode("latin-1", errors="replace")
    qatorlar = list(csv.reader(io.StringIO(matn)))
    qatorlar = [[c.strip() for c in q] for q in qatorlar if any(x.strip() for x in q)]
    if not qatorlar:
        return {}
    natija: Dict[str, float] = {}

    def _qosh(nom: str, qiymat: Any) -> None:
        son = _son(qiymat)
        if son is None:
            return
        natija[_nomni_tozala(nom)] = son

    bosh = [c.lower() for c in qatorlar[0]]
    nom_idx = next(
        (i for i, c in enumerate(bosh) if c in {"name", "analyte", "test", "parameter", "ko‘rsatkich", "nom"}),
        None,
    )
    qiymat_idx = next(
        (i for i, c in enumerate(bosh) if c in {"value", "result", "qiymat", "natija"}),
        None,
    )
    if nom_idx is not None and qiymat_idx is not None:
        for qator in qatorlar[1:]:
            if max(nom_idx, qiymat_idx) >= len(qator):
                continue
            _qosh(qator[nom_idx], qator[qiymat_idx])
        return natija
    if len(qatorlar[0]) >= 2 and _son(qatorlar[0][1]) is None:
        for qator in qatorlar[1:]:
            if len(qator) < 2:
                continue
            _qosh(qator[0], qator[1])
        return natija
    for qator in qatorlar:
        if len(qator) >= 2:
            _qosh(qator[0], qator[1])
        elif len(qator) == 1 and "=" in qator[0]:
            chap, ong = qator[0].split("=", 1)
            _qosh(chap, ong)
    return natija


def matndan_lab(matn: str) -> Dict[str, float]:
    """Erkin matn/PDF matnidan ma’lum analitlar yonidagi sonlarni qidiradi.

    Args:
        matn: Hisobot matni.

    Returns:
        Topilgan qiymatlar. To‘liq panel emas.
    """
    natija: Dict[str, float] = {}
    past = matn.lower().replace("\u00a0", " ")
    andozalar = [
        (r"hs[\s\-]?troponin\s*i", "troponin_i"),
        (r"troponin\s*i", "troponin_i"),
        (r"nt[\s\-]?pro[\s\-]?bnp", "nt_probnp"),
        (r"\bbnp\b", "bnp"),
        (r"creatinine|kreatinin", "kreatinin"),
        (r"potassium|\bkaliy\b|\bk\+", "kaliy"),
        (r"sodium|\bnatriy\b|\bna\+", "natriy"),
        (r"glucose|\bglyukoza\b", "glyukoza"),
        (r"\bldl\b", "ldl"),
        (r"\bhdl\b", "hdl"),
        (r"hba1c|hb\s*a1c", "hba1c"),
        (r"hemoglobin|gemoglobin", "gemoglobin"),
        (r"\bcrp\b", "crp"),
        (r"\btroponin\b", "troponin"),
    ]
    for andoza, kalit in andozalar:
        if kalit in natija:
            continue
        m = re.search(andoza + r"[^\d]{0,40}([0-9]+(?:[.,][0-9]+)?)", past, re.I)
        if not m:
            m = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*(?:ng/l|pg/ml|mmol|umol|µmol)[^\n]{0,20}" + andoza, past, re.I)
        if m:
            son = _son(m.group(1))
            if son is not None:
                natija[kalit] = son
    return natija


def pdf_dan_lab(bayt: bytes) -> Tuple[Dict[str, float], str]:
    """PDF baytidan matn olib analitlarni ajratadi.

    Args:
        bayt: PDF fayl.

    Returns:
        (qiymatlar, xabar). pypdf yo‘q bo‘lsa bo‘sh lug‘at.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return {}, "pypdf o‘rnatilmagan. `pip install pypdf` yoki CSV yuklang."
    try:
        oquvchi = PdfReader(io.BytesIO(bayt))
        matn = "\n".join((sahifa.extract_text() or "") for sahifa in oquvchi.pages)
    except Exception as exc:
        return {}, f"PDF o‘qilmadi: {exc}"
    if not matn.strip():
        return {}, "PDF dan matn chiqmadi (skan bo‘lishi mumkin)."
    return matndan_lab(matn), ""


def fayldan_lab(bayt: bytes, nom: str) -> Tuple[Dict[str, float], str]:
    """CSV, TXT yoki PDF dan laboratoriya qiymatlarini o‘qiydi.

    Args:
        bayt: Fayl tarkibi.
        nom: Fayl nomi (kengaytma uchun).

    Returns:
        (qiymatlar, xabar).
    """
    past = (nom or "").lower()
    if past.endswith(".pdf"):
        qiymat, xato = pdf_dan_lab(bayt)
        if xato:
            return qiymat, xato
        return qiymat, f"PDF dan {len(qiymat)} ta ko‘rsatkich."
    if past.endswith(".csv"):
        qiymat = csv_dan_lab(bayt)
        return qiymat, f"CSV dan {len(qiymat)} ta ko‘rsatkich."
    try:
        matn = bayt.decode("utf-8-sig")
    except UnicodeDecodeError:
        matn = bayt.decode("latin-1", errors="replace")
    if "," in matn.split("\n", 1)[0] or ";" in matn.split("\n", 1)[0]:
        qiymat = csv_dan_lab(bayt)
        if qiymat:
            return qiymat, f"Matn/CSV dan {len(qiymat)} ta ko‘rsatkich."
    qiymat = matndan_lab(matn)
    return qiymat, f"Matndan {len(qiymat)} ta ko‘rsatkich."


def dorilarni_ajrat(xom: Any) -> List[Dict[str, str]]:
    """Dori tarixini tuzilgan yozuvlarga ajratadi (nom, doza, chastota).

    Args:
        xom: Sat r, satrlar ro‘yxati yoki {nom, doza, chastota} lug‘atlari.
             Qator: `nom | doza | chastota` yoki `aspirin 75 mg od`.

    Returns:
        Tuzilgan ro‘yxat. Bo‘sh maydonlar '' .
    """
    yozuvlar: List[Dict[str, str]] = []
    if xom is None or xom == "":
        return yozuvlar
    if isinstance(xom, dict):
        xom = [xom]
    if isinstance(xom, list):
        qatorlar: List[Any] = xom
    else:
        qatorlar = re.split(r"[\n;]+", str(xom))
    for element in qatorlar:
        if isinstance(element, dict):
            nom = str(element.get("nom") or element.get("name") or "").strip()
            if not nom:
                continue
            yozuvlar.append(
                {
                    "nom": nom,
                    "doza": str(element.get("doza") or element.get("dose") or "").strip(),
                    "chastota": str(element.get("chastota") or element.get("frequency") or "").strip(),
                }
            )
            continue
        satr = str(element).strip()
        if not satr:
            continue
        if "|" in satr:
            qism = [p.strip() for p in satr.split("|")]
            yozuvlar.append(
                {
                    "nom": qism[0],
                    "doza": qism[1] if len(qism) > 1 else "",
                    "chastota": qism[2] if len(qism) > 2 else "",
                }
            )
            continue
        m = re.match(
            r"^([A-Za-zА-Яа-яЁёўғҳқ0-9\-\+]+(?:\s+[A-Za-zА-Яа-яЁёўғҳқ0-9\-\+]+){0,3})\s+"
            r"([0-9]+(?:[.,][0-9]+)?\s*(?:mg|mcg|g|iu|ml)?)\s*(.*)$",
            satr,
            re.I,
        )
        if m:
            yozuvlar.append(
                {"nom": m.group(1).strip(), "doza": m.group(2).strip(), "chastota": m.group(3).strip()}
            )
        else:
            yozuvlar.append({"nom": satr, "doza": "", "chastota": ""})
    return yozuvlar


def _dori_satr(dori: Dict[str, str]) -> str:
    """Bitta dorini RAG/fellow uchun qisqa satrga yozadi.

    Args:
        dori: nom, doza, chastota.

    Returns:
        `nom|doza|chastota` (bo‘sh qismlar tashlanadi).
    """
    qism = [dori.get("nom") or ""]
    if dori.get("doza"):
        qism.append(dori["doza"])
    if dori.get("chastota"):
        qism.append(dori["chastota"])
    return "|".join(qism)


def process_lab(bemor: Dict[str, Any]) -> Dict[str, Any]:
    """Forma, CSV/PDF va dorilarni barqaror lab natijasiga yig‘adi.

    Args:
        bemor: laboratoriya lug‘ati, lab_fayl_bayt/lab_fayl_nomi, dorilar.

    Returns:
        ok, matn, rag_satr, tokenlar, qiymatlar, dorilar. Tashxis qo‘yilmaydi.
    """
    lab: Dict[str, float] = {}
    forma = bemor.get("laboratoriya") or bemor.get("lab") or {}
    if isinstance(forma, dict):
        for nom, xom in forma.items():
            son = _son(xom)
            if son is not None:
                lab[_nomni_tozala(str(nom))] = son

    fayl_xabar = ""
    bayt = bemor.get("lab_fayl_bayt")
    nom = str(bemor.get("lab_fayl_nomi") or "lab.csv")
    if isinstance(bayt, bytes) and bayt:
        fayldan, fayl_xabar = fayldan_lab(bayt, nom)
        lab.update(fayldan)

    qatorlar: List[str] = []
    tokenlar: List[str] = ["laboratory"]
    for kalit in sorted(lab.keys()):
        son = lab[kalit]
        birlik, izoh = LAB_IZOH.get(kalit, ("", "laboratoriya ko‘rsatkichi"))
        qatorlar.append(f"{kalit}={son} {birlik} ({izoh})".strip())
        tokenlar.extend(_tokenlar(kalit))

    dorilar = dorilarni_ajrat(
        bemor.get("dorilar_tuzilgan") or bemor.get("dorilar") or bemor.get("meds") or bemor.get("medication_history")
    )
    if dorilar:
        dori_qism = "; ".join(_dori_satr(d) for d in dorilar)
        qatorlar.append("MEDS " + dori_qism)
        for d in dorilar:
            tokenlar.extend(_tokenlar(_dori_satr(d)))

    if not qatorlar:
        return {
            "ok": False,
            "xabar": fayl_xabar or "Laboratoriya qiymatlari kiritilmagan.",
            "matn": "",
            "rag_satr": "",
            "tokenlar": [],
            "qiymatlar": {},
            "dorilar": [],
        }

    lab_qism = " ".join(f"{k}={v}" for k, v in sorted(lab.items()))
    meds_qism = "; ".join(_dori_satr(d) for d in dorilar)
    rag_satr = "LAB " + lab_qism
    if meds_qism:
        rag_satr += " | MEDS " + meds_qism
    matn = "Lab technician: " + "; ".join(qatorlar)
    xabar = "Laboratoriya va dori tarixi matn/token ko‘rinishiga keltirildi. Tashxis emas."
    if fayl_xabar:
        xabar = fayl_xabar + " " + xabar
    return {
        "ok": True,
        "xabar": xabar,
        "matn": matn,
        "rag_satr": rag_satr,
        "tokenlar": sorted(set(tokenlar)),
        "qiymatlar": lab,
        "dorilar": dorilar,
    }
