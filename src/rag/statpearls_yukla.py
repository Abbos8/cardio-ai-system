"""NCBI StatPearls kardiologiya maqolalarini data/raw/statpearls/ ga yuklaydi.

Rasmiy E-utilities (PubMed kitob yozuvi) sarlavha, muallif, xulosa va
bo‘lim nomlarini beradi. Bookshelf HTML ochiq qaytsa, bo‘lim matni qo‘shiladi.
So‘rovlar orasida User-Agent va time.sleep(1) — NCBI limitiga rioya.
Matn gitga kiritilmaydi. Tashxis o‘rnini bosmaydi.
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

LOYIHA = Path(__file__).resolve().parents[2]
STATPEARLS_DIR = LOYIHA / "data" / "raw" / "statpearls"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
USER_AGENT = (
    "CardioAIResearchBot/0.1 (local RAG ingest; educational CDS; not for redistribution)"
)

# 20 ta asosiy patologiya. Har birida StatPearls dagi aniq sarlavha variantlari.
MAVZULAR = [
    {"id": "stemi", "sarlavhalar": ["Acute ST-Segment Elevation Myocardial Infarction (STEMI)", "STEMI"]},
    {"id": "nstemi", "sarlavhalar": ["Non-ST Segment Elevation Myocardial Infarction", "Non ST Segment Elevation Myocardial Infarction"]},
    {"id": "unstable_angina", "sarlavhalar": ["Unstable Angina"]},
    {"id": "heart_failure", "sarlavhalar": ["Congestive Heart Failure", "Heart Failure"]},
    {"id": "atrial_fibrillation", "sarlavhalar": ["Atrial Fibrillation"]},
    {"id": "lbbb", "sarlavhalar": ["Left Bundle Branch Block"]},
    {"id": "rbbb", "sarlavhalar": ["Right Bundle Branch Block"]},
    {"id": "cardiogenic_shock", "sarlavhalar": ["Cardiogenic Shock"]},
    {"id": "aortic_stenosis", "sarlavhalar": ["Aortic Stenosis"]},
    {"id": "aortic_regurgitation", "sarlavhalar": ["Aortic Regurgitation"]},
    {"id": "mitral_regurgitation", "sarlavhalar": ["Mitral Regurgitation"]},
    {"id": "mitral_stenosis", "sarlavhalar": ["Mitral Stenosis"]},
    {"id": "ventricular_tachycardia", "sarlavhalar": ["Ventricular Tachycardia"]},
    {"id": "atrial_flutter", "sarlavhalar": ["Atrial Flutter"]},
    {"id": "complete_heart_block", "sarlavhalar": ["Third-Degree Atrioventricular Block", "Complete Heart Block"]},
    {"id": "acute_pericarditis", "sarlavhalar": ["Pericarditis", "Acute Pericarditis"]},
    {"id": "myocarditis", "sarlavhalar": ["Myocarditis"]},
    {"id": "infective_endocarditis", "sarlavhalar": ["Infective Endocarditis"]},
    {"id": "aortic_dissection", "sarlavhalar": ["Aortic Dissection"]},
    {"id": "hypertrophic_cardiomyopathy", "sarlavhalar": ["Hypertrophic Cardiomyopathy"]},
]


def _sorov(url: str, timeout: int = 45) -> tuple:
    """Bitta NCBI URL ni User-Agent bilan o‘qiydi.

    Args:
        url: E-utilities yoki Bookshelf manzili.
        timeout: Sekund.

    Returns:
        (ok, matn). Xatoda (False, xabar).
    """
    talab = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json,application/xml,text/html,*/*",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(talab, timeout=timeout) as javob:
            xom = javob.read()
        return True, xom.decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return False, str(exc)


def _element_matn(el: Optional[ET.Element]) -> str:
    """XML element ichidagi barcha matn tugunlarini bitta qatorga yig‘adi.

    Args:
        el: Abstract yoki sarlavha. None bo‘lishi mumkin.

    Returns:
        Tegsiz, siqilgan matn.
    """
    if el is None:
        return ""
    return " ".join("".join(el.itertext()).split())


def qidir_pmid(sarlavha: str) -> List[str]:
    """StatPearls kitobida sarlavha bo‘yicha PubMed identifikatorlarini qidiradi.

    Args:
        sarlavha: Maqola sarlavhasi (inglizcha).

    Returns:
        PMID ro‘yxati. Topilmasa bo‘sh.
    """
    import json

    atama = f'"{sarlavha}"[Title] AND StatPearls[book]'
    qs = urllib.parse.urlencode(
        {
            "db": "pubmed",
            "term": atama,
            "retmode": "json",
            "retmax": "8",
            "tool": "CardioAIResearchBot",
        }
    )
    ok, matn = _sorov(f"{EUTILS}/esearch.fcgi?{qs}")
    if not ok:
        return []
    try:
        data = json.loads(matn)
    except json.JSONDecodeError:
        return []
    return list((data.get("esearchresult") or {}).get("idlist") or [])


def sarlavha_balli(sorov: str, sarlavha: str) -> int:
    """Asosiy StatPearls maqolasini Nursing/Archived variantdan ajratadi.

    Args:
        sorov: Qidirilgan patologiya sarlavhasi.
        sarlavha: PubMed dagi maqola sarlavhasi.

    Returns:
        Yuqori ball — shu patologiya uchun asosiy maqola.
    """
    a = sorov.casefold().strip(" .")
    b = sarlavha.casefold().strip(" .")
    if not a or not b:
        return 0
    jarima = 0
    if "(nursing)" in b:
        jarima -= 50
    if "(archived)" in b:
        jarima -= 30
    if b.startswith(("surgical ", "ems ", "catheter-based")) or "ablation" in b:
        jarima -= 40
    if b == a:
        asos = 100
    elif b.startswith(a + " ") or b.startswith(a + "("):
        asos = 90
    elif a in b:
        asos = 60
    else:
        return 0
    return asos + jarima


def sarlavhalarni_ol(pmidlar: List[str]) -> Dict[str, str]:
    """PubMed esummary dan PMID → sarlavha lug‘atini oladi.

    Args:
        pmidlar: esearch natijasi.

    Returns:
        Bo‘sh bo‘lishi mumkin, agar API javob bermasa.
    """
    import json

    if not pmidlar:
        return {}
    qs = urllib.parse.urlencode(
        {
            "db": "pubmed",
            "id": ",".join(pmidlar),
            "retmode": "json",
            "tool": "CardioAIResearchBot",
        }
    )
    ok, matn = _sorov(f"{EUTILS}/esummary.fcgi?{qs}")
    if not ok:
        return {}
    try:
        data = json.loads(matn)
    except json.JSONDecodeError:
        return {}
    natija = (data.get("result") or {})
    lugat: Dict[str, str] = {}
    for pmid in natija.get("uids") or pmidlar:
        item = natija.get(str(pmid)) or {}
        if isinstance(item, dict) and item.get("title"):
            lugat[str(pmid)] = str(item["title"]).rstrip(".")
    return lugat


def xml_dan_yozuvlar(xml_matn: str) -> Dict[str, Dict]:
    """PubMed kitob XML idan PMID bo‘yicha maqola yozuvlarini ajratadi.

    Args:
        xml_matn: efetch natijasi.

    Returns:
        pmid → sarlavha, nbk, mualliflar, yil, xulosa, bolimlar, copyright.
    """
    try:
        ildiz = ET.fromstring(xml_matn)
    except ET.ParseError:
        return {}
    natija: Dict[str, Dict] = {}
    for hujjat in ildiz.findall(".//BookDocument"):
        pmid = _element_matn(hujjat.find("PMID"))
        if not pmid:
            continue
        nbk = ""
        for aid in hujjat.findall(".//ArticleId"):
            if (aid.get("IdType") or "") == "bookaccession":
                nbk = (aid.text or "").strip()
        mualliflar = []
        for muallif in hujjat.findall("./AuthorList/Author"):
            ism = " ".join(
                qism
                for qism in (
                    _element_matn(muallif.find("ForeName")),
                    _element_matn(muallif.find("LastName")),
                )
                if qism
            )
            if ism:
                mualliflar.append(ism)
        yil = _element_matn(hujjat.find("./Book/PubDate/Year"))
        bolimlar = []
        for bolim in hujjat.findall(".//Section/SectionTitle"):
            nom = _element_matn(bolim)
            if nom and nom not in bolimlar:
                bolimlar.append(nom)
        natija[pmid] = {
            "pmid": pmid,
            "sarlavha": _element_matn(hujjat.find("ArticleTitle")) or pmid,
            "nbk": nbk,
            "mualliflar": mualliflar,
            "yil": yil,
            "xulosa": _element_matn(hujjat.find(".//AbstractText")),
            "bolimlar": bolimlar,
            "copyright": _element_matn(hujjat.find(".//CopyrightInformation")),
        }
    return natija


def yozuvdan_markdown(yozuv: Dict, qoshimcha: str = "") -> str:
    """StatPearls yozuvini sarlavha va bo‘limli Markdown ga aylantiradi.

    Args:
        yozuv: xml_dan_yozuvlar qiymati.
        qoshimcha: Bookshelf dan olingan bo‘lim matni (bo‘lsa).

    Returns:
        .md fayl matni. Lokal RAG uchun; tarqatish emas.
    """
    nbk = yozuv.get("nbk") or ""
    url = f"https://www.ncbi.nlm.nih.gov/books/{nbk}" if nbk else ""
    qatorlar = [
        f"# {yozuv.get('sarlavha') or 'StatPearls'}",
        "",
        f"Manba: NCBI Bookshelf / StatPearls ({nbk or 'NBK yo‘q'})",
        f"URL: {url}",
        f"PMID: {yozuv.get('pmid') or ''}",
        f"Mualliflar: {', '.join(yozuv.get('mualliflar') or [])}",
        f"Yil: {yozuv.get('yil') or ''}",
        "",
        "Lokal CardiacRAG indeksi uchun. Tarqatilmaydi. Tashxis o‘rnini bosmaydi.",
        yozuv.get("copyright") or "Copyright StatPearls Publishing LLC.",
        "",
        "## Xulosa",
        "",
        yozuv.get("xulosa") or "(xulosa yo‘q)",
        "",
        "## Bo‘limlar",
        "",
    ]
    for bolim in yozuv.get("bolimlar") or []:
        qatorlar.append(f"- {bolim}")
    if qoshimcha.strip():
        qatorlar.extend(["", "## To‘liq matn", "", qoshimcha.strip()])
    return "\n".join(qatorlar).strip() + "\n"


def bookshelf_matn(nbk: str) -> str:
    """Bookshelf HTML dan bo‘lim matnini olishga urinadi.

    Args:
        nbk: Masalan NBK482255.

    Returns:
        Toza bo‘limlar yoki bo‘sh satr (captcha / blok).
    """
    if not nbk:
        return ""
    ok, html = _sorov(f"https://www.ncbi.nlm.nih.gov/books/{nbk}/")
    if not ok:
        return ""
    past = html.lower()
    if "recaptcha" in past or "challengepage" in past or len(html) < 1500:
        return ""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return ""
    shorba = BeautifulSoup(html, "html.parser")
    for teg in shorba(["script", "style", "nav", "footer", "header", "form"]):
        teg.decompose()
    asos = shorba.find(id="maincontent") or shorba.find("article") or shorba.body
    if asos is None:
        return ""
    qatorlar: List[str] = []
    for sarlavha in asos.find_all(["h2", "h3"]):
        nom = " ".join(sarlavha.get_text(" ", strip=True).split())
        if not nom:
            continue
        if nom.lower() in {"references", "review questions"}:
            break
        qatorlar.append(f"## {nom}")
        for aka in sarlavha.find_next_siblings():
            if getattr(aka, "name", None) in {"h2", "h3"}:
                break
            matn = " ".join(aka.get_text(" ", strip=True).split())
            if matn:
                qatorlar.append(matn)
    return "\n\n".join(qatorlar).strip()


def _efetch(pmidlar: List[str]) -> str:
    """Tanlangan PMID larni bitta PubMed XML so‘rovida oladi.

    Args:
        pmidlar: StatPearls kitob maqolalari.

    Returns:
        XML matn yoki bo‘sh satr.
    """
    if not pmidlar:
        return ""
    qs = urllib.parse.urlencode(
        {
            "db": "pubmed",
            "id": ",".join(pmidlar),
            "retmode": "xml",
            "tool": "CardioAIResearchBot",
        }
    )
    ok, matn = _sorov(f"{EUTILS}/efetch.fcgi?{qs}", timeout=90)
    return matn if ok else ""


def barcha_statpearls(kutish: float = 1.0) -> Dict[str, object]:
    """20 ta kardiologik StatPearls maqolasini data/raw/statpearls/ ga yozadi.

    Args:
        kutish: NCBI so‘rovlari orasidagi pauza (sekund). Kamida 1.

    Returns:
        ok, bor, xato va xabarlar. Tashxis emas.
    """
    pauza = max(1.0, float(kutish))
    STATPEARLS_DIR.mkdir(parents=True, exist_ok=True)
    hisob: Dict[str, int] = {"ok": 0, "bor": 0, "xato": 0}
    xabarlar: List[str] = []
    tanlov: List[Dict] = []
    for mavzu in MAVZULAR:
        yol = STATPEARLS_DIR / f"{mavzu['id']}.md"
        if yol.exists() and yol.stat().st_size > 400:
            hisob["bor"] += 1
            continue
        eng_yaxshi: Optional[tuple] = None
        for sarlavha in mavzu["sarlavhalar"]:
            ids = qidir_pmid(sarlavha)
            time.sleep(pauza)
            if not ids:
                continue
            lugat = sarlavhalarni_ol(ids)
            time.sleep(pauza)
            for pmid, nom in lugat.items():
                ball = sarlavha_balli(sarlavha, nom)
                if eng_yaxshi is None or ball > eng_yaxshi[0]:
                    eng_yaxshi = (ball, pmid, nom)
        if eng_yaxshi is None or eng_yaxshi[0] < 40:
            hisob["xato"] += 1
            xabarlar.append(f"{mavzu['id']}: StatPearls topilmadi")
            continue
        tanlov.append({"id": mavzu["id"], "pmid": eng_yaxshi[1], "yol": yol})
    xml = _efetch([q["pmid"] for q in tanlov])
    if tanlov:
        time.sleep(pauza)
    yozuvlar = xml_dan_yozuvlar(xml) if xml else {}
    bookshelf_ochiq: Optional[bool] = None
    for qator in tanlov:
        yozuv = yozuvlar.get(qator["pmid"])
        if not yozuv or len(yozuv.get("xulosa") or "") < 80:
            hisob["xato"] += 1
            xabarlar.append(f"{qator['id']}: XML yozuvi yo‘q ({qator['pmid']})")
            continue
        qoshimcha = ""
        if bookshelf_ochiq is not False and yozuv.get("nbk"):
            qoshimcha = bookshelf_matn(str(yozuv["nbk"]))
            time.sleep(pauza)
            if bookshelf_ochiq is None:
                bookshelf_ochiq = bool(qoshimcha)
        qator["yol"].write_text(yozuvdan_markdown(yozuv, qoshimcha), encoding="utf-8")
        meta = qator["yol"].with_suffix(".url.txt")
        nbk = yozuv.get("nbk") or ""
        meta.write_text(
            f"https://www.ncbi.nlm.nih.gov/books/{nbk}\nPMID:{yozuv.get('pmid')}\n",
            encoding="utf-8",
        )
        hisob["ok"] += 1
    if bookshelf_ochiq is False:
        xabarlar.append("Bookshelf HTML bloklandi; E-utilities xulosasi saqlandi")
    natija: Dict[str, object] = dict(hisob)
    natija["xabarlar"] = xabarlar
    return natija


if __name__ == "__main__":
    natija = barcha_statpearls()
    print(
        f"StatPearls: yangi={natija.get('ok')} bor={natija.get('bor')} xato={natija.get('xato')}"
    )
    for xabar in natija.get("xabarlar") or []:
        print("  xato:", xabar)
