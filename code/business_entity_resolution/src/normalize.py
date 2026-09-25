"""Language-aware, country-agnostic normalisation of business names and addresses.

Nothing here branches on the ``country`` value: every rule (US, Indian and French
abbreviations, postcode regexes, Indic transliteration) is applied to every record.

Main entry point: :func:`normalize_frame` which adds the derived fields
``name_norm, name_core, name_legal, name_skel, addr_norm, addr_no_landmark,
addr_numbers, postcode, city`` to a raw source DataFrame.

All dictionaries below are hand-written from general linguistic knowledge
(no external data).
"""
from __future__ import annotations

import re
import unicodedata
from multiprocessing import Pool

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Indic script -> Latin transliteration.
# All major Indic Unicode blocks share the Devanagari layout (ISCII heritage), so we
# first map a code point of Bengali/Gurmukhi/Gujarati/Oriya/Tamil/Telugu/Kannada/
# Malayalam to its Devanagari equivalent by offset, then use one Devanagari table.
# --------------------------------------------------------------------------------------
_INDIC_BLOCKS = [0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]
_DEVA_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j",
    "झ": "jh", "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t",
    "थ": "th", "द": "d", "ध": "dh", "न": "n", "ऩ": "n", "प": "p", "फ": "ph", "ब": "b",
    "भ": "bh", "म": "m", "य": "y", "र": "r", "ऱ": "r", "ल": "l", "ळ": "l", "ऴ": "l",
    "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h", "क़": "k", "ख़": "kh", "ग़": "g",
    "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f", "य़": "y",
}
_DEVA_VOW = {
    "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri", "ए": "e",
    "ऐ": "ai", "ओ": "o", "औ": "au", "ऍ": "e", "ऎ": "e", "ऑ": "o", "ऒ": "o",
}
_DEVA_SIGN = {
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "े": "e", "ै": "ai",
    "ो": "o", "ौ": "au", "ॅ": "e", "ॆ": "e", "ॉ": "o", "ॊ": "o",
}
_DEVA_MISC = {"ं": "n", "ँ": "n", "ः": "h", "ऽ": "", "़": "", "।": " ", "॥": " "}
_VIRAMA = "्"
# Tamil has an "aytham" (ஃ) used for f/z sounds in loanwords (ஃப = f).
_EXTRA = {"ஃ": "", "‌": "", "‍": ""}
# Malayalam "chillu" letters live outside the aligned range: map to Devanagari cons+virama.
_CHILLU = {"ൺ": "ण्", "ൻ": "न्", "ർ": "र्", "ൽ": "ल्", "ൾ": "ळ्", "ൿ": "क्",
           "ൔ": "म्", "ൕ": "य्", "ൖ": "ळ्"}
_DEVA_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}


def _to_devanagari(ch: str) -> str:
    """Map a code point of any Indic block to its Devanagari counterpart."""
    cp = ord(ch)
    for base in _INDIC_BLOCKS:
        if base <= cp < base + 0x80:
            return chr(cp - base + 0x0900)
    return ch


def transliterate_indic(text: str) -> str:
    """Transliterate Indic-script text to a rough Latin form (schwa-deletion aware).

    Non-Indic characters are passed through unchanged.
    """
    if not any("ऀ" <= c <= "෿" for c in text):
        return text
    out = []
    text = "".join(_CHILLU.get(c, c) for c in text)
    chars = [_EXTRA.get(c, None) if c in _EXTRA else _to_devanagari(c) for c in text]
    chars = [c for c in chars if c]  # drop ZWJ/ZWNJ etc.
    n = len(chars)
    i = 0
    while i < n:
        c = chars[i]
        if c in _DEVA_CONS:
            out.append(_DEVA_CONS[c])
            nxt = chars[i + 1] if i + 1 < n else ""
            if nxt == "़":  # nukta: skip
                i += 1
                nxt = chars[i + 1] if i + 1 < n else ""
            if nxt == _VIRAMA:
                i += 2
                continue
            if nxt in _DEVA_SIGN:
                out.append(_DEVA_SIGN[nxt])
                i += 2
                continue
            # inherent schwa, dropped at end of word
            if nxt and (nxt in _DEVA_CONS or nxt in _DEVA_MISC):
                out.append("a")
            i += 1
            continue
        if c in _DEVA_VOW:
            out.append(_DEVA_VOW[c])
        elif c in _DEVA_SIGN:
            out.append(_DEVA_SIGN[c])
        elif c in _DEVA_MISC:
            out.append(_DEVA_MISC[c])
        elif c in _DEVA_DIGITS:
            out.append(_DEVA_DIGITS[c])
        elif c == _VIRAMA:
            pass
        elif "ऀ" <= c <= "ॿ":
            pass  # unknown Devanagari sign
        else:
            out.append(c)
        i += 1
    return "".join(out)


# --------------------------------------------------------------------------------------
# Canonicalisation dictionaries (general knowledge; applied to all countries)
# --------------------------------------------------------------------------------------
LEGAL_CANON = {
    # English / India
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "pvtltd": "pvt ltd",
    "limited": "ltd", "ltd": "ltd", "limitada": "ltd",
    "llp": "llp", "llc": "llc", "lllp": "llp", "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp", "co": "co", "company": "co", "cos": "co",
    "lp": "lp", "plc": "plc", "pllc": "llc", "pc": "pc", "ltda": "ltd", "opc": "opc",
    "public": "public",
    # France
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "scop": "scop", "selarl": "selarl", "societe": "societe", "ste": "societe",
    # transliterated Indic legal words (skeleton-ish spellings)
    "praivet": "pvt", "praivhet": "pvt", "praiveta": "pvt", "praivet.": "pvt", "prayvet": "pvt",
    "privet": "pvt", "praivat": "pvt", "praivetu": "pvt", "praivettu": "pvt", "pra": "pvt",
    "limited": "ltd", "limitet": "ltd", "limitad": "ltd", "limitedu": "ltd", "limitted": "ltd",
    "limitid": "ltd", "limitedd": "ltd", "li": "ltd", "limitedi": "ltd",
    "elelpi": "llp", "elalpi": "llp", "elelpii": "llp", "eleelsi": "llc", "elelsi": "llc",
    "kampani": "co", "kampni": "co", "kamp": "co",
}
# words that are removed from name_core in addition to legal forms (DBA markers)
NAME_DROP = {"the", "dba", "fka", "aka", "formerly", "known", "as", "doing", "business", "and", "of",
             "de", "du", "des", "la", "le", "les", "et", "d", "l"}
DBA_RE = re.compile(r"\b(?:d/?b/?a|f/?k/?a|a/?k/?a|formerly known as|doing business as|trading as|t/?a)\b")

STREET_CANON = {
    "street": "st", "st": "st", "str": "st", "road": "rd", "rd": "rd", "avenue": "ave", "ave": "ave",
    "av": "ave", "avn": "ave", "avenu": "ave", "boulevard": "blvd", "blvd": "blvd", "bd": "blvd",
    "bld": "blvd", "boul": "blvd", "drive": "dr", "dr": "dr", "lane": "ln", "ln": "ln", "court": "ct",
    "ct": "ct", "place": "pl", "pl": "pl", "plaza": "plz", "parkway": "pkwy", "pkwy": "pkwy",
    "highway": "hwy", "hwy": "hwy", "circle": "cir", "cir": "cir", "terrace": "ter", "ter": "ter",
    "trail": "trl", "trl": "trl", "square": "sq", "sq": "sq", "way": "way", "suite": "ste",
    "ste": "ste", "apartment": "apt", "apt": "apt", "unit": "unit", "floor": "fl", "fl": "fl",
    "flr": "fl", "building": "bldg", "bldg": "bldg", "north": "n", "south": "s", "east": "e",
    "west": "w", "mount": "mt", "mt": "mt", "saint": "st", "fort": "ft", "ft": "ft",
    "center": "ctr", "centre": "ctr", "ctr": "ctr", "expressway": "expy", "township": "twp",
    # India
    "nagar": "nagar", "ngr": "nagar", "marg": "marg", "mg": "mg", "sector": "sec", "sec": "sec",
    "number": "no", "no": "no", "num": "no", "house": "h", "hno": "h", "plot": "plot",
    "flat": "flat", "flatno": "flat", "cross": "cross", "main": "main", "colony": "colony",
    "layout": "layout", "district": "dist", "dist": "dist", "taluk": "tq", "tq": "tq",
    "village": "vill", "vill": "vill", "post": "po", "po": "po", "opposite": "opp", "opp": "opp",
    "near": "near", "nr": "near", "behind": "behind", "bh": "behind", "beside": "near",
    "ground": "gf", "gf": "gf", "first": "1", "second": "2", "third": "3", "fourth": "4",
    # France
    "rue": "rue", "r": "rue", "avenue.": "ave", "chemin": "chemin", "ch": "chemin", "che": "chemin",
    "allee": "allee", "all": "allee", "alle": "allee", "impasse": "imp", "imp": "imp",
    "route": "rte", "rte": "rte", "cours": "crs", "crs": "crs", "quai": "quai", "residence": "res",
    "res": "res", "lotissement": "lot", "lot": "lot", "cite": "cite", "square.": "sq",
    "faubourg": "fbg", "fbg": "fbg", "cedex": "cedex", "bis": "bis", "ter.": "ter",
}
LANDMARK_RE = re.compile(r"\b(?:near|nr|opp|opposite|behind|beside|next to|landmark)\b[^,]*")
NULL_TOKENS = {"null", "none", "na", "n/a", "nan", "nil", "unknown"}
TOKEN_RE = re.compile(r"[a-z0-9]+")
NUM_RE = re.compile(r"\d+")
POSTCODE_RE = re.compile(r"(?<!\d)(\d{6}|\d{3}\s\d{3}|\d{5})(?![\d/-])")
ORD_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th|er|e|eme|ème)\b")

# Region / state names: canonicalised to a short code so "Texas" == "TX" etc.
# (general knowledge; treated as generic "region" tokens, never used to branch logic)
REGION_CANON = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "telangana": "tg", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "jammu and kashmir": "jk",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    "hauts de france": "hdf", "nouvelle aquitaine": "naq", "pays de la loire": "pdl",
    "ile de france": "idf", "grand est": "ges", "normandie": "nor", "bretagne": "bre",
    "occitanie": "occ", "provence alpes cote d azur": "pac", "auvergne rhone alpes": "ara",
    "bourgogne franche comte": "bfc", "centre val de loire": "cvl", "corse": "cor",
}
_REGION_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, REGION_CANON), key=len, reverse=True)) + r")\b")


def basic_clean(s: str) -> str:
    """NFKD + accent strip + Indic transliteration + lowercase + punctuation unify."""
    if not s:
        return ""
    s = transliterate_indic(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("&", " and ").replace("+", " and ").replace("@", " ")
    s = s.replace("'", "").replace("’", "")
    return s


def collapse_initials(tokens: list[str]) -> list[str]:
    """Merge runs of single-letter tokens ("l l c" -> "llc", "s a s" -> "sas")."""
    out, run = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(t)
    if run:
        out.append("".join(run))
    return out


_SKEL_SUBS = [("ph", "f"), ("ch", "c"), ("sh", "s"), ("th", "t"), ("gh", "g"), ("kh", "k"),
              ("bh", "b"), ("dh", "d"), ("jh", "j"), ("ck", "k"), ("q", "k"), ("c", "k"),
              ("x", "ks"), ("z", "s"), ("w", "b"), ("v", "b"), ("y", "i"), ("g", "j"), ("f", "p")]
# consonant skeletons of (transliterated) legal words, e.g. "praivet" -> "prbt"
LEGAL_SKEL = {"prbt": "pvt", "prvt": "pvt", "lmtd": "ltd", "lmtt": "ltd", "lmt": "ltd", "lmtdd": "ltd",
              "elp": "llp", "elelp": "llp", "ells": "llc", "elels": "llc", "elsi": "llc"}
_VOWELS = set("aeiou")


def skeleton(tok: str) -> str:
    """Phonetic consonant skeleton of a Latin token.

    Aligns English spellings with rough transliterations
    (e.g. "technologies"/"teknolojij" -> "tknlg"/"tknlj").
    """
    if not tok or tok.isdigit():
        return tok
    t = tok
    for a, b in _SKEL_SUBS:
        t = t.replace(a, b)
    first = t[0]
    rest = [c for c in t[1:] if c not in _VOWELS and c != "h"]
    s = first + "".join(rest)
    # collapse doubles
    out = [s[0]]
    for c in s[1:]:
        if c != out[-1]:
            out.append(c)
    s = "".join(out)
    if len(s) > 3 and s.endswith("s"):
        s = s[:-1]
    return s


def normalize_name(raw: str) -> tuple[str, str, str, str]:
    """Return (name_norm, name_core, name_legal, name_skel) for a raw business name."""
    s = basic_clean(raw)
    s = DBA_RE.sub(" ", s)
    s = re.sub(r"\.com\b|\.net\b|\.org\b|\bwww\.", " ", s)
    toks = collapse_initials(TOKEN_RE.findall(s))
    norm, core, legal = [], [], []
    for t in toks:
        c = LEGAL_CANON.get(t)
        if not c and len(t) > 4 and not t.isdigit():
            c = LEGAL_SKEL.get(skeleton(t))
        if c:
            for p in c.split():
                norm.append(p)
                legal.append(p)
            continue
        norm.append(t)
        if t not in NAME_DROP:
            core.append(t)
    if not core:  # name consisted only of legal words: keep them as core
        core = [t for t in norm]
    legal_s = " ".join(sorted(set(legal)))
    skel = " ".join(skeleton(t) for t in core)
    return " ".join(norm), " ".join(core), legal_s, skel


def normalize_address(raw: str) -> tuple[str, str, str, str, str]:
    """Return (addr_norm, addr_no_landmark, addr_numbers, postcode, city).

    * addr_norm: canonical street types / regions, null tokens removed.
    * addr_numbers: space-joined numeric tokens (house/street numbers), postcode excluded.
    * postcode: first 5/6-digit code found anywhere (no country gating).
    * city: best-effort - the last comma-separated component that has no digits
      and is not a region name.
    """
    s = basic_clean(raw)
    if not s:
        return "", "", "", "", ""
    m = POSTCODE_RE.search(s)
    postcode = m.group(1).replace(" ", "") if m else ""
    s = ORD_RE.sub(r"\1", s)
    s = _REGION_RE.sub(lambda mm: " " + REGION_CANON[mm.group(1)] + " ", re.sub(r"-", " ", s))
    parts = [p.strip() for p in s.split(",")]
    no_lm = LANDMARK_RE.sub(" ", s)

    def canon(text: str) -> list[str]:
        """Tokenise and canonicalise street words, dropping null tokens."""
        out = []
        for t in TOKEN_RE.findall(text):
            if t in NULL_TOKENS:
                continue
            out.append(STREET_CANON.get(t, t))
        return out

    toks = canon(s)
    nums = [t for t in toks if t.isdigit() and t != postcode]
    region_codes = set(REGION_CANON.values())
    city = ""
    for p in reversed(parts):
        ct = canon(p)
        if not ct or any(ch.isdigit() for ch in p):
            continue
        if len(ct) <= 2 and all(t in region_codes for t in ct):
            continue
        city = " ".join(t for t in ct if t not in {"city", "twp", "urban", "rural"})
        break
    return " ".join(toks), " ".join(canon(no_lm)), " ".join(nums), postcode, city


NORM_COLS = ["name_norm", "name_core", "name_legal", "name_skel",
             "addr_norm", "addr_no_landmark", "addr_numbers", "postcode", "city"]


def _norm_chunk(args):
    """Worker: normalise lists of names and addresses into a pyarrow-string DataFrame."""
    names, addrs = args
    rows = [normalize_name(n) + normalize_address(a) for n, a in zip(names, addrs)]
    return pd.DataFrame(rows, columns=NORM_COLS).astype("string[pyarrow]")


def normalize_frame(df: pd.DataFrame, n_jobs: int = 8, chunk: int = 50_000) -> pd.DataFrame:
    """Add all normalised fields to ``df`` (in parallel); returns a new DataFrame."""
    names = df["business_name"].astype(str).tolist()
    addrs = df["business_address"].astype(str).tolist()
    tasks = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    if n_jobs > 1 and len(tasks) > 1:
        with Pool(n_jobs) as pool:
            res = pool.map(_norm_chunk, tasks)
    else:
        res = [_norm_chunk(t) for t in tasks]
    out = pd.concat(res, ignore_index=True)
    out.index = df.index
    return pd.concat([df, out], axis=1)


if __name__ == "__main__":
    tests = [
        ("डिजिटल इंफोटेक प्राइवेट लिमिटेड", "E-32, E-Wing, Nr.Shivaji Maharaj Statue, Vasai, Thane, Maharashtra"),
        ("ಸೌತ್ ಪ್ರೊಡಕ್ಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "NO 07, BENGALURU, Karnataka"),
        ("Digital Infotech Private Limited", "110 1/2 Azalea Ave, Hueytown, Alabama"),
        ("Fractales Amis Groupe S.A.S", "23 Rue Icmre, La Teste-de-buch, Gironde 33260"),
        ("Eisenberg & Volkman Bitwise L.L.C.", "Montpelier, OH, 318 Court Street"),
        ("Veonexx F/K/A Corey Bright Inc", "1600 Southeastern Avenue, Unit UNIT 305, Sioux Falls, SD"),
        ("பர்ஃபெக்ட் இன்ஃப்ராஸ்ட்ரக்சர்", "Chennai, TN 600040"),
        ("ശ്യാം ഇൻഫ്രാസ്ട്രക്ചർ", "x"), ("শিবা ইনভেস্টমেন্ট এলএলপি", "x"),
    ]
    for n, a in tests:
        print(n, "->", normalize_name(n))
        print("   ", a, "->", normalize_address(a))
