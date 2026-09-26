"""Deterministic, offline Indic-script -> Latin transliteration and phonetic keys.

No external library or data: the nine major Brahmic scripts in Unicode
(Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada,
Malayalam) share one layout (each block is 0x80 wide and letters sit at the same
offsets), so a single offset table covers all of them.

The goal is not a scholarly transliteration but a form whose *phonetic key*
agrees with the English spelling, e.g. "पायोनियर" -> "payoniyar" -> key "pnr",
the same key as "Pioneer".
"""

import unicodedata

BRAHMIC_START, BRAHMIC_END = 0x0900, 0x0DFF

# offset within a script block -> Latin
_CONSONANTS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "n",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v",
    0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_INDEPENDENT_VOWELS = {
    0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu", 0x0B: "ri",
    0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o",
    0x13: "o", 0x14: "au", 0x60: "rii", 0x61: "lii",
}
_VOWEL_SIGNS = {
    0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu", 0x43: "ri", 0x44: "rii",
    0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
    0x4C: "au", 0x57: "au", 0x62: "li", 0x63: "lii",
}
_NUKTA_FORMS = {"ph": "f", "j": "z", "k": "q", "d": "r", "dh": "rh", "kh": "kh", "g": "g"}
_LABIALS = {"p", "ph", "b", "bh", "m", "f"}
_VIRAMA, _NUKTA, _ANUSVARA, _CANDRABINDU, _VISARGA = 0x4D, 0x3C, 0x02, 0x01, 0x03
# Malayalam chillu letters (consonants without inherent vowel).
_CHILLU = {0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k"}
_MALAYALAM_RRA = 0x0D31  # doubled (with virama) it is pronounced "tt"


def is_brahmic(ch):
    return BRAHMIC_START <= ord(ch) <= BRAHMIC_END


def to_latin(text):
    """Transliterate Brahmic-script characters in ``text``; other characters pass through."""
    out = []
    pending = False  # a consonant is waiting for its (inherent) vowel
    chars = list(text)
    for idx, ch in enumerate(chars):
        cp = ord(ch)
        prev2_cp, prev_cp = (ord(chars[idx - 2]) if idx >= 2 else 0), (ord(chars[idx - 1]) if idx >= 1 else 0)
        if cp in (0x200C, 0x200D):  # zero-width (non-)joiner
            continue
        if cp in _CHILLU:
            if pending:
                out.append("a")
                pending = False
            out.append(_CHILLU[cp])
            continue
        if not is_brahmic(ch):
            if pending:
                pending = False  # word-final inherent vowel is dropped (schwa deletion)
            out.append(ch)
            continue
        off = cp & 0x7F
        if unicodedata.category(ch) == "Nd":
            if pending:
                pending = False
            out.append(str(unicodedata.digit(ch)))
            continue
        if off in _CONSONANTS:
            if pending:
                out.append("a")
            if cp == _MALAYALAM_RRA and prev2_cp == _MALAYALAM_RRA and (prev_cp & 0x7F) == _VIRAMA:
                out[-1] = "t"
                out.append("t")
            else:
                out.append(_CONSONANTS[off])
            pending = True
        elif off == _NUKTA:
            if out and out[-1] in _NUKTA_FORMS:
                out[-1] = _NUKTA_FORMS[out[-1]]
        elif off in _VOWEL_SIGNS:
            out.append(_VOWEL_SIGNS[off])
            pending = False
        elif off == _VIRAMA:
            pending = False
        elif off in (_ANUSVARA, _CANDRABINDU):
            if pending:
                out.append("a")
                pending = False
            nxt = _next_consonant(chars, idx + 1)
            at_word_end = idx + 1 >= len(chars) or not is_brahmic(chars[idx + 1])
            out.append("m" if (nxt in _LABIALS or at_word_end) else "n")
        elif off == _VISARGA:
            if pending:
                out.append("a")
                pending = False
            out.append("h")
        elif off in _INDEPENDENT_VOWELS:
            if pending:
                out.append("a")
                pending = False
            out.append(_INDEPENDENT_VOWELS[off])
        else:  # other signs (avagraha, stress marks, ...): drop
            if pending:
                out.append("a")
                pending = False
    return "".join(out)


def _next_consonant(chars, start):
    if start < len(chars) and is_brahmic(chars[start]):
        return _CONSONANTS.get(ord(chars[start]) & 0x7F)
    return None


# ---------------------------------------------------------------- phonetic keys
_OCR_DIGITS = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})
_VOWELS = set("aeiouy")


def phonetic_key(token):
    """Consonant skeleton of a lowercase ASCII token, robust to spelling and
    transliteration variants ("consultants" and "kansaltents" -> "knsltnts").

    Pure numbers return "" (numbers are handled as address tokens instead).
    Digits inside words are read as look-alike letters ("m0lecular").
    """
    if not token or token.isdigit():
        return ""
    t = token.translate(_OCR_DIGITS)
    if not t.isascii() or not t.isalpha():
        return ""
    t = t.replace("ph", "f").replace("x", "ks")
    out = []
    for i, c in enumerate(t):
        if c == "h" and i > 0 and t[i - 1] not in _VOWELS:
            continue  # aspiration / digraph second letter: th, sh, ch, kh, bh ...
        if c in _VOWELS:
            if i == 0:
                out.append("a")  # keep "starts with a vowel", not which vowel
            continue
        out.append({"c": "k", "q": "k", "z": "j", "g": "j", "w": "v"}.get(c, c))
    key = []
    for c in out:
        if not key or key[-1] != c:
            key.append(c)
    return "".join(key)
