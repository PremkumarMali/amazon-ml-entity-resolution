"""Per-record blocking features: sets of string tokens per field.

Fields
  n   informative name tokens (original script)
  p   phonetic keys of the name after offline transliteration (script/typo robust)
  pb  adjacent phonetic-key pairs of the name (rare even when each word is common)
  a   address tokens (leading zeros of numbers removed)
  ab  adjacent address-token pairs that contain a number ("3 55", "no 52")
  g   character trigrams of the transliterated, space-free name (fallback only)

``FEATURE_VERSION`` must change whenever the output of this module changes, so
cached encodings are rebuilt.
"""

from .normalize import NAME_STOPWORDS, address_tokens, has_nonlatin, name_tokens
from .transliterate import phonetic_key, to_latin

FEATURE_VERSION = 2
FLAG_NONLATIN_NAME = 1  # record flags (bit mask)
FLAG_EMPTY_ADDRESS = 2
NAME_FIELDS = ("n", "p", "pb")
ADDRESS_FIELDS = ("a", "ab")
FALLBACK_FIELDS = ("g",)
ALL_FIELDS = NAME_FIELDS + ADDRESS_FIELDS + FALLBACK_FIELDS

PHONETIC_STOPWORDS = frozenset(
    {phonetic_key(w) for w in NAME_STOPWORDS if w.isascii()} | {"prvt", "lmtd", "pvt", "ltd", "lmt"}
)


def latin_name(name):
    return to_latin(name) if has_nonlatin(name) else name


def _pairs(tokens):
    return {" ".join(sorted(tokens[i:i + 2])) for i in range(len(tokens) - 1)}


def _address_tokens(address):
    return [(t.lstrip("0") or "0") if t.isdigit() else t for t in address_tokens(address)]


def trigrams(text):
    s = "".join(text.split())
    return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else ({s} if s else set())


def record_flags(name, address):
    """Bit mask of structural traits that make a record's scores low (see engine)."""
    return (FLAG_NONLATIN_NAME if has_nonlatin(name) else 0) | (0 if address_tokens(address) else FLAG_EMPTY_ADDRESS)


def record_features(name, address, with_fallback=True):
    """``{field: set(tokens)}`` for one record."""
    latin = latin_name(name)
    latin_tokens = name_tokens(latin)
    keys = []
    for t in latin_tokens:
        k = phonetic_key(t)
        if len(k) >= 2 and k not in PHONETIC_STOPWORDS:
            keys.append(k)
    addr = _address_tokens(address)
    feats = {
        "n": set(name_tokens(name)),
        "p": set(keys),
        "pb": _pairs(keys),
        "a": set(addr),
        "ab": {p for p in _pairs(addr) if any(c.isdigit() for c in p)},
    }
    if with_fallback:
        feats["g"] = trigrams(" ".join(latin_tokens))
    return feats
