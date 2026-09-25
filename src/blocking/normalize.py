"""Text normalisation and tokenisation used by blocking.

Kept deliberately simple and deterministic: lowercase, strip accents from Latin
letters only (Indic vowel signs are kept so words in Devanagari etc. stay intact),
split on anything that is not a letter, digit or combining mark.
"""

import re
import unicodedata

# Legal / generic words that are frequently added, dropped or swapped between
# sources (measured in docs/blocking_data_analysis.md) and carry no identity.
LEGAL_WORDS = frozenset(
    "inc incorporated llc ltd limited pvt private corp corporation co company "
    "lp llp plc pllc pc".split()
)
NOISE_WORDS = frozenset(
    "the of and services service center centre partners group mr mrs ms smt shri sri "
    "dba fka aka formerly known as d b a com www".split()
)
INDIC_LEGAL_WORDS = frozenset({"प्राइवेट", "लिमिटेड", "एलएलपी", "प्रा", "लि"})
NAME_STOPWORDS = LEGAL_WORDS | NOISE_WORDS | INDIC_LEGAL_WORDS

# Tokens produced by address placeholders such as "<NULL>" and "N/A".
ADDRESS_STOPWORDS = frozenset({"null", "n", "a"})

_LEADING_NUMBER = re.compile(r"0*(\d+)")


def is_nonlatin_letter(ch):
    return ch.isalpha() and ord(ch) > 0x24F


def has_nonlatin(text):
    return any(is_nonlatin_letter(c) for c in text)


def fold(text):
    """Return the lowercase, accent-folded word tokens of ``text``."""
    kept = []
    prev_latin = False
    for ch in unicodedata.normalize("NFKD", text):
        if unicodedata.combining(ch) and prev_latin:
            continue  # accent on a Latin letter
        prev_latin = ord(ch) < 0x250
        kept.append(ch)
    lowered = "".join(kept).lower()
    cleaned = "".join(
        c if (c.isalnum() or unicodedata.category(c)[0] == "M") else " " for c in lowered
    )
    return cleaned.split()


def name_tokens(name):
    """Informative name tokens (order preserved, stop-words removed)."""
    return [t for t in fold(name) if t not in NAME_STOPWORDS]


def address_tokens(address):
    """Address tokens with placeholder fragments removed."""
    return [t for t in fold(address) if t not in ADDRESS_STOPWORDS]


def house_number(address):
    """First token starting with digits, without '#' or leading zeros ('' if none)."""
    for part in address.replace("#", " ").replace(",", " ").split():
        m = _LEADING_NUMBER.match(part)
        if m:
            return m.group(1)
    return ""
