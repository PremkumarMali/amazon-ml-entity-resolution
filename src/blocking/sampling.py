"""Deterministic, order-independent sampling of entity IDs."""

import hashlib

DEFAULT_SALT = "blocking-validation-v1"


def hash_fraction(entity_id, salt=DEFAULT_SALT):
    """Map an ID to a stable number in [0, 1)."""
    digest = hashlib.md5(f"{salt}:{entity_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def in_sample(entity_id, rate, salt=DEFAULT_SALT):
    """True for a stable ``rate`` fraction of IDs, independent of file order."""
    return hash_fraction(entity_id, salt) < rate
