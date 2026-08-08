"""Text -> fixed-width hashed character n-gram vector."""

import zlib

import numpy as np

FEATURE_DIM = 4096
NGRAM_SIZES = (3, 4, 5)


def normalise_text(text: str) -> str:
    """Lowercase and collapse all whitespace runs to single spaces."""
    return " ".join(text.lower().split())


def hash_ngram(ngram: str) -> int:
    """Stable across processes. NEVER use builtin hash() here."""
    return zlib.crc32(ngram.encode("utf-8")) % FEATURE_DIM


def featurise(text: str) -> np.ndarray:
    """Return an L2-normalised (4096,) float32 vector of char n-gram counts."""
    cleaned = normalise_text(text)
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    count = 0
    for n in NGRAM_SIZES:
        for i in range(len(cleaned) - n + 1):
            vec[hash_ngram(cleaned[i : i + n])] += 1.0
            count += 1

    if count == 0:
        raise ValueError(
            f"no n-grams produced: text is shorter than {min(NGRAM_SIZES)} "
            f"characters after whitespace collapsing (got {len(cleaned)!r})"
        )

    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise ValueError("zero-norm feature vector despite non-zero n-gram count")

    return (vec / norm).astype(np.float32)


def featurise_batch(texts) -> np.ndarray:
    """Return an (N, 4096) float32 matrix, one L2-normalised row per text."""
    texts = list(texts)
    out = np.zeros((len(texts), FEATURE_DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        out[i] = featurise(text)
    return out
