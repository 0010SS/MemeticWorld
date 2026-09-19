"""Local, deterministic text embeddings.

Upstream Generative Agents uses OpenAI `text-embedding-ada-002`. For an offline,
free and bit-reproducible hackathon build we default to a hashed bag of word
unigrams/bigrams + character trigrams (feature hashing, sublinear TF, L2-norm).
It captures lexical relevance well and topical relevance only weakly.
`sentence_transformers` can be selected in config if installed.
"""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache

import numpy as np

_STOP = set("""a an the and or but if then so of to in on at for with by from as is are was were be been
being it its this that these those i you he she they we me him her them my your his their our
do does did done have has had not no yes just very really about into over after before up down out
there here what which who whom when where why how all any some can could would should will shall
may might must also too than""".split())

_WORD = re.compile(r"[a-z0-9']+")


def _h(s: str, dim: int) -> tuple[int, float]:
    d = hashlib.md5(s.encode()).digest()
    return int.from_bytes(d[:4], "little") % dim, (1.0 if d[4] & 1 else -1.0)


class HashEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def features(self, text: str):
        words = [w for w in _WORD.findall(text.lower())]
        content = [w for w in words if w not in _STOP]
        feats = {}
        for w in content:
            feats["w:" + w] = feats.get("w:" + w, 0) + 1.0
            if len(w) > 3:  # crude stemming signal via prefixes
                feats["p:" + w[:5]] = feats.get("p:" + w[:5], 0) + 0.5
        for a, b in zip(content, content[1:]):
            k = f"b:{a}_{b}"
            feats[k] = feats.get(k, 0) + 0.7
        for w in content:
            s = f"#{w}#"
            for i in range(len(s) - 2):
                k = "c:" + s[i:i + 3]
                feats[k] = feats.get(k, 0) + 0.15
        return feats

    def __call__(self, text: str) -> list[float]:
        return list(_cached(self.dim, text))


@lru_cache(maxsize=50000)
def _cached(dim: int, text: str):
    v = np.zeros(dim, dtype=np.float64)
    for k, c in HashEmbedder(dim).features(text).items():
        i, s = _h(k, dim)
        v[i] += s * (1.0 + math.log(c)) if c >= 1 else s * c
    n = np.linalg.norm(v)
    if n > 0:
        v /= n
    return tuple(float(x) for x in np.round(v, 6))


class STEmbedder:
    def __init__(self, model="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model)

    def __call__(self, text: str) -> list[float]:
        return [float(x) for x in self.m.encode([text], normalize_embeddings=True)[0]]


def make_embedder(cfg: dict):
    kind = (cfg or {}).get("backend", "hash")
    if kind == "sentence_transformers":
        return STEmbedder(cfg.get("model", "all-MiniLM-L6-v2"))
    return HashEmbedder(int((cfg or {}).get("dim", 512)))


def cos(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))
