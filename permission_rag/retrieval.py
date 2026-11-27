"""Hybrid retrieval: BM25 + dense, fused with RRF — over ACL-visible chunks only.

Design notes (borrowed from Gopi's rag-skills findings):
- BM25 (Okapi, k1=1.5, b=0.75) and a dense retriever are fused with
  Reciprocal Rank Fusion (Cormack et al., SIGIR 2009, k=60) — rank-only
  fusion, so scores on different scales never mix.
- RRF scores carry no abstention signal (measured in rag-skills: hit-control
  gap ~0.000), so abstention is gated on content-word overlap instead —
  see :mod:`permission_rag.answer`.
- Dense backend: Ollama embeddings (``nomic-embed-text``) when an Ollama
  server has that model, otherwise a deterministic *hashed* embedding —
  each token maps (via md5) to a fixed random unit vector, summed and
  L2-normalised. The hashed backend is a lightweight stand-in, NOT a neural
  encoder; the demo prints which backend actually ran. A production
  deployment would use a real encoder (e.g. all-MiniLM-L6-v2).

CRITICAL: the retriever is always built on the ACL-visible subset for one
user (see :class:`PermissionAwareRetriever`). BM25 IDF statistics are
computed on visible chunks only, so even term statistics cannot leak
denied documents.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from collections import Counter, defaultdict

from .acl import User, visible_chunks

TOKEN_RE = re.compile(r"[^a-z0-9\s]")
RRF_K = 60
DENSE_DIM = 256
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_EMBED_MODEL = "nomic-embed-text"


def tokenize(text: str) -> list:
    return TOKEN_RE.sub(" ", text.lower()).split()


# --------------------------------------------------------------------- BM25
class Bm25Retriever:
    """BM25 Okapi, pure Python. k1=1.5, b=0.75."""

    name = "bm25"

    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    def fit(self, chunks):
        self.chunks = chunks
        self.docs = [tokenize(c.text) for c in chunks]
        n = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(n, 1)
        df = Counter()
        for toks in self.docs:
            df.update(set(toks))
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5))
                    for t, c in df.items()}

    def search(self, query):
        qtoks = tokenize(query)
        out = []
        for i, toks in enumerate(self.docs):
            tf = Counter(toks)
            dl = len(toks)
            score = 0.0
            for t in qtoks:
                if t not in self.idf:
                    continue
                f = tf.get(t, 0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += self.idf[t] * f * (self.k1 + 1) / denom
            out.append((score, i))
        return sorted(out, key=lambda x: -x[0])


# -------------------------------------------------------------------- dense
def _token_vector(token: str, dim: int = DENSE_DIM):
    """Deterministic pseudo-random unit vector for one token (hashed)."""
    digest = hashlib.md5(token.encode()).digest()
    seed = int.from_bytes(digest[:8], "little")
    # xorshift64 — no numpy needed, fully deterministic
    x = seed or 0x9E3779B97F4A7C15
    vec = {}
    for _ in range(8):  # 8 non-zero dims per token: sparse, fast, stable
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        idx = x % dim
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        sign = 1.0 if (x & 1) else -1.0
        vec[idx] = vec.get(idx, 0.0) + sign
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def hashed_embed(text: str, dim: int = DENSE_DIM, idf: dict | None = None) -> dict:
    """Bag-of-hashed-token-vectors embedding, L2-normalised (sparse dict).

    Tokens are weighted by IDF when provided — without it, stopwords
    dominate the sum and the embedding is mostly noise (measured: dense
    top-1 agreed with BM25 on only 3/8 probe queries unweighted).
    """
    vec = Counter()
    for tok in set(tokenize(text)):
        w = idf.get(tok, max(idf.values()) if idf else 1.0) if idf else 1.0
        for k, v in _token_vector(tok, dim).items():
            vec[k] += w * v
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def _cosine_sparse(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def ollama_embeddings_available() -> bool:
    """True if a local Ollama server lists the embedding model."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            models = json.load(resp).get("models", [])
        return any(m.get("name", "").startswith(OLLAMA_EMBED_MODEL)
                   for m in models)
    except Exception:
        return False


def ollama_embed(texts: list) -> list:
    """Embed texts with Ollama ``nomic-embed-text``; returns dense lists."""
    out = []
    for text in texts:
        payload = json.dumps(
            {"model": OLLAMA_EMBED_MODEL, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/embeddings", data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            out.append(json.load(resp)["embedding"])
    return out


class DenseRetriever:
    """Dense retrieval: Ollama embeddings when available, else hashed.

    ``backend`` reports which one actually ran — the demo prints it, so a
    hashed-embedding run is never passed off as a neural-encoder run.
    """

    name = "dense"
    backend = "hashed"

    def __init__(self, use_ollama: bool = True):
        self.use_ollama = use_ollama and ollama_embeddings_available()
        self.backend = "ollama:nomic-embed-text" if self.use_ollama else "hashed"

    def fit(self, chunks):
        self.chunks = chunks
        if self.use_ollama:
            embs = ollama_embed([c.text for c in chunks])
            self.vecs = embs  # dense lists, already ~unit norm
        else:
            docs = [tokenize(c.text) for c in chunks]
            n = len(docs)
            df = Counter()
            for toks in docs:
                df.update(set(toks))
            self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5))
                        for t, c in df.items()}
            self.vecs = [hashed_embed(c.text, idf=self.idf) for c in chunks]

    def search(self, query):
        if self.use_ollama:
            q = ollama_embed([query])[0]
            qn = math.sqrt(sum(v * v for v in q)) or 1.0
            out = []
            for i, vec in enumerate(self.vecs):
                dot = sum(a * b for a, b in zip(q, vec))
                vn = math.sqrt(sum(v * v for v in vec)) or 1.0
                out.append((dot / (qn * vn), i))
        else:
            q = hashed_embed(query, idf=getattr(self, "idf", None))
            out = [(_cosine_sparse(q, vec), i)
                   for i, vec in enumerate(self.vecs)]
        return sorted(out, key=lambda x: -x[0])


# ------------------------------------------------------------------- hybrid
class HybridRetriever:
    """RRF fusion over BM25 + dense. Rank-only, k=60."""

    name = "hybrid"

    def __init__(self, dense: DenseRetriever | None = None):
        self.parts = [Bm25Retriever(), dense or DenseRetriever()]

    def fit(self, chunks):
        self.chunks = chunks
        for p in self.parts:
            p.fit(chunks)

    def search(self, query):
        fused = defaultdict(float)
        for p in self.parts:
            for rank, (_, idx) in enumerate(p.search(query), start=1):
                fused[idx] += 1.0 / (RRF_K + rank)
        return [(score, idx) for idx, score in
                sorted(fused.items(), key=lambda kv: -kv[1])]


# ------------------------------------------------------- permission-aware
class PermissionAwareRetriever:
    """Retrieval with ACL pre-filtering enforced at retrieval time.

    ``search(query, user)`` first restricts the index to chunks the user
    may see, builds the hybrid retriever on that subset only, and returns
    hits from that subset. Denied documents never reach the ranker.
    """

    def __init__(self, chunks):
        self.all_chunks = chunks
        self._cache = {}  # frozenset(roles) -> HybridRetriever

    def _retriever_for(self, user: User) -> HybridRetriever:
        key = user.roles
        if key not in self._cache:
            visible = visible_chunks(user, self.all_chunks)
            retr = HybridRetriever()
            retr.fit(visible)
            self._cache[key] = retr
        return self._cache[key]

    def search(self, query: str, user: User, top_k: int = 3):
        """Return [(score, Chunk)] — only chunks ``user`` may see."""
        retr = self._retriever_for(user)
        ranked = retr.search(query)[:top_k]
        return [(score, retr.chunks[idx]) for score, idx in ranked]

    def visible_doc_ids(self, user: User) -> set:
        return {c.doc_id for c in visible_chunks(user, self.all_chunks)}

    def dense_backend(self) -> str:
        return self._retriever_for(
            User("probe", ["__none__"])).parts[1].backend
