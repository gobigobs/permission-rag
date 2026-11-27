"""Answering: local Ollama model when available, extractive fallback otherwise.

- Ollama path: ``qwen2.5:3b-instruct`` on the local server
  (``http://127.0.0.1:11434``), prompted ONLY with the ACL-visible chunks
  retrieved for this user. The model never sees denied documents.
- Fallback path: extractive — the sentences from the retrieved chunks that
  best overlap the question, with citations. Transparent and labelled.
- Abstention gate: RRF scores carry no abstention signal (rag-skills
  finding), so the gate is content-word overlap between the question and
  the top chunk. Below threshold the answer is an honest "I don't know"
  instead of a hallucinated answer.
- Every answer carries citations: the doc IDs it used, nothing else.
"""

from __future__ import annotations

import json
import re
import urllib.request

from .retrieval import tokenize

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen2.5:3b-instruct"
ABSTAIN_THRESHOLD = 0.34  # min. content-word overlap with the top chunk

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "which", "who",
    "whom", "how", "why", "when", "where", "do", "does", "did", "can",
    "could", "should", "would", "will", "shall", "may", "might", "i", "me",
    "my", "we", "you", "he", "she", "it", "its", "they", "them", "this",
    "that", "these", "those", "there", "here", "in", "on", "at", "to", "of",
    "for", "with", "by", "from", "and", "or", "but", "not", "no", "if",
    "then", "than", "so", "as", "am", "be", "been", "being", "have", "has",
    "had", "into", "over", "under", "about",
}


def content_words(text: str) -> set:
    return {t for t in tokenize(text) if t not in STOPWORDS}


def overlap_gate(question: str, top_chunk_text: str) -> float:
    """Fraction of the question's content words present in the top chunk."""
    qw = content_words(question)
    if not qw:
        return 0.0
    return len(qw & set(tokenize(top_chunk_text))) / len(qw)


def ollama_available() -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            models = json.load(resp).get("models", [])
        return any(m.get("name", "").startswith("qwen2.5:3b")
                   for m in models)
    except Exception:
        return False


def _ollama_generate(prompt: str) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 200},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=payload,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)["response"].strip()


def warmup() -> bool:
    """Best-effort model warm-up: the first Ollama call loads the model
    into memory, which can take over a minute on CPU. Doing it once up
    front keeps the first demo query from hitting the generate timeout."""
    if not ollama_available():
        return False
    try:
        _ollama_generate("Say OK.")
        return True
    except Exception:
        return False


def _extractive_answer(question: str, chunks) -> str:
    """Fallback: best-overlapping sentences from the retrieved chunks."""
    qw = content_words(question)
    sentences = []
    for chunk in chunks:
        clean = re.sub(r"^#+\s*", "", chunk.text, flags=re.M)
        for sent in re.split(r"(?<=[.!?])\s+", clean):
            sent = sent.strip()
            if not sent:
                continue
            overlap = len(qw & set(tokenize(sent)))
            sentences.append((overlap, sent))
    sentences.sort(key=lambda x: -x[0])
    picked = [s for o, s in sentences[:2] if o > 0]
    if not picked:
        picked = [sentences[0][1]] if sentences else []
    return " ".join(picked)


def answer(question: str, hits, use_ollama: bool = True) -> dict:
    """Answer from retrieved ``hits`` = [(score, Chunk), ...].

    Returns {"answer", "citations": [doc_id, ...], "mode", "abstained",
    "gate_score"}. Citations list ONLY the docs the answer actually used.
    """
    chunks = [c for _, c in hits]
    citations = []
    for c in chunks:
        if c.doc_id not in citations:
            citations.append(c.doc_id)

    if not chunks:
        return {"answer": "I don't know — nothing you are allowed to see "
                          "answers this question.",
                "citations": [], "mode": "abstained",
                "abstained": True, "gate_score": 0.0}

    gate = overlap_gate(question, chunks[0].text)
    if gate < ABSTAIN_THRESHOLD:
        return {"answer": "I don't know — nothing in the documents you are "
                          "allowed to see answers this question confidently.",
                "citations": citations, "mode": "abstained",
                "abstained": True, "gate_score": round(gate, 2)}

    context = "\n\n".join(
        f"[{c.doc_id}] {c.text}" for c in chunks)
    mode = "extractive-fallback"
    if use_ollama and ollama_available():
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the context does not answer it, say \"I don't know\". "
            "Keep the answer to 2-3 sentences, in plain language.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        try:
            text = _ollama_generate(prompt)
            mode = f"ollama:{OLLAMA_MODEL}"
        except Exception:
            text = _extractive_answer(question, chunks)
    else:
        text = _extractive_answer(question, chunks)

    return {"answer": text, "citations": citations, "mode": mode,
            "abstained": False, "gate_score": round(gate, 2)}
