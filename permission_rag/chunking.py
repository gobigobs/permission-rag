"""Chunking: split documents into word-boundary chunks (stdlib only).

Mirrors the approach in Gopi's rag-skills project (chunking A: 500 chars /
100 overlap) so results stay comparable in spirit.
"""

from __future__ import annotations

from .acl import Chunk, Document


def chunk_text(text: str, size: int = 500, overlap: int = 100) -> list:
    """Greedy char chunks, split on word boundaries, never mid-word."""
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_document(doc: Document, size: int = 500, overlap: int = 100) -> list:
    """Chunk one document; every chunk inherits the document's ACL."""
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}#c{i}",
            doc_id=doc.doc_id,
            title=doc.title,
            text=piece,
            source=doc.source,
            acl=doc.acl,
        )
        for i, piece in enumerate(chunk_text(doc.text, size, overlap))
    ]


def chunk_documents(docs, size: int = 500, overlap: int = 100) -> list:
    out = []
    for doc in docs:
        out.extend(chunk_document(doc, size, overlap))
    return out
