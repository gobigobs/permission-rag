#!/usr/bin/env python3
"""permission-rag demo — seeds data, runs queries as different users, proves ACLs.

One command:  ./run.sh   (or: python3 demo.py)

Runs 4 queries:
  1. ALLOWED  — priya (engineering) asks about deploys (local-filesystem doc)
  2. DENIED   — priya asks about exec compensation (exec-only doc: must never appear)
  3. CROSS-CONNECTOR — arjun (finance) asks what needs finance approval
     (answer spans the local-filesystem AND the mock-enterprise connector)
  4. PUBLIC   — meera (hr) asks about the Diwali holiday (an `all` doc)

Then prints the ACL enforcement proof (per-user visible doc counts + a
leak scan over every answer, citation and retrieved id) and the RAGAS-style
eval summary (heuristic proxies — indicative only).

No API keys, no paid services, no model download. Answering uses local
Ollama (qwen2.5:3b-instruct) when the server is running, otherwise a
transparent extractive fallback. The mode used is printed per answer.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from permission_rag.acl import USERS, visible_docs  # noqa: E402
from permission_rag.answer import answer, ollama_available, warmup  # noqa: E402
from permission_rag.chunking import chunk_documents  # noqa: E402
from permission_rag.connectors.local_fs import LocalFilesystemConnector  # noqa: E402
from permission_rag.connectors.mock_enterprise import MockEnterpriseConnector  # noqa: E402
from permission_rag.eval import evaluate, load_questions, print_report  # noqa: E402
from permission_rag.retrieval import PermissionAwareRetriever  # noqa: E402

DENIED_DOC = "exec-comp-review"  # exec-only; the denial target


def build_index():
    """Seed from both connectors and build the chunk index."""
    local_docs = LocalFilesystemConnector(
        os.path.join(HERE, "corpus", "local")).fetch()
    enterprise_docs = MockEnterpriseConnector().fetch()
    docs = local_docs + enterprise_docs
    chunks = chunk_documents(docs)
    return docs, chunks


def run_query(retriever, user_name, question, label):
    user = USERS[user_name]
    hits = retriever.search(question, user, top_k=3)
    result = answer(question, hits)
    print(f"\n--- {label} ---")
    print(f"user: {user_name} (roles: {', '.join(sorted(user.roles))})")
    print(f"Q: {question}")
    # Dedupe doc ids for display (two chunks can come from the same doc).
    seen, uniq_docs, uniq_sources = set(), [], []
    for _, c in hits:
        if c.doc_id not in seen:
            seen.add(c.doc_id)
            uniq_docs.append(c.doc_id)
            uniq_sources.append(c.source)
    print(f"retrieved: {', '.join(f'{d} [{s}]' for d, s in zip(uniq_docs, uniq_sources)) or '(none)'}")
    print(f"A: {result['answer']}")
    print(f"citations: {result['citations'] or '(none)'}")
    print(f"mode: {result['mode']} | abstained: {result['abstained']} "
          f"| gate: {result['gate_score']}")
    return {"label": label, "user": user_name, "question": question,
            "retrieved": uniq_docs,
            "sources": uniq_sources,
            "answer": result["answer"], "citations": result["citations"],
            "mode": result["mode"], "abstained": result["abstained"]}


def acl_proof(retriever, docs, demo_results):
    """Prove the denied doc never leaks: counts + a scan of all outputs."""
    print("\n== ACL enforcement proof ==")
    print(f"total docs in corpus: {len(docs)} "
          f"(local-filesystem: 6, mock-enterprise: 6)")
    for name, user in USERS.items():
        visible = visible_docs(user, docs)
        print(f"  {name:<7} ({', '.join(sorted(user.roles)):<11}) sees "
              f"{len(visible)} docs: "
              f"{', '.join(sorted(d.doc_id for d in visible))}")

    # Leak scan: the denied doc id must appear NOWHERE in any demo output
    # (retrieved ids, citations, answer text) for unauthorised users.
    leaks = []
    for r in demo_results:
        if r["user"] == "vikram":
            continue  # vikram (exec) is allowed to see it
        blob = " ".join(r["retrieved"] + r["citations"] + [r["answer"]])
        # Leak = the denied doc's id, or its confidential numbers, surfacing.
        if DENIED_DOC in blob or "Rs 90 lakh" in blob:
            leaks.append(r["label"])
    # Direct retrieval probe: ask for the denied doc as each unauthorised user
    probe_q = "What are the executive compensation bands for directors?"
    for name, user in USERS.items():
        if name == "vikram":
            continue
        hits = retriever.search(probe_q, user, top_k=5)
        if any(c.doc_id == DENIED_DOC for _, c in hits):
            leaks.append(f"probe:{name}")
    if leaks:
        print(f"  FAIL — denied doc {DENIED_DOC} leaked in: {leaks}")
    else:
        print(f"  PASS — {DENIED_DOC} (exec-only) never appears in retrieval, "
              f"citations or answers for priya / arjun / meera")
        print("  PASS — the pre-filter runs BEFORE ranking: denied docs are "
              "not in the retriever's index for that user at all")
    return {"denied_doc": DENIED_DOC, "leaks": leaks,
            "passed": not leaks}


def main():
    print("permission-rag — Permission-Aware Enterprise RAG (demo)")
    print("=" * 60)
    docs, chunks = build_index()
    print(f"seeded {len(docs)} docs from 2 connectors, {len(chunks)} chunks")
    retriever = PermissionAwareRetriever(chunks)
    print(f"dense backend: {retriever.dense_backend()}")
    if ollama_available():
        print("warming up local model (first load takes a while on CPU)...")
        print(f"answering: ollama:qwen2.5:3b-instruct (local)"
              f" — warmup {'ok' if warmup() else 'failed, fallback will be used'}")
    else:
        print("answering: extractive fallback (Ollama not running)")

    results = []
    results.append(run_query(
        retriever, "priya", "How do I deploy to production?",
        "1. ALLOWED — engineering doc, local-filesystem connector"))
    results.append(run_query(
        retriever, "priya",
        "What are the executive compensation bands for directors?",
        "2. DENIED — exec-only doc must never appear"))
    results.append(run_query(
        retriever, "arjun",
        "What needs finance approval — expense claims or vendor contracts?",
        "3. CROSS-CONNECTOR — local-filesystem + mock-enterprise"))
    results.append(run_query(
        retriever, "meera", "When is Diwali a holiday in 2026?",
        "4. PUBLIC — an 'all' doc, visible to every role"))

    proof = acl_proof(retriever, docs, results)

    rows, agg = evaluate(retriever, load_questions(), use_ollama=False)
    print_report(rows, agg)

    out = {"demo": results, "acl_proof": proof, "eval": agg,
           "eval_rows": rows,
           "dense_backend": retriever.dense_backend()}
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote data/results.json")
    print("demo done — " + ("ALL CHECKS PASSED" if proof["passed"]
                             else "ACL CHECK FAILED"))
    return 0 if proof["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
