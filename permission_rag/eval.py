"""RAGAS-style eval, run locally — heuristic scores, labelled indicative.

For each question (asked AS a specific user, so ACLs apply):
- context recall@k: is the expected doc in the retrieved top-k?
- faithfulness: fraction of the answer's content words that appear in the
  retrieved context (a cheap stand-in for RAGAS faithfulness, which
  normally uses an LLM judge to check each claim against the context).
- answer relevancy: cosine similarity between the question and the answer
  under the hashed embedding (a stand-in for RAGAS answer relevancy, which
  normally generates questions from the answer and compares embeddings).

These are HEURISTIC proxies on a 10-question toy set — they show the
harness works, they are NOT production RAGAS numbers. The demo and README
say so in plain language. A real RAGAS run needs a judge LLM and a bigger
question set.
"""

from __future__ import annotations

import json
import os

from .acl import USERS
from .answer import answer, content_words
from .retrieval import hashed_embed, _cosine_sparse

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_questions(path=None):
    path = path or os.path.join(HERE, "eval", "questions.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["questions"]


def faithfulness(answer_text: str, context_text: str) -> float:
    """Share of the answer's content words supported by the context."""
    aw = content_words(answer_text)
    if not aw:
        return 0.0
    ctx = set()
    from .retrieval import tokenize
    ctx.update(tokenize(context_text))
    return len(aw & ctx) / len(aw)


def answer_relevancy(question: str, answer_text: str) -> float:
    """Hashed-embedding cosine between question and answer (0..1)."""
    return max(0.0, _cosine_sparse(hashed_embed(question),
                                   hashed_embed(answer_text)))


def evaluate(retriever, questions, top_k=3, use_ollama=False):
    """Run the eval set. ``use_ollama=False`` keeps eval fast + deterministic.

    Returns (per-question rows, aggregate dict).
    """
    rows = []
    for q in questions:
        user = USERS[q["user"]]
        hits = retriever.search(q["question"], user, top_k=top_k)
        got_ids = [c.doc_id for _, c in hits]
        result = answer(q["question"], hits, use_ollama=use_ollama)
        context = " ".join(c.text for _, c in hits)
        row = {
            "id": q["id"],
            "user": q["user"],
            "question": q["question"],
            "expected_doc": q["expected_doc"],
            "retrieved_docs": got_ids,
            "context_recall@3": 1 if q["expected_doc"] in got_ids else 0,
            "faithfulness": round(
                0.0 if result["abstained"] else
                faithfulness(result["answer"], context), 3),
            "answer_relevancy": round(
                0.0 if result["abstained"] else
                answer_relevancy(q["question"], result["answer"]), 3),
            "abstained": result["abstained"],
            "answer": result["answer"],
        }
        rows.append(row)
    scored = [r for r in rows if not r["abstained"]]
    agg = {
        "n_questions": len(rows),
        "n_scored": len(scored),
        "n_abstained": len(rows) - len(scored),
        "context_recall@3": round(
            sum(r["context_recall@3"] for r in rows) / len(rows), 3)
        if rows else 0.0,
        "mean_faithfulness": round(
            sum(r["faithfulness"] for r in scored) / len(scored), 3)
        if scored else 0.0,
        "mean_answer_relevancy": round(
            sum(r["answer_relevancy"] for r in scored) / len(scored), 3)
        if scored else 0.0,
    }
    return rows, agg


def print_report(rows, agg):
    print("\n== RAGAS-style eval (heuristic proxies — indicative only) ==")
    print(f"{'id':<4} {'user':<7} {'recall@3':<9} {'faith.':<7} "
          f"{'relev.':<7} {'abst.':<6} expected doc")
    print("-" * 72)
    for r in rows:
        print(f"{r['id']:<4} {r['user']:<7} {r['context_recall@3']:<9} "
              f"{r['faithfulness']:<7} {r['answer_relevancy']:<7} "
              f"{str(r['abstained']):<6} {r['expected_doc']}")
    print("-" * 72)
    print(f"questions: {agg['n_questions']} (scored {agg['n_scored']}, "
          f"abstained {agg['n_abstained']})")
    print(f"context recall@3:       {agg['context_recall@3']:.3f}")
    print(f"mean faithfulness:      {agg['mean_faithfulness']:.3f} "
          f"(heuristic proxy — see README)")
    print(f"mean answer relevancy:  {agg['mean_answer_relevancy']:.3f} "
          f"(heuristic proxy — see README)")
