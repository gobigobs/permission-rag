# permission-rag — Permission-Aware Enterprise RAG

**Hybrid BM25 + dense retrieval over two connectors, with ACLs enforced at retrieval time — before ranking, so a document you may not see can never leak through a snippet, a citation, or a count.**

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![No API key](https://img.shields.io/badge/demo-no%20API%20key-orange)
![Local LLM](https://img.shields.io/badge/answers-local%20Ollama-gold)

> **Proof, not mockups:** the screenshots below are the real running demo, captured with headless Chromium from the demo's own output page (`docs/results.html`, rendered from `data/results.json`).

![Demo — allowed, denied, cross-connector queries](docs/screenshots/demo-top.png)
![Demo — full page with ACL proof and eval table](docs/screenshots/demo-full.png)

Re-capture with `python3 scripts/render_results.py && node scripts/capture_screenshot.mjs` (after running the demo).

## What it does

A company chatbot that answers questions from internal docs — but only from the docs *you* are allowed to see. Every document carries an ACL (allowed roles: `engineering`, `finance`, `hr`, `exec`, or `all`). Every query carries a user identity + roles. The ACL pre-filter runs **before** retrieval: the retriever is built on the visible subset only, so denied documents never reach the ranker, the answerer, or the citation list.

The demo seeds 12 documents from **two working connectors** and runs 4 queries:

| # | Query | User | What it proves |
|---|---|---|---|
| 1 | How do I deploy to production? | priya (engineering) | Allowed — answers from a `local-filesystem` doc, cited |
| 2 | What are the executive compensation bands for directors? | priya (engineering) | **Denied** — the exec-only doc never appears; the system abstains instead of leaking |
| 3 | What needs finance approval — expense claims or vendor contracts? | arjun (finance) | Cross-connector — answer spans `local-filesystem` **and** `mock-enterprise` |
| 4 | When is Diwali a holiday in 2026? | meera (hr) | Public — an `all` doc, visible to every role |

Then it prints the ACL enforcement proof (per-user visible doc counts + a leak scan over every answer, citation and retrieved id) and a RAGAS-style eval summary.

## Quick start — one command

```bash
./run.sh
```

No install, no API key, no model download. Pure Python 3.12 stdlib. If a local Ollama server is running with `qwen2.5:3b-instruct`, answers come from that model; otherwise a transparent extractive fallback runs (the mode is printed per answer). First run takes ~2 min warm / ~4–5 min cold on CPU while Ollama loads the model.

```bash
python3 -m unittest discover -s tests -v   # 14 tests, incl. the ACL leak tests
```

## Architecture

```
corpus/local/*.md            mock enterprise API (Drive/Notion-shaped)
  (ACL in frontmatter)         (ACL per doc in list_documents())
        │                              │
        └──────┬───────────────────────┘
               ▼
   connectors: LocalFilesystemConnector / MockEnterpriseConnector
   (a doc without an ACL is SKIPPED — fail closed, never public)
               │
               ▼
   chunking (500 chars / 100 overlap, chunks inherit the doc ACL)
               │
               ▼
   ACL PRE-FILTER  ← user + roles ─── visible chunks only
               │
               ▼
   hybrid retrieval: BM25 (Okapi) + dense, fused with RRF (k=60)
               │
               ▼
   abstention gate (content-word overlap — RRF scores can't gate)
               │
               ▼
   answering: local Ollama qwen2.5:3b-instruct, else extractive fallback
               │
               ▼
   cited answer: doc IDs only, from the visible set
```

Key files:

| File | What it is |
|---|---|
| `permission_rag/acl.py` | `User`, `Document`, `can_access`, `visible_docs` / `visible_chunks` — the pre-filter |
| `permission_rag/connectors/local_fs.py` | Local-filesystem connector (Markdown + ACL frontmatter) |
| `permission_rag/connectors/mock_enterprise.py` | Mock enterprise connector (Drive/Notion-style API, seeded docs) |
| `permission_rag/retrieval.py` | BM25 + dense (Ollama embeddings when available, else IDF-weighted hashed embeddings) + RRF hybrid, per-user index |
| `permission_rag/answer.py` | Ollama answering + extractive fallback + abstention gate |
| `permission_rag/eval.py` | RAGAS-style eval (heuristic proxies — see below) |
| `demo.py` | The demo: seeds, runs the 4 queries, prints the ACL proof + eval |
| `tests/test_acl.py`, `tests/test_retrieval.py` | 14 stdlib `unittest` tests |
| `eval/questions.json` | 10 eval questions, each asked as a specific user |

### Why the pre-filter is structural, not a scrub

`PermissionAwareRetriever.search(query, user)` builds the hybrid retriever on `visible_chunks(user, ...)` only — BM25 IDF statistics included. A denied document is not in that user's index at all, so there is no post-retrieval step that could forget to remove it, and no score, count, or snippet of it can surface. `tests/test_acl.py` proves it: the denied doc is queried directly as each unauthorised user and asserted absent from retrieval results, citations, and answer text — while the authorised exec user *does* get it.

### Dense retrieval, honestly labelled

The dense backend is **IDF-weighted hashed embeddings** by default (deterministic token-vector sums, pure stdlib — no neural encoder is downloaded). The demo prints `dense backend: hashed`, so a hashed run is never passed off as a neural-encoder run. If a local Ollama server has `nomic-embed-text`, dense retrieval uses real Ollama embeddings instead (`dense backend: ollama:nomic-embed-text`). A production deployment would swap in a real encoder such as `all-MiniLM-L6-v2`. The unweighted hashed variant was measured first and was too noisy (dense top-1 agreed with BM25 on only 3/8 probe queries); IDF weighting fixed that, and the hybrid now agrees with BM25 on the eval set.

This builds on Gopi's `rag-skills` project (hybrid BM25 + dense, RRF fusion, abstention calibration): same chunking, same RRF (k=60), same content-word-overlap abstention gate — because that project's real evals showed RRF scores carry no abstention signal (hit−control gap ≈ 0.000).

## Real results (measured 25 Nov 2026, this machine)

Demo run (`./run.sh`, exit 0, "ALL CHECKS PASSED"):

- 12 docs seeded (6 local-filesystem + 6 mock-enterprise), 16 chunks
- ACL proof: **PASS** — `exec-comp-review` never appears for priya / arjun / meera in retrieval, citations or answers; visible counts priya 5, arjun 5, meera 4, vikram (exec) 4 of 12
- Answering: `ollama:qwen2.5:3b-instruct` (local); dense backend: `hashed`
- Tests: **14/14 pass** (`python3 -m unittest discover -s tests -v`)

RAGAS-style eval, 10 questions (eval answers use the extractive fallback for determinism):

| Metric | Score |
|---|---|
| Context recall@3 | **1.000** (10/10 expected docs retrieved) |
| Mean faithfulness | **1.000** |
| Mean answer relevancy | **0.522** |

**Read these as indicative only.** Faithfulness / relevancy here are cheap heuristic proxies — content-word overlap with the retrieved context, and hashed-embedding cosine between question and answer — not real RAGAS LLM-judge scores. Faithfulness is 1.000 largely because extractive answers quote the context verbatim. A real RAGAS run needs a judge LLM, a bigger question set, and questions written by someone who doesn't know the docs.

## Limitations (read before citing)

- **Toy corpus**: 12 short seeded docs, 10 hand-written eval questions. Rankings and scores will move on a real knowledge base; the 1.000 recall is a small-corpus result, not a production claim.
- **Mock enterprise connector**: the "enterprise" source is a seeded in-code API, not a real Drive/Notion/Confluence connection — no OAuth, no live sync, no permission sync from the source system. Swapping in the real API means replacing one connector class; the ACL plumbing stays the same.
- **Hashed dense embeddings by default**: a lightweight stand-in, not a neural encoder. Real dense quality needs `nomic-embed-text` via Ollama or a sentence-transformer model.
- **Heuristic eval, not RAGAS**: faithfulness / relevancy are proxies (see above). No cross-encoder reranking in this repo (rag-skills showed it wins on accuracy but costs 25–55× latency).
- **ACL model is role-based only**: roles/groups, no per-user exceptions, no document-level ownership, no audit log, no handling of ACL changes after indexing (the index is rebuilt per run).
- **Single-process, in-memory demo**: no persistence, no vector DB, no auth on the demo itself — it assumes the caller has already authenticated the user and knows their roles.
- **Local LLM quality**: a 3B local model answers well on this tiny corpus; it is not a substitute for a larger model or for human review on sensitive content.

## Extending it

- **Real connector**: implement `Connector.fetch()` against the Drive/Notion/Confluence API (OAuth, pagination, `acl` mapped from the source's sharing settings). Fail closed: any doc without a resolvable ACL is skipped.
- **Real embeddings**: run Ollama with `nomic-embed-text`, or install `sentence-transformers` and plug `all-MiniLM-L6-v2` into `DenseRetriever`.
- **Real RAGAS**: point `permission_rag/eval.py`'s faithfulness/relevancy at an LLM judge instead of the heuristics, and grow `eval/questions.json` past 10 questions.
- **Production ACLs**: sync ACLs from the source system on every ingest, re-index on change, log every denied access attempt.

## No secrets

No API keys, tokens, or passwords anywhere in this repo. Ollama is local-only (`127.0.0.1:11434`); nothing leaves the machine.

Built by [Gopi](https://www.linkedin.com/in/rgobinath) (Senior AI Product Engineer). MIT — see [LICENSE](LICENSE).
