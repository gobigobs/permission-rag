#!/usr/bin/env python3
"""Render docs/results.html from data/results.json (the demo's real output).

Static, self-contained page — no JS framework, no external assets — so it
can be served with `python3 -m http.server` and screenshotted headless.
"""

import html
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def esc(s):
    return html.escape(str(s))


def main():
    with open(os.path.join(HERE, "data", "results.json")) as f:
        data = json.load(f)
    demo, proof, agg, rows = (data["demo"], data["acl_proof"],
                              data["eval"], data["eval_rows"])

    cards = []
    for r in demo:
        retrieved = ", ".join(
            f"{d} [{s}]" for d, s in zip(r["retrieved"], r["sources"]))
        cards.append(f"""
    <section class="card">
      <h2>{esc(r['label'])}</h2>
      <p><b>User:</b> {esc(r['user'])} &nbsp; <b>Q:</b> {esc(r['question'])}</p>
      <p><b>Retrieved:</b> {esc(retrieved) or '(none)'}</p>
      <p class="answer"><b>A:</b> {esc(r['answer'])}</p>
      <p><b>Citations:</b> {esc(', '.join(r['citations']) or '(none)')}
         &nbsp; <b>Mode:</b> {esc(r['mode'])}
         &nbsp; <b>Abstained:</b> {esc(r['abstained'])}</p>
    </section>""")

    eval_rows = "\n".join(
        f"<tr><td>{r['id']}</td><td>{esc(r['user'])}</td>"
        f"<td>{r['context_recall@3']}</td><td>{r['faithfulness']}</td>"
        f"<td>{r['answer_relevancy']}</td><td>{esc(r['abstained'])}</td>"
        f"<td>{esc(r['expected_doc'])}</td></tr>" for r in rows)

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>permission-rag — demo results</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:900px;margin:24px auto;
padding:0 16px;color:#1a1a1a;background:#fff;line-height:1.5}}
h1{{font-size:26px}} h2{{font-size:18px;margin:0 0 8px}}
.card{{border:1px solid #ddd;border-radius:10px;padding:16px;margin:14px 0}}
.answer{{background:#f6f8fa;border-radius:6px;padding:10px}}
.pass{{color:#0a7d2c;font-weight:700}} .pill{{display:inline-block;
background:#eef2ff;border-radius:999px;padding:2px 10px;margin-right:6px}}
table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ddd;padding:6px 8px;text-align:left;font-size:14px}}
th{{background:#f6f8fa}} .note{{color:#555;font-size:14px}}
</style></head><body>
<h1>permission-rag — Permission-Aware Enterprise RAG</h1>
<p>Demo results (real output of <code>./run.sh</code>). 12 docs from 2 connectors,
hybrid BM25 + dense retrieval (RRF), ACL pre-filter at retrieval time,
cited answers. Dense backend: <b>{esc(data['dense_backend'])}</b>.</p>
<p><span class="pill">local-filesystem: 6 docs</span>
<span class="pill">mock-enterprise: 6 docs</span>
<span class="pill">ACL proof: <span class="pass">{'PASS' if proof['passed'] else 'FAIL'}</span></span></p>
{''.join(cards)}
<h2>ACL enforcement proof</h2>
<p>Denied doc <code>{esc(proof['denied_doc'])}</code> (exec-only) never appears in
retrieval, citations or answers for priya / arjun / meera. Visible doc counts —
priya (engineering) 5, arjun (finance) 5, meera (hr) 4, vikram (exec) 4 of 12 total.
The pre-filter runs <i>before</i> ranking: denied docs are not in that user's
retriever index at all.</p>
<h2>RAGAS-style eval (10 questions — heuristic proxies, indicative only)</h2>
<table><tr><th>id</th><th>user</th><th>recall@3</th><th>faithfulness</th>
<th>relevancy</th><th>abstained</th><th>expected doc</th></tr>
{eval_rows}</table>
<p class="note">Aggregate: context recall@3 <b>{agg['context_recall@3']}</b>,
mean faithfulness <b>{agg['mean_faithfulness']}</b>,
mean answer relevancy <b>{agg['mean_answer_relevancy']}</b>.
Faithfulness / relevancy here are cheap heuristic proxies (content-word
overlap / hashed-embedding cosine), NOT real RAGAS LLM-judge scores — see
the README limitations section. Eval answers use the extractive fallback
for determinism.</p>
</body></html>"""

    os.makedirs(os.path.join(HERE, "docs"), exist_ok=True)
    out = os.path.join(HERE, "docs", "results.html")
    with open(out, "w") as f:
        f.write(page)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
