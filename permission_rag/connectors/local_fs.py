"""Local-filesystem connector.

Reads ``*.md`` files from a directory. Each file carries its ACL in a
tiny YAML-style frontmatter block::

    ---
    id: eng-deploy-runbook
    title: Deploy Runbook — Shipping to Production
    acl: [engineering]
    ---

Files without an ``id`` use the filename; files without an ``acl`` are
SKIPPED (fail closed — an unlabelled document is never treated as public).
"""

from __future__ import annotations

import os
import re

from ..acl import Document
from . import Connector

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _parse_frontmatter(raw: str):
    """Return (meta dict, body). Meta values are plain strings / [a, b] lists."""
    m = FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw
    meta = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = [v.strip().strip("\"'") for v in value[1:-1].split(",") if v.strip()]
        else:
            value = value.strip("\"'")
        meta[key.strip()] = value
    return meta, m.group(2)


class LocalFilesystemConnector(Connector):
    """Reads Markdown docs (+ ACL frontmatter) from a local directory."""

    name = "local-filesystem"

    def __init__(self, directory: str):
        self.directory = directory

    def fetch(self) -> list:
        docs, skipped = [], []
        for fname in sorted(os.listdir(self.directory)):
            if not fname.endswith(".md"):
                continue
            path = os.path.join(self.directory, fname)
            with open(path, encoding="utf-8") as f:
                meta, body = _parse_frontmatter(f.read())
            if "acl" not in meta:
                skipped.append(fname)  # fail closed: no ACL, no ingest
                continue
            docs.append(Document(
                doc_id=meta.get("id") or fname[:-3],
                title=meta.get("title") or fname[:-3],
                text=body.strip(),
                source=self.name,
                acl=meta["acl"],
            ))
        self.skipped_no_acl = skipped
        return docs
