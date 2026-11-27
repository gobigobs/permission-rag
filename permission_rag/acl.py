"""Access control: users, documents, and the pre-filter.

The rule is deliberately boring and total: a document is visible to a user
iff ``"all"`` is in the document's ACL or the user's roles intersect the
ACL. There is no post-filtering step anywhere in the pipeline — callers
must call :func:`visible_chunks` (or :func:`visible_docs`) *before*
retrieval, and the retriever only ever receives the visible subset.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class User:
    """A querying user: an identity plus the roles/groups they belong to."""

    name: str
    roles: frozenset = frozenset()

    def __init__(self, name: str, roles):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "roles", frozenset(r.lower() for r in roles))


@dataclass
class Document:
    """One retrievable document, with its ACL attached at ingest time."""

    doc_id: str
    title: str
    text: str
    source: str  # connector name, e.g. "local-filesystem" / "mock-enterprise"
    acl: frozenset = frozenset({"all"})

    def __init__(self, doc_id, title, text, source, acl=("all",)):
        self.doc_id = doc_id
        self.title = title
        self.text = text
        self.source = source
        self.acl = frozenset(a.lower() for a in acl)


@dataclass
class Chunk:
    """A chunk of a document; inherits the document's ACL and identity."""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    source: str
    acl: frozenset = field(default_factory=lambda: frozenset({"all"}))


def can_access(user: User, doc: Document) -> bool:
    """True iff the user may see this document."""
    return "all" in doc.acl or bool(user.roles & doc.acl)


def visible_docs(user: User, docs) -> list:
    """ACL pre-filter: return only the documents this user may see.

    This runs BEFORE retrieval. Anything not returned here is invisible to
    the rest of the pipeline — it cannot be ranked, cited, or counted.
    """
    return [d for d in docs if can_access(user, d)]


def visible_chunks(user: User, chunks) -> list:
    """ACL pre-filter over chunks (same rule, chunk inherits its doc's ACL)."""
    return [c for c in chunks if "all" in c.acl or bool(user.roles & c.acl)]


# Demo users — one per role, so the demo can show allow / deny clearly.
USERS = {
    "priya": User("priya", ["engineering"]),
    "arjun": User("arjun", ["finance"]),
    "meera": User("meera", ["hr"]),
    "vikram": User("vikram", ["exec"]),
}
