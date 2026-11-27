"""Permission-aware RAG — hybrid retrieval with ACL pre-filtering.

Every document carries an ACL (allowed roles; ``all`` means every user).
Every query carries a user identity + roles. Documents the user may not
see are excluded BEFORE any ranking or scoring — the retriever never sees
them, so they cannot leak via snippets, citations, or counts.
"""

__version__ = "0.1.0"
