"""Connectors: where documents come from.

Every connector returns :class:`~permission_rag.acl.Document` objects with
an ACL attached at ingest time. A connector that cannot prove a document's
ACL must not return the document at all — fail closed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Connector(ABC):
    """A document source. ``fetch()`` returns all documents it can see."""

    name: str = "connector"

    @abstractmethod
    def fetch(self) -> list:
        raise NotImplementedError
