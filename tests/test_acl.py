"""ACL tests — the load-bearing guarantee of this repo.

A denied document must NEVER appear for an unauthorised user: not in
retrieval results, not in citations, not in answer text. The pre-filter
runs before ranking, so the proof is structural, not a post-hoc scrub.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from permission_rag.acl import USERS, visible_docs  # noqa: E402
from permission_rag.answer import answer  # noqa: E402
from permission_rag.chunking import chunk_documents  # noqa: E402
from permission_rag.connectors.local_fs import LocalFilesystemConnector  # noqa: E402
from permission_rag.connectors.mock_enterprise import MockEnterpriseConnector  # noqa: E402
from permission_rag.retrieval import PermissionAwareRetriever  # noqa: E402

DENIED_DOC = "exec-comp-review"
PROBE = "What are the executive compensation bands for directors?"


def build():
    docs = (LocalFilesystemConnector(os.path.join(HERE, "corpus", "local")).fetch()
            + MockEnterpriseConnector().fetch())
    return docs, PermissionAwareRetriever(chunk_documents(docs))


class TestAclPreFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs, cls.retriever = build()

    def test_corpus_has_both_connectors(self):
        sources = {d.source for d in self.docs}
        self.assertEqual(sources, {"local-filesystem", "mock-enterprise"})
        self.assertEqual(len(self.docs), 12)

    def test_denied_doc_invisible_to_unauthorised_users(self):
        for name in ("priya", "arjun", "meera"):
            visible = {d.doc_id for d in visible_docs(USERS[name], self.docs)}
            self.assertNotIn(DENIED_DOC, visible, name)

    def test_denied_doc_visible_to_exec(self):
        visible = {d.doc_id for d in visible_docs(USERS["vikram"], self.docs)}
        self.assertIn(DENIED_DOC, visible)

    def test_denied_doc_never_retrieved(self):
        """The core test: query for the denied doc as each unauthorised user."""
        for name in ("priya", "arjun", "meera"):
            hits = self.retriever.search(PROBE, USERS[name], top_k=5)
            got = [c.doc_id for _, c in hits]
            self.assertNotIn(DENIED_DOC, got, f"{name} retrieved {got}")

    def test_denied_doc_never_in_answer_or_citations(self):
        for name in ("priya", "arjun", "meera"):
            hits = self.retriever.search(PROBE, USERS[name], top_k=3)
            result = answer(PROBE, hits, use_ollama=False)
            self.assertNotIn(DENIED_DOC, result["citations"])
            self.assertNotIn(DENIED_DOC, result["answer"])
            # Confidential numbers from the denied doc must not surface either.
            self.assertNotIn("Rs 90 lakh", result["answer"])

    def test_denied_query_abstains_for_unauthorised_user(self):
        hits = self.retriever.search(PROBE, USERS["priya"], top_k=3)
        result = answer(PROBE, hits, use_ollama=False)
        self.assertTrue(result["abstained"])

    def test_allowed_user_gets_denied_doc(self):
        hits = self.retriever.search(PROBE, USERS["vikram"], top_k=3)
        got = [c.doc_id for _, c in hits]
        self.assertIn(DENIED_DOC, got)
        result = answer(PROBE, hits, use_ollama=False)
        self.assertIn(DENIED_DOC, result["citations"])

    def test_visible_counts(self):
        expected = {"priya": 5, "arjun": 5, "meera": 4, "vikram": 4}
        for name, n in expected.items():
            self.assertEqual(
                len(visible_docs(USERS[name], self.docs)), n, name)


if __name__ == "__main__":
    unittest.main()
