"""Retrieval tests — hybrid BM25 + dense works, and spans both connectors."""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from permission_rag.acl import USERS  # noqa: E402
from permission_rag.chunking import chunk_documents  # noqa: E402
from permission_rag.connectors.local_fs import LocalFilesystemConnector  # noqa: E402
from permission_rag.connectors.mock_enterprise import MockEnterpriseConnector  # noqa: E402
from permission_rag.retrieval import (  # noqa: E402
    Bm25Retriever, DenseRetriever, HybridRetriever, PermissionAwareRetriever,
)


def build():
    docs = (LocalFilesystemConnector(os.path.join(HERE, "corpus", "local")).fetch()
            + MockEnterpriseConnector().fetch())
    return PermissionAwareRetriever(chunk_documents(docs))


class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = build()

    def test_hybrid_finds_deploy_runbook(self):
        hits = self.retriever.search("How do I deploy to production?",
                                     USERS["priya"], top_k=3)
        self.assertEqual(hits[0][1].doc_id, "eng-deploy-runbook")

    def test_hybrid_finds_expense_policy(self):
        hits = self.retriever.search("approval limit for expense claims",
                                     USERS["arjun"], top_k=3)
        self.assertEqual(hits[0][1].doc_id, "fin-expense-policy")

    def test_cross_connector_query_spans_both_sources(self):
        hits = self.retriever.search(
            "What needs finance approval — expense claims or vendor contracts?",
            USERS["arjun"], top_k=3)
        sources = {c.source for _, c in hits}
        self.assertEqual(sources, {"local-filesystem", "mock-enterprise"})

    def test_bm25_and_dense_agree_on_easy_query(self):
        chunks = self.retriever.all_chunks
        bm25, dense = Bm25Retriever(), DenseRetriever(use_ollama=False)
        for r in (bm25, dense):
            r.fit(chunks)
        q = "How many days of annual leave do employees get?"
        self.assertEqual(chunks[bm25.search(q)[0][1]].doc_id, "hr-leave-policy")
        self.assertEqual(chunks[dense.search(q)[0][1]].doc_id, "hr-leave-policy")

    def test_dense_backend_is_labelled(self):
        backend = self.retriever.dense_backend()
        self.assertIn(backend, ("hashed", "ollama:nomic-embed-text"))

    def test_rrf_hybrid_is_a_real_fusion(self):
        # Hybrid top-1 should match at least one of its parts' top-1 on an
        # easy query (sanity: fusion is not returning arbitrary order).
        chunks = self.retriever.all_chunks
        hybrid = HybridRetriever(DenseRetriever(use_ollama=False))
        hybrid.fit(chunks)
        q = "What is the Q3 budget for engineering?"
        top = chunks[hybrid.search(q)[0][1]].doc_id
        self.assertEqual(top, "fin-q3-budget")


if __name__ == "__main__":
    unittest.main()
