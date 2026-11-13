"""Opt-in live checks: RUN_INTEGRATION=1 python -m unittest tests.test_integrations"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@unittest.skipUnless(os.getenv("RUN_INTEGRATION") == "1", "requires local Ollama and PostgreSQL")
class LocalServiceIntegrationTests(unittest.TestCase):
    def test_embedding_and_database_schema(self):
        from embedding_client import get_embedding
        from db import verify_schema
        self.assertEqual(len(get_embedding("integration probe")), 768)
        verify_schema()

    def test_retrieval_round_trip(self):
        from retriever import retrieve_chunks
        result = retrieve_chunks("How much annual leave do employees receive?")
        self.assertEqual(result.status, "relevant", result.error_message)
        self.assertTrue(any(c["document_id"] == "employee_leave_policy" for c in result.chunks))
        self.assertTrue(all(c["similarity"] == 1 - c["distance"] for c in result.chunks))

    def test_irrelevant_query_is_filtered(self):
        from retriever import retrieve_chunks, STATUS_LOW_RELEVANCE
        result = retrieve_chunks("What is the capital of Japan?")
        self.assertEqual(result.status, STATUS_LOW_RELEVANCE)

    def test_missing_policy_subject_is_filtered(self):
        from retriever import retrieve_chunks, STATUS_LOW_RELEVANCE
        result = retrieve_chunks("What is our maternity leave policy?")
        self.assertEqual(result.status, STATUS_LOW_RELEVANCE)

    def test_missing_policy_subject_is_filtered(self):
        from retriever import retrieve_chunks, STATUS_LOW_RELEVANCE
        result = retrieve_chunks("What is our maternity leave policy?")
        self.assertEqual(result.status, STATUS_LOW_RELEVANCE)

    def test_current_policy_wins(self):
        from retriever import retrieve_chunks
        result = retrieve_chunks("What is the annual paid leave allowance?")
        self.assertTrue(result.chunks)
        self.assertTrue(all(c["document_id"] != "employee_leave_policy_old" for c in result.chunks))
        self.assertTrue(any((c["metadata"] or {}).get("policy_status") == "current" for c in result.chunks))


if __name__ == "__main__":
    unittest.main()
