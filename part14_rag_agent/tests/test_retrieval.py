import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import retriever
from chunking import recursive_chunk


class RetrievalTests(unittest.TestCase):
    def test_chunking_retains_all_non_whitespace_text(self):
        source = ("First paragraph with useful policy information.\n\n" * 18)
        chunks = recursive_chunk(source, chunk_size=80, overlap=12)
        joined = " ".join(chunks)
        for token in source.split():
            self.assertIn(token, joined)

    @patch.object(retriever, "search_chunks", return_value=[])
    @patch.object(retriever, "get_embedding", return_value=[0.0] * 768)
    def test_no_results(self, *_):
        self.assertEqual(retriever.retrieve_chunks("x").status, retriever.STATUS_NO_RESULTS)

    @patch.object(retriever, "search_chunks", return_value=[{"similarity": .2, "metadata": {}, "document_id": "doc"}])
    @patch.object(retriever, "get_embedding", return_value=[0.0] * 768)
    def test_low_relevance(self, *_):
        self.assertEqual(retriever.retrieve_chunks("x", min_similarity=.5).status, retriever.STATUS_LOW_RELEVANCE)

    @patch.object(retriever, "search_chunks", return_value=[{
        "similarity": .8, "metadata": {"policy_status": "current"},
        "document_id": "employee_leave_policy", "content": "Employees receive 20 paid leave days per calendar year.",
    }])
    @patch.object(retriever, "get_embedding", return_value=[0.0] * 768)
    def test_high_similarity_does_not_override_missing_subject_terms(self, *_):
        result = retriever.retrieve_chunks("What is our maternity leave policy?")
        self.assertEqual(result.status, retriever.STATUS_LOW_RELEVANCE)

    @patch.object(retriever, "search_chunks", return_value=[{
        "similarity": .8, "metadata": {"policy_status": "current"},
        "document_id": "employee_leave_policy", "content": "Employees receive 20 paid leave days per calendar year.",
    }])
    @patch.object(retriever, "get_embedding", return_value=[0.0] * 768)
    def test_high_similarity_does_not_override_missing_subject_terms(self, *_):
        result = retriever.retrieve_chunks("What is our maternity leave policy?")
        self.assertEqual(result.status, retriever.STATUS_LOW_RELEVANCE)

    @patch.object(retriever, "get_embedding", side_effect=RuntimeError("offline"))
    def test_embedding_error_is_reported(self, _):
        result = retriever.retrieve_chunks("x")
        self.assertEqual(result.status, retriever.STATUS_ERROR)
        self.assertIn("offline", result.error_message)


if __name__ == "__main__":
    unittest.main()
