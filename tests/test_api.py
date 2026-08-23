"""API endpoint tests for the Hindi RAG API."""

import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from rag.api import app
from rag.retriever import RetrievedChunk
from rag.compressor import CompressionResult


class TestRAGAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)
        app.state.retriever = MagicMock()
        app.state.compressor = MagicMock()
        app.state.generator = MagicMock()

        # Mock retrieval response
        app.state.retriever.search.return_value = [
            RetrievedChunk(
                record={
                    "chunk_id": "doc1::chunk-0001",
                    "title": "PMJDY Scheme",
                    "source_url": "https://pmjdy.gov.in",
                    "text": "प्रधानमंत्री जन धन योजना में 2 लाख का दुर्घटना बीमा है।",
                },
                score=0.88,
            )
        ]

        # Mock compression response
        app.state.compressor.compress.return_value = CompressionResult(
            text="योजना 2 लाख दुर्घटना बीमा",
            retained_tokens=4,
            input_tokens=9,
            retention_ratio=0.44,
            used_fallback=False,
        )

        # Mock generator response
        app.state.generator.generate.return_value = "प्रधानमंत्री जन धन योजना में 2 लाख रुपये का दुर्घटना बीमा लाभ मिलता है। [Source 1]"

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("generator", data)

    def test_rag_query_with_answer_generation(self):
        payload = {
            "question": "प्रधानमंत्री जन धन योजना में बीमा लाभ क्या है?",
            "top_k": 1,
            "generate": True,
        }
        response = self.client.post("/v1/rag/query", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["question"], payload["question"])
        self.assertEqual(len(data["contexts"]), 1)
        self.assertIn("prompt", data)
        self.assertEqual(
            data["answer"],
            "प्रधानमंत्री जन धन योजना में 2 लाख रुपये का दुर्घटना बीमा लाभ मिलता है। [Source 1]",
        )

    def test_rag_query_without_answer_generation(self):
        payload = {
            "question": "प्रधानमंत्री जन धन योजना में बीमा लाभ क्या है?",
            "top_k": 1,
            "generate": False,
        }
        response = self.client.post("/v1/rag/query", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsNone(data["answer"])


if __name__ == "__main__":
    unittest.main()
