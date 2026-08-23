"""Tests for the main dashboard web app and /api/predict endpoint."""

import unittest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.main import app, inference_service
from rag.compressor import CompressionResult


class TestAppMain(unittest.TestCase):
    def setUp(self):
        inference_service.compressor = MagicMock()
        inference_service.generator = MagicMock()
        inference_service.state.ready = True
        inference_service.state.error = None

        inference_service.compressor.compress.return_value = CompressionResult(
            text="आवासीय प्रशासकीय इमारते दौलतखाना",
            retained_tokens=4,
            input_tokens=15,
            retention_ratio=0.2667,
            used_fallback=False,
        )
        inference_service.generator.generate.return_value = "दौलतखाना [Source 1]"

        self.client = TestClient(app, raise_server_exceptions=False)

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")

    def test_runs(self):
        response = self.client.get("/api/runs")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json().get("runs"), list)

    def test_predict_endpoint_custom_context(self):
        payload = {
            "question": "मुग़ल काल में आवासीय और प्रशासनिक भवन को क्या कहा जाता था?",
            "context": "भारत की सबसे बड़ी सामूहिक मस्जिद है, साथ ही आवासीय तथा प्रशासकीय इमारते हैं जिसे दौलतखाना कहते हैं।",
            "generate_answer": True,
            "retrieve_rag": False,
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("compressed_text", data)
        self.assertIn("stats", data)
        self.assertIn("answer", data)
        self.assertIn("prompt", data)
        self.assertEqual(data["answer"], "दौलतखाना [Source 1]")


if __name__ == "__main__":
    unittest.main()

