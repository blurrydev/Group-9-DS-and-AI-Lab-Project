"""Tests for the Hindi RAG pipeline components: retriever, compressor, generator, and prompts."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from rag.generator import MockGenerator, OpenAIGenerator, create_generator
from rag.prompts import answer_prompt
from rag.retriever import BM25Retriever


class TestRAGComponents(unittest.TestCase):
    def test_answer_prompt(self):
        question = "योजना का उद्देश्य क्या है?"
        contexts = [
            {"title": "Doc 1", "source_url": "https://gov.in/doc1", "text": "यह योजना का मुख्य उद्देश्य है।"}
        ]
        prompt = answer_prompt(question, contexts)
        self.assertIn("Doc 1", prompt)
        self.assertIn("https://gov.in/doc1", prompt)
        self.assertIn(question, prompt)
        self.assertIn("यह योजना का मुख्य उद्देश्य है।", prompt)

    def test_mock_generator(self):
        generator = MockGenerator(response_prefix="[Ans]: ")
        ans = generator.generate("some prompt")
        self.assertTrue(ans.startswith("[Ans]: "))

    def test_create_generator_mock_fallback(self):
        with patch.dict("os.environ", {}, clear=True):
            gen = create_generator(allow_mock_fallback=True)
            self.assertIsInstance(gen, MockGenerator)

            gen_none = create_generator(allow_mock_fallback=False)
            self.assertIsNone(gen_none)

    def test_create_generator_with_key(self):
        gen = create_generator(api_key="sk-test", base_url="https://api.openai.com/v1", model="gpt-4o-mini")
        self.assertIsInstance(gen, OpenAIGenerator)
        self.assertEqual(gen.api_key, "sk-test")
        self.assertEqual(gen.model, "gpt-4o-mini")

    def test_create_generator_gemini_key(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "AIzaSyFakeKey123"}, clear=True):
            gen = create_generator()
            self.assertIsInstance(gen, OpenAIGenerator)
            self.assertEqual(gen.api_key, "AIzaSyFakeKey123")
            self.assertEqual(gen.base_url, "https://generativelanguage.googleapis.com/v1beta/openai/")
            self.assertEqual(gen.model, "gemini-3.6-flash")


    def test_create_generator_groq_key(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "gsk_FakeGroqKey123"}, clear=True):
            gen = create_generator()
            self.assertIsInstance(gen, OpenAIGenerator)
            self.assertEqual(gen.api_key, "gsk_FakeGroqKey123")
            self.assertEqual(gen.base_url, "https://api.groq.com/openai/v1")
            self.assertEqual(gen.model, "llama-3.3-70b-versatile")

    def test_bm25_retriever(self):
        chunks = [
            {"text": "प्रधानमंत्री जन धन योजना एक वित्तीय समावेशन कार्यक्रम है।", "document_id": "1"},
            {"text": "मौसम आज बहुत अच्छा और धूप वाला है।", "document_id": "2"},
        ]
        retriever = BM25Retriever(chunks)
        results = retriever.search("जन धन योजना", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].record["document_id"], "1")


if __name__ == "__main__":
    unittest.main()

