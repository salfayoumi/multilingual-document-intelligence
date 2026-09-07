from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from api import app, engine


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        engine.clear()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        engine.clear()

    def test_demo_to_grounded_answer_flow(self) -> None:
        load_response = self.client.post("/demo")
        self.assertEqual(load_response.status_code, 201)
        self.assertEqual(load_response.json()["documents"], 6)

        answer_response = self.client.post(
            "/ask",
            json={"query": "When should the vibration warning be escalated?", "top_k": 3},
        )
        self.assertEqual(answer_response.status_code, 200)
        payload = answer_response.json()
        self.assertTrue(payload["supported"])
        self.assertEqual(payload["citations"][0]["source"], "maintenance_en.md")
        self.assertIn("[1]", payload["answer"])

    def test_search_requires_an_index(self) -> None:
        response = self.client.post("/search", json={"query": "vibration"})
        self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main()

