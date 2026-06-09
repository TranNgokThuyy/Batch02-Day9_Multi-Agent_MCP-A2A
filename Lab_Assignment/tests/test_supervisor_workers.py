import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))


class TestSupervisorWorkers(unittest.TestCase):
    def test_supervisor_returns_expected_shape(self):
        from src.supervisor_workers import supervisor_answer

        sources = [
            {
                "content": "Nguon hop le ve phong chong ma tuy.",
                "score": 0.9,
                "source": "hybrid",
                "metadata": {"source": "law.md", "year": "2021"},
            }
        ]

        with (
            patch("src.supervisor_workers.retrieve", return_value=sources),
            patch(
                "src.supervisor_workers.generate_with_citation",
                return_value={"answer": "Cau tra loi [law.md, 2021]", "sources": sources},
            ),
        ):
            result = supervisor_answer("Hinh phat tang tru ma tuy?", top_k=1)

        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertIn("sources", result)
        self.assertIn("trace", result)
        self.assertEqual(len(result["sources"]), 1)

    def test_empty_query_does_not_crash(self):
        from src.supervisor_workers import supervisor_answer

        result = supervisor_answer("", top_k=5)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["answer"], "I cannot verify this information")
        self.assertEqual(result["sources"], [])

    def test_uses_three_named_worker_steps(self):
        from src.supervisor_workers import supervisor_answer

        result = supervisor_answer("", top_k=5)
        steps = result["trace"]["worker_steps"]
        worker_names = {step["worker"] for step in steps}

        self.assertGreaterEqual(len(steps), 3)
        self.assertIn("retrieval_worker", worker_names)
        self.assertIn("evidence_worker", worker_names)
        self.assertIn("generation_worker", worker_names)

    def test_respects_top_k(self):
        from src.supervisor_workers import supervisor_answer

        sources = [
            {"content": f"Source {i}", "score": 1.0, "source": "hybrid", "metadata": {}}
            for i in range(5)
        ]

        with (
            patch("src.supervisor_workers.retrieve", return_value=sources),
            patch(
                "src.supervisor_workers.generate_with_citation",
                return_value={"answer": "ok", "sources": sources},
            ),
        ):
            result = supervisor_answer("query", top_k=2)

        self.assertLessEqual(len(result["sources"]), 2)
        self.assertEqual(result["trace"]["source_count"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
