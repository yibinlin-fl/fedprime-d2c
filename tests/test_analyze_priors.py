from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_priors import plot_round_curves, write_round_summary


class AnalyzePriorsTests(unittest.TestCase):
    def test_round_summary_and_plot_are_exported(self):
        rows = []
        for round_idx in (3, 4):
            for client_id in (0, 1):
                rows.append(
                    {
                        "round": str(round_idx),
                        "client": str(client_id),
                        "prior_l1": "0.4",
                        "prior_kl": "0.2",
                        "prior_cosine_similarity": "0.8",
                        "predicted_entropy": "2.0",
                        "oracle_entropy": "1.0",
                        "predicted_normalized_entropy": "0.9",
                        "oracle_normalized_entropy": "0.5",
                        "top_class_match": "1.0",
                    }
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            summary_path = output_dir / "summary.csv"
            plot_path = output_dir / "curves.png"
            write_round_summary(rows, summary_path)
            plot_round_curves(rows, plot_path)
            self.assertTrue(summary_path.exists())
            self.assertTrue(plot_path.exists())
            with summary_path.open(newline="", encoding="utf-8") as f:
                summary_rows = list(csv.DictReader(f))
            self.assertEqual(len(summary_rows), 2)
            self.assertEqual(summary_rows[0]["records"], "2")


if __name__ == "__main__":
    unittest.main()
