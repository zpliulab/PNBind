from pathlib import Path
import unittest

from pnbind.evaluation import run_config


ROOT = Path(__file__).resolve().parents[1]


class EvaluationTest(unittest.TestCase):
    def test_table2_pnbind_rows(self):
        results, alignment = run_config(ROOT / "benchmarks/config.json")
        expected = {
            "DNA_Test_129": (129, 37515, 0.5860, 0.9525),
            "DNA_Test_181": (181, 75088, 0.4252, 0.9198),
            "RNA_Test_117": (117, 37345, 0.3512, 0.8759),
            "RNA_Test_285": (285, 45317, 0.4161, 0.8679),
        }
        for result in results:
            n_chains, n_residues, mcc, auc = expected[result.dataset]
            self.assertEqual(result.n_chains, n_chains)
            self.assertEqual(result.n_residues, n_residues)
            self.assertEqual(round(result.mcc, 4), mcc)
            self.assertEqual(round(result.auc, 4), auc)
        self.assertEqual(len(alignment), 22)


if __name__ == "__main__":
    unittest.main()
