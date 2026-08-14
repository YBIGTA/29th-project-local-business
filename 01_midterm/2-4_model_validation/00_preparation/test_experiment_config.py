import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_config import EXPECTED_SPLIT_COUNTS, split_2_4


class ExperimentConfigTest(unittest.TestCase):
    def test_split_boundaries(self):
        months = ["2021-12", "2022-01", "2022-08", "2022-09", "2023-02"]
        expected = ["train", "valid", "valid", "test", "test"]
        self.assertEqual([split_2_4(month) for month in months], expected)

    def test_split_total(self):
        self.assertEqual(sum(EXPECTED_SPLIT_COUNTS.values()), 68_087)

    def test_out_of_range_month_is_rejected(self):
        with self.assertRaises(ValueError):
            split_2_4("2023-03")


if __name__ == "__main__":
    unittest.main()
