import unittest

import pandas as pd

from src.ncaa_prediction.data_loader import parse_submission_ids, validate_submission_shape


class SubmissionValidationTests(unittest.TestCase):
    def test_parse_submission_ids(self):
        frame = pd.DataFrame({"ID": ["2026_1101_1102"], "Pred": [0.5]})
        parsed = parse_submission_ids(frame)

        self.assertEqual(parsed.loc[0, "Season"], 2026)
        self.assertEqual(parsed.loc[0, "TeamA"], 1101)
        self.assertEqual(parsed.loc[0, "TeamB"], 1102)

    def test_validate_submission_shape(self):
        sample = pd.DataFrame({"ID": ["2026_1101_1102"], "Pred": [0.5]})
        predictions = pd.DataFrame({"ID": ["2026_1101_1102"], "Pred": [0.6]})

        result = validate_submission_shape(sample, predictions)
        self.assertEqual(result["missing"], 0)
        self.assertEqual(result["extra"], 0)


if __name__ == "__main__":
    unittest.main()
