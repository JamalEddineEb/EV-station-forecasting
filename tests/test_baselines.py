import tempfile
import unittest

import numpy as np

from src.model_selection.baselines import (
    make_baseline_predictions,
    save_baseline_predictions,
)


class BaselinePredictionTests(unittest.TestCase):
    def test_baseline_archives_match_prediction_schema(self):
        val_history = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        test_history = np.array([[9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float32)
        y_val = np.zeros((2, 3), dtype=np.float32)
        y_test = np.ones((2, 3), dtype=np.float32)

        predictions = make_baseline_predictions(val_history, test_history, 3)

        np.testing.assert_array_equal(
            predictions["persistence"][0], [[4, 4, 4], [8, 8, 8]]
        )
        np.testing.assert_array_equal(
            predictions["seasonal_naive"][1], [[9, 10, 11], [13, 14, 15]]
        )

        with tempfile.TemporaryDirectory() as outdir:
            save_baseline_predictions(predictions, y_val, y_test, outdir)
            for name, (val_pred, test_pred) in predictions.items():
                with np.load(
                    "{}/{}_predictions.npz".format(outdir, name),
                    allow_pickle=False,
                ) as archive:
                    self.assertEqual(
                        set(archive.files),
                        {"val_pred", "test_pred", "y_val", "y_test"},
                    )
                    np.testing.assert_array_equal(archive["val_pred"], val_pred)
                    np.testing.assert_array_equal(archive["test_pred"], test_pred)
                    np.testing.assert_array_equal(archive["y_val"], y_val)
                    np.testing.assert_array_equal(archive["y_test"], y_test)


if __name__ == "__main__":
    unittest.main()
