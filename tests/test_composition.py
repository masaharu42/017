from __future__ import annotations

import unittest

import pandas as pd

from ebsd_diffusion.composition import CompositionConditioner, parse_composition_argument


class CompositionConditionerTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            [
                {"alloy_id": 1, "conditional_ready": True, "Ni": 40, "Fe": 50, "Cr": 10, "Ta": 0},
                {"alloy_id": 2, "conditional_ready": True, "Ni": 45, "Fe": 43, "Cr": 12, "Ta": 0},
                {"alloy_id": 3, "conditional_ready": True, "Ni": 50, "Fe": 36, "Cr": 14, "Ta": 0.2},
                {"alloy_id": 4, "conditional_ready": False, "Ni": None, "Fe": None, "Cr": None, "Ta": None},
            ]
        )

    def test_closure_and_singleton_features_are_removed(self):
        conditioner = CompositionConditioner.fit(
            self.frame,
            candidate_features=["Ni", "Fe", "Cr", "Ta"],
            alloy_ids=[1, 2, 3],
            closure_drop_feature="Fe",
            min_nonzero_samples=2,
        )
        self.assertEqual(conditioner.active_features, ["Ni", "Cr"])
        self.assertEqual(conditioner.state.dropped_features["Fe"], "closure_balance_removed")
        self.assertIn("nonzero_in_only_1", conditioner.state.dropped_features["Ta"])

    def test_transform_training_center_is_zero(self):
        conditioner = CompositionConditioner.fit(
            self.frame,
            candidate_features=["Ni", "Cr"],
            alloy_ids=[1, 2, 3],
            closure_drop_feature=None,
        )
        values = conditioner.transform_frame(self.frame.iloc[:3])
        self.assertAlmostEqual(float(values.mean(axis=0).max()), 0.0, places=7)

    def test_parse_custom_composition(self):
        parsed = parse_composition_argument("Ni=45, Cr=17, Nb=3.5")
        self.assertEqual(parsed, {"Ni": 45.0, "Cr": 17.0, "Nb": 3.5})


if __name__ == "__main__":
    unittest.main()

