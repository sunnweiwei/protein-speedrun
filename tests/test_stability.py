import unittest

from speedrun.stability import (
    confirmed_time_to_target,
    summarize_official_results,
    summarize_seed_results,
)
from speedrun.calibrate import recommend_target


class StableTargetTest(unittest.TestCase):
    def test_requires_consecutive_checkpoints(self):
        history = [
            {"step": 1, "contact_p_at_l": 0.31, "training_seconds": 1.0},
            {"step": 2, "contact_p_at_l": 0.29, "training_seconds": 2.0},
            {"step": 3, "contact_p_at_l": 0.32, "training_seconds": 3.0},
            {"step": 4, "contact_p_at_l": 0.33, "training_seconds": 4.0},
        ]
        self.assertEqual(
            confirmed_time_to_target(
                history, target=0.30, consecutive_passes=2
            ),
            4.0,
        )

    def test_calibration_has_no_target_time(self):
        self.assertIsNone(
            confirmed_time_to_target(
                [
                    {
                        "step": 1,
                        "contact_p_at_l": 1.0,
                        "training_seconds": 1.0,
                    }
                ],
                target=None,
                consecutive_passes=2,
            )
        )

    def test_multi_seed_record_is_one_median_number(self):
        results = [
            {"seed": 42, "confirmed_seconds_to_target": 10.0},
            {"seed": 66, "confirmed_seconds_to_target": 11.0},
            {"seed": 101, "confirmed_seconds_to_target": 12.0},
            {"seed": 2024, "confirmed_seconds_to_target": 13.0},
            {"seed": 8888, "confirmed_seconds_to_target": None},
        ]
        summary = summarize_seed_results(
            results,
            official_seeds=[42, 66, 101, 2024, 8888],
            required_seed_passes=4,
        )
        self.assertTrue(summary["qualified"])
        self.assertEqual(summary["median_confirmed_seconds_to_target"], 11.5)

    def test_official_record_refuses_mixed_candidates(self):
        protocol = {
            "target": {"value": 0.3},
            "stability": {
                "consecutive_checkpoint_passes": 2,
                "official_seeds": [42, 66],
                "required_seed_passes": 2,
            },
            "hardware": {
                "accelerators_per_run": 1,
                "accelerator_name_contains": "H100 80GB",
            },
        }
        base = {
            "objective": "mlm",
            "corpus_sha256": "corpus",
            "target_contact_p_at_l": 0.3,
            "consecutive_checkpoint_passes": 2,
            "protocol_sha256": "protocol",
            "candidate_config_sha256": "config",
            "train_code_sha256": "train",
            "model_code_sha256": "model",
            "confirmed_seconds_to_target": 10.0,
            "hardware": {
                "accelerator_count": 1,
                "accelerator_name": "NVIDIA H100 80GB HBM3",
            },
        }
        results = [
            {**base, "candidate_id": "a", "seed": 42},
            {**base, "candidate_id": "b", "seed": 66},
        ]
        with self.assertRaises(ValueError):
            summarize_official_results(results, protocol)

    def test_calibration_requires_margin_and_consecutive_crossing(self):
        histories = {
            42: [
                {"step": 0, "contact_p_at_l": 0.20, "training_seconds": 0.0},
                {"step": 1, "contact_p_at_l": 0.25, "training_seconds": 1.0},
                {"step": 2, "contact_p_at_l": 0.26, "training_seconds": 2.0},
                {"step": 3, "contact_p_at_l": 0.27, "training_seconds": 3.0},
            ],
            66: [
                {"step": 0, "contact_p_at_l": 0.21, "training_seconds": 0.0},
                {"step": 1, "contact_p_at_l": 0.24, "training_seconds": 1.1},
                {"step": 2, "contact_p_at_l": 0.25, "training_seconds": 2.1},
                {"step": 3, "contact_p_at_l": 0.26, "training_seconds": 3.1},
            ],
        }
        recommendation = recommend_target(
            histories,
            consecutive_passes=2,
            initial_margin=0.01,
            final_margin=0.01,
            quantum=0.01,
        )
        self.assertEqual(
            recommendation["recommended_target_contact_p_at_l"], 0.25
        )


if __name__ == "__main__":
    unittest.main()
