"""Tests for the fixed evaluation harness.

`fixed_utils.evaluate` defines what "better" means for every experiment in
every session, so it is the one thing in this repo that must not drift.
"""

import unittest

import numpy as np

from fixed_utils import evaluate, rmse, train_test_split, trip_rmse

import pandas as pd


class RmseTests(unittest.TestCase):
    def test_zero_when_exact(self) -> None:
        actual = np.array([1.0, 2.0, 3.0])
        self.assertEqual(rmse(actual, actual.copy()), 0.0)

    def test_known_value(self) -> None:
        # errors of +1, -1, +1, -1 -> sqrt(mean(1,1,1,1)) == 1.0
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([2.0, 1.0, 4.0, 3.0])
        self.assertAlmostEqual(rmse(actual, predicted), 1.0)


class TripRmseTests(unittest.TestCase):
    """Trip energy is sum(rate * miles) over the links of a journey."""

    def test_hand_computed_two_journeys(self) -> None:
        # journey A: 2 links, journey B: 2 links
        journey_id = np.array(["A", "A", "B", "B"])
        miles = np.array([2.0, 3.0, 1.0, 4.0])
        actual = np.array([0.10, 0.20, 0.50, 0.25])
        predicted = np.array([0.15, 0.20, 0.50, 0.20])

        # A actual = 0.10*2 + 0.20*3 = 0.80 ; A pred = 0.15*2 + 0.20*3 = 0.90
        # B actual = 0.50*1 + 0.25*4 = 1.50 ; B pred = 0.50*1 + 0.20*4 = 1.30
        # diffs = -0.10, +0.20 -> sqrt(mean(0.01, 0.04)) = sqrt(0.025)
        expected = float(np.sqrt(0.025))
        self.assertAlmostEqual(
            trip_rmse(actual, predicted, journey_id, miles), expected
        )

    def test_zero_when_exact(self) -> None:
        journey_id = np.array(["A", "A", "B"])
        miles = np.array([1.0, 2.0, 3.0])
        actual = np.array([0.1, 0.2, 0.3])
        self.assertEqual(trip_rmse(actual, actual.copy(), journey_id, miles), 0.0)

    def test_offsetting_link_errors_cancel_within_a_trip(self) -> None:
        """The bias/variance split that motivates tracking both metrics.

        Two link errors of equal magnitude and opposite sign cancel in the
        trip total, so trip_rmse is 0 while link rmse is not.
        """
        journey_id = np.array(["A", "A"])
        miles = np.array([1.0, 1.0])
        actual = np.array([0.10, 0.20])
        predicted = np.array([0.15, 0.15])

        self.assertAlmostEqual(trip_rmse(actual, predicted, journey_id, miles), 0.0)
        self.assertGreater(rmse(actual, predicted), 0.0)

    def test_journey_order_does_not_matter(self) -> None:
        """Grouping is by id, not by adjacency — interleaved rows are fine."""
        journey_id = np.array(["A", "B", "A", "B"])
        miles = np.array([2.0, 1.0, 3.0, 4.0])
        actual = np.array([0.10, 0.50, 0.20, 0.25])
        predicted = np.array([0.15, 0.50, 0.20, 0.20])

        expected = float(np.sqrt(0.025))
        self.assertAlmostEqual(
            trip_rmse(actual, predicted, journey_id, miles), expected
        )


class EvaluateTests(unittest.TestCase):
    def test_returns_both_metrics(self) -> None:
        journey_id = np.array(["A", "A", "B", "B"])
        miles = np.array([2.0, 3.0, 1.0, 4.0])
        actual = np.array([0.10, 0.20, 0.50, 0.25])
        predicted = np.array([0.15, 0.20, 0.50, 0.20])

        result = evaluate(actual, predicted, journey_id=journey_id, miles=miles)

        self.assertEqual(set(result), {"rmse", "trip_rmse"})
        self.assertAlmostEqual(result["rmse"], rmse(actual, predicted))
        self.assertAlmostEqual(
            result["trip_rmse"], trip_rmse(actual, predicted, journey_id, miles)
        )

    def test_journey_id_and_miles_are_keyword_only(self) -> None:
        """Positional misuse must fail loudly rather than silently mis-score."""
        arr = np.array([1.0, 2.0])
        with self.assertRaises(TypeError):
            evaluate(arr, arr, arr, arr)  # type: ignore[misc]


class TrainTestSplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_total(self) -> None:
        df = pd.DataFrame({"x": range(1000)})

        train_a, test_a = train_test_split(df, test_size=0.2, random_seed=42)
        train_b, test_b = train_test_split(df, test_size=0.2, random_seed=42)

        self.assertEqual(len(train_a) + len(test_a), len(df))
        self.assertEqual(list(test_a["x"]), list(test_b["x"]))
        self.assertEqual(list(train_a["x"]), list(train_b["x"]))

    def test_split_is_disjoint(self) -> None:
        df = pd.DataFrame({"x": range(1000)})
        train, test = train_test_split(df, test_size=0.2, random_seed=42)
        self.assertEqual(set(train["x"]) & set(test["x"]), set())


if __name__ == "__main__":
    unittest.main()
