"""Unit tests for min_k_for_requirement, row_at_k, selected_up_to."""

import unittest

import pandas as pd

from lgc.metrics import min_k_for_requirement, row_at_k, selected_up_to

class MinKForRequirement(unittest.TestCase):
    def setUp(self) -> None:
        self.per_k = pd.DataFrame(
            {
                "k_links": [1, 2, 3, 4, 5],
                "packet_coverage_pct": [70, 88, 95, 97, 99],
                "worst_sensor_coverage_pct": [0, 70, 85, 90, 92],
                "link_added": ["a", "b", "c", "d", "e"],
            }
        )

    def test_returns_smallest_k_that_meets_both(self) -> None:
        self.assertEqual(min_k_for_requirement(self.per_k, 90, 80), 3)

    def test_none_when_infeasible(self) -> None:
        self.assertIsNone(min_k_for_requirement(self.per_k, 100, 100))

    def test_none_when_only_one_constraint_met(self) -> None:
        self.assertEqual(min_k_for_requirement(self.per_k, 85, 75), 3)

class RowAtKAndSelectedUpTo(unittest.TestCase):
    def setUp(self) -> None:
        self.ordering = pd.DataFrame(
            {
                "k_links": [1, 2, 3],
                "link_added": ["l1", "l2", "l3"],
                "packet_coverage_pct": [10.0, 20.0, 30.0],
            }
        )

    def test_row_at_k_returns_the_matching_row(self) -> None:
        row = row_at_k(self.ordering, 2)
        self.assertEqual(row["link_added"], "l2")

    def test_selected_up_to_returns_sorted_prefix(self) -> None:
        self.assertEqual(selected_up_to(self.ordering, 2), ["l1", "l2"])

    def test_selected_up_to_full_ordering(self) -> None:
        self.assertEqual(selected_up_to(self.ordering, 5), ["l1", "l2", "l3"])

if __name__ == "__main__":
    unittest.main()
