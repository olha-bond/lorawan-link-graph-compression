"""Consistency checks for the greedy solvers on a small synthetic instance."""

import unittest

import numpy as np

from lgc.greedy.fairness_aware import fairness_aware_greedy_generic
from lgc.greedy.multicover import multicover_greedy

class GreedyOnTinyInstance(unittest.TestCase):
    """3 sensors, 6 links, 6 packets. Each packet covered by two links."""

    def setUp(self) -> None:
        self.packet_matrix = np.array(
            [
                [1, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 0],
            ],
            dtype=bool,
        )
        self.packet_sensor = np.array(["s1", "s1", "s2", "s2", "s3", "s3"])
        self.sensor_total = {"s1": 2, "s2": 2, "s3": 2}
        self.link_ids = ["l0", "l1", "l2", "l3", "l4", "l5"]
        self.sensors = ["s1", "s2", "s3"]

    def test_fairness_aware_selects_one_link_per_sensor_for_full_coverage(self) -> None:
        hist = fairness_aware_greedy_generic(
            packet_matrix=self.packet_matrix,
            packet_sensor=self.packet_sensor,
            sensor_total=self.sensor_total,
            link_ids=self.link_ids,
            sensors_list=self.sensors,
            lam=1.0,
        )
        row_k3 = hist[hist["k_links"] == 3].iloc[0]
        self.assertGreaterEqual(row_k3["worst_sensor_coverage_pct"], 50.0)

    def test_coverage_only_greedy_can_leave_sensors_uncovered(self) -> None:
        hist = fairness_aware_greedy_generic(
            packet_matrix=self.packet_matrix,
            packet_sensor=self.packet_sensor,
            sensor_total=self.sensor_total,
            link_ids=self.link_ids,
            sensors_list=self.sensors,
            lam=0.0,
        )
        row_k1 = hist[hist["k_links"] == 1].iloc[0]
        self.assertEqual(row_k1["worst_sensor_coverage_pct"], 0.0)

    def test_multicover_stops_when_thresholds_met(self) -> None:
        hist = multicover_greedy(
            packet_matrix=self.packet_matrix,
            packet_sensor=self.packet_sensor,
            sensor_total=self.sensor_total,
            link_ids=self.link_ids,
            sensors_list=self.sensors,
            P_min_pct=50.0,
            S_min_pct=50.0,
        )
        self.assertLessEqual(len(hist), 3)
        last = hist.iloc[-1]
        self.assertGreaterEqual(last["packet_coverage_pct"], 50.0)
        self.assertGreaterEqual(last["worst_sensor_coverage_pct"], 50.0)

if __name__ == "__main__":
    unittest.main()
