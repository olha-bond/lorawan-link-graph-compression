"""Unit tests for the sensor-separable exact DP."""

import unittest

from lgc.exact.dp import solve_exact_min_links_generic

class ExactDPSmall(unittest.TestCase):
    """A hand-solvable instance with 2 sensors and 4 gateway subsets each."""

    def setUp(self) -> None:
        self.pattern_lookup = {
            "s1": [
                (frozenset({"gA"}), 30),
                (frozenset({"gB"}), 10),
                (frozenset({"gA", "gB"}), 50),
                (frozenset(), 10),
            ],
            "s2": [
                (frozenset({"gA"}), 10),
                (frozenset({"gB"}), 0),
                (frozenset({"gA", "gB"}), 40),
                (frozenset(), 0),
            ],
        }
        self.sensor_total = {"s1": 100, "s2": 50}
        self.total_packets = 150
        self.sensor_gateway_lists = {"s1": ["gA", "gB"], "s2": ["gA", "gB"]}

    def solve(self, P_min: float, S_min: float) -> dict:
        return solve_exact_min_links_generic(
            P_min_pct=P_min,
            S_min_pct=S_min,
            sensors=["s1", "s2"],
            pattern_lookup_by_sensor=self.pattern_lookup,
            sensor_total=self.sensor_total,
            total_packets=self.total_packets,
            sensor_gateway_lists=self.sensor_gateway_lists,
            max_budget=4,
        )

    def test_relaxed_requirement_finds_two_link_optimum(self) -> None:
        res = self.solve(P_min=80.0, S_min=80.0)
        self.assertEqual(res["status"], "Optimal")
        self.assertEqual(res["n_links"], 2)

    def test_strict_worst_sensor_requires_more_links(self) -> None:
        res = self.solve(P_min=90.0, S_min=95.0)
        self.assertEqual(res["status"], "Infeasible")

    def test_selected_links_are_reconstructed(self) -> None:
        res = self.solve(P_min=70.0, S_min=70.0)
        self.assertEqual(res["status"], "Optimal")
        self.assertIsNotNone(res["selected_links"])
        for link in res["selected_links"]:
            self.assertTrue(link.startswith("s1→") or link.startswith("s2→"))

if __name__ == "__main__":
    unittest.main()
