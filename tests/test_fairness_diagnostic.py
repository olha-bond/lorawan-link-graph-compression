"""Tests for the fairness worst-tie diagnostic trace."""

import numpy as np

from lgc.analysis.fairness_diagnostic import fairness_worst_tie_diagnostic


def test_shared_worst_steps_have_zero_fairness_gain() -> None:
    diagnostic = fairness_worst_tie_diagnostic(lam=1.0)

    shared = diagnostic[diagnostic["step"].isin([1, 2])]
    assert (shared["n_sensors_at_current_worst"] > 1).all()
    assert np.allclose(shared["fairness_gain_pct"].to_numpy(), 0.0)


def test_unique_worst_step_activates_fairness_gain() -> None:
    diagnostic = fairness_worst_tie_diagnostic(lam=1.0)

    step3 = diagnostic[diagnostic["step"] == 3]
    selected = step3[step3["selected"]].iloc[0]

    assert (step3["n_sensors_at_current_worst"] == 1).all()
    assert selected["candidate_link"] == "s3→a"
    assert selected["fairness_gain_pct"] == 50.0
    assert selected["marginal_score"] > step3.loc[
        step3["candidate_link"] == "s1→b", "marginal_score"
    ].iloc[0]
