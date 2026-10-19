"""The risk band reads the interval's lower bound for "critical" (the safety
call) and the point estimate for "warning" (planning). No models needed."""

import pytest

from turbofan_rul.inference import RISK_THRESHOLDS, risk_level_for


@pytest.mark.parametrize("point, low, expected", [
    (60.0, 12.0, "critical"),   # healthy-looking estimate, but it could fail within 20 cycles
    (15.0, 22.0, "critical"),   # point estimate below 20 even if the bound isn't
    (40.0, 25.0, "warning"),    # best guess under 50, lower bound above 20
    (49.9, 20.0, "warning"),    # exactly at the critical threshold is not critical
    (80.0, 30.0, "healthy"),
    (50.0, 20.0, "healthy"),    # thresholds are strict "under"
])
def test_risk_level_for(point, low, expected):
    assert risk_level_for(point, low) == expected


def test_critical_never_less_cautious_than_point_rule():
    """Whatever the lower bound, the band is at least as severe as the old
    point-estimate-only rule."""
    order = {"healthy": 0, "warning": 1, "critical": 2}
    for point in range(0, 130, 5):
        old = ("critical" if point < RISK_THRESHOLDS["critical"]
               else "warning" if point < RISK_THRESHOLDS["warning"] else "healthy")
        for low in range(0, point + 1, 5):
            assert order[risk_level_for(float(point), float(low))] >= order[old]
