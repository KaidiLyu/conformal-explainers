import numpy as np
import pytest
import torch

from conformal_explainers.calibration import (
    MaskConfig,
    calibrate_threshold,
    conformity_score,
    make_mask,
    prepare_scores,
    wilson_interval,
)


def test_superpixel_fraction_and_count_have_distinct_meanings():
    a = np.array([[4.0, 0.0, 3.0], [0.0, 0.0, 2.0]])
    segments = np.array([[7, 7, 9], [7, 7, 9]])
    fraction = make_mask(a, 2, MaskConfig("superpixels", 0.5, "fraction"), segments)
    count = make_mask(a, 2, MaskConfig("superpixels", 1, "count"), segments)
    assert fraction.sum() == 2
    assert count.sum() == 6


def test_constant_scaled_scores_and_fallback_are_finite():
    cfg = MaskConfig("scaled", 100, "count")
    a = np.full((2, 2), -3.0)
    assert np.array_equal(prepare_scores(a, cfg), np.zeros((2, 2)))
    assert make_mask(a, -np.inf, cfg, np.zeros((2, 2))).all()
    assert not make_mask(a, 0, cfg, np.zeros((2, 2))).any()


def test_zero_negative_scores_are_valid():
    class Constant(torch.nn.Module):
        def forward(self, x):
            return torch.ones(len(x), 2)

    a = np.array([[-2.0, -1.0], [0.0, -3.0]])
    assert conformity_score(Constant(), torch.ones(3, 2, 2), a, MaskConfig()) == 0
    assert conformity_score(Constant(), torch.ones(3, 2, 2), a - 1, MaskConfig()) == -1


def test_nonmonotonic_predictions_scan_all_candidates():
    class Flips(torch.nn.Module):
        def forward(self, x):
            n = x[:, 0].sum((1, 2))
            # full=4 and singleton=1 agree; intermediate masks do not.
            score = (n - 2) * (n - 3)
            return torch.stack([torch.zeros_like(score), score - 0.1], dim=1)

    a = np.array([[0.0, 1.0], [2.0, 3.0]])
    assert (
        conformity_score(Flips(), torch.ones(3, 2, 2), a, MaskConfig(), steps=3) == 3.0
    )


@pytest.mark.parametrize("confidence", [0.85, 0.9, 0.95, 0.99])
def test_paper_quantile_matches_literal_equation_with_ties(confidence):
    s = np.array([-np.inf, -3.0, -1.0, 0.0, 0.0, 0.0, 2.0, 8.0, 9.0, 10.0])
    expected = max(t for t in s if ((s >= t).sum() + 1) / (len(s) + 1) >= confidence)
    assert calibrate_threshold(s, confidence) == expected


def test_small_sample_conservative_fallback_and_confidence_direction():
    assert calibrate_threshold([1, 2, 3], 0.99, "conservative") == -np.inf
    s = np.linspace(-5, 5, 100)
    thresholds = [calibrate_threshold(s, c) for c in [0.85, 0.9, 0.95, 0.99]]
    assert thresholds == sorted(thresholds, reverse=True)


@pytest.mark.parametrize(
    "scores,confidence", [([], 0.9), ([np.nan], 0.9), ([1], 1), ([np.inf], 0.9)]
)
def test_invalid_quantiles(scores, confidence):
    with pytest.raises(ValueError):
        calibrate_threshold(scores, confidence)


def test_wilson_interval_accounts_for_sample_size():
    lo, hi = wilson_interval(10, 10)
    assert 0.7 < lo < 0.8 and hi == pytest.approx(1.0)
