"""Statistics pinned to textbook / hand-computed values (METHODOLOGY 1.7, P1)."""

from __future__ import annotations

import math

import pytest

from agentmemeval.analysis.stats import (
    Interval,
    bootstrap_interval,
    cohen_kappa,
    cohen_kappa_with_interval,
    mcnemar_exact,
    min_discordant_for_significance,
    paired_accuracy_difference,
    wilson,
)

pytestmark = pytest.mark.unit


class TestWilson:
    @pytest.mark.parametrize(
        ("k", "n", "low", "high"),
        [
            # Newcombe (1998), Stat Med 17:857-872, Table II, method 3 (Wilson score).
            (81, 263, 0.2553, 0.3662),
            (15, 148, 0.0624, 0.1605),
            (0, 20, 0.0000, 0.1611),
            (1, 29, 0.0061, 0.1718),
            # The worked example in METHODOLOGY 1.7: 6/11 = 55% [28%, 79%].
            (6, 11, 0.2801, 0.7873),
        ],
    )
    def test_reference_values(self, k, n, low, high):
        i = wilson(k, n)
        assert (round(i.low, 4), round(i.high, 4)) == (low, high)

    def test_symmetry(self):
        a, b = wilson(3, 10), wilson(7, 10)
        assert a.low == pytest.approx(1 - b.high)
        assert a.high == pytest.approx(1 - b.low)

    def test_stays_inside_unit_interval_at_the_edges(self):
        assert wilson(0, 5).low == 0.0
        assert wilson(5, 5).high == 1.0

    def test_zero_denominator_has_no_interval(self):
        assert wilson(0, 0) is None

    @pytest.mark.parametrize(("k", "n"), [(-1, 5), (6, 5)])
    def test_rejects_impossible_counts(self, k, n):
        with pytest.raises(ValueError):
            wilson(k, n)


class TestMcNemar:
    @pytest.mark.parametrize("n", range(1, 11))
    def test_all_discordant_pairs_one_way_is_two_times_half_to_the_n(self, n):
        assert mcnemar_exact(n, 0) == pytest.approx(min(1.0, 2 * 0.5**n))
        assert mcnemar_exact(0, n) == mcnemar_exact(n, 0)

    def test_power_table_from_methodology(self):
        """n = 6 -> 0.031, n = 5 -> 0.063: fewer than 6 regressions can never reach 0.05."""
        assert mcnemar_exact(6, 0) == pytest.approx(0.03125)
        assert mcnemar_exact(5, 0) == pytest.approx(0.0625)
        assert min_discordant_for_significance(0.05) == 6

    def test_hand_computed_mixed_case(self):
        # 2 * P(X <= 3; n=14, p=0.5) = 2 * (1 + 14 + 91 + 364) / 2**14
        assert mcnemar_exact(3, 11) == pytest.approx(2 * 470 / 16384)

    def test_balanced_and_empty(self):
        assert mcnemar_exact(5, 5) == 1.0
        assert mcnemar_exact(0, 0) == 1.0

    def test_rejects_negative_counts(self):
        with pytest.raises(ValueError):
            mcnemar_exact(-1, 2)


def _raters(table):
    """[[yes/yes, yes/no], [no/yes, no/no]] -> two label lists."""
    (a, b), (c, d) = table
    r1 = ["Y"] * (a + b) + ["N"] * (c + d)
    r2 = ["Y"] * a + ["N"] * b + ["Y"] * c + ["N"] * d
    return r1, r2


class TestCohenKappa:
    @pytest.mark.parametrize(
        ("table", "kappa"),
        [
            # Wikipedia "Cohen's kappa": worked example and the two
            # "same agreement (60%), different kappa" tables.
            ([[20, 5], [10, 15]], 0.4),
            ([[45, 15], [25, 15]], 0.1304),
            ([[25, 35], [5, 35]], 0.2593),
        ],
    )
    def test_reference_values(self, table, kappa):
        assert round(cohen_kappa(*_raters(table)), 4) == kappa

    def test_perfect_and_chance_agreement(self):
        assert cohen_kappa(list("YNYN"), list("YNYN")) == 1.0
        assert cohen_kappa(list("YYNN"), list("YNYN")) == 0.0

    def test_undefined_when_both_raters_use_one_label(self):
        assert cohen_kappa(list("YYY"), list("YYY")) is None

    def test_works_for_more_than_two_labels(self):
        a = ["CORRECT", "WRONG", "UNSURE", "CORRECT"]
        b = ["CORRECT", "WRONG", "WRONG", "CORRECT"]
        # po = 3/4; pe = (2*2 + 1*2 + 1*0) / 16 = 6/16
        assert cohen_kappa(a, b) == pytest.approx((0.75 - 0.375) / (1 - 0.375))

    def test_rejects_unequal_lengths(self):
        with pytest.raises(ValueError):
            cohen_kappa(["Y"], ["Y", "N"])

    def test_interval_contains_the_estimate(self):
        r1, r2 = _raters([[20, 5], [10, 15]])
        kappa, interval = cohen_kappa_with_interval(r1, r2)
        assert interval.low < kappa < interval.high


class TestBootstrap:
    def test_is_reproducible_with_a_seed_and_changes_without(self):
        # Continuous values: with a few binary values, percentiles fall on the same
        # discrete grid for any seed and the "changes" half would be meaningless.
        data = [0.3, 1.7, 2.2, 0.9, 4.1, 3.3, 0.1, 2.8, 1.4, 3.9]

        def mean(s):
            return sum(s) / len(s)

        assert bootstrap_interval(data, mean, seed=1) == bootstrap_interval(data, mean, seed=1)
        assert bootstrap_interval(data, mean, seed=1) != bootstrap_interval(data, mean, seed=2)

    def test_constant_data_gives_a_zero_width_interval(self):
        assert bootstrap_interval([3.0] * 20, lambda s: sum(s) / len(s)) == Interval(3.0, 3.0)

    def test_mean_of_large_sample_matches_the_normal_approximation(self):
        data = [1] * 600 + [0] * 400  # p = 0.6, n = 1000
        i = bootstrap_interval(data, lambda s: sum(s) / len(s), n_resamples=2000)
        half = 1.96 * math.sqrt(0.6 * 0.4 / 1000)  # ~0.0304
        assert i.low == pytest.approx(0.6 - half, abs=0.006)
        assert i.high == pytest.approx(0.6 + half, abs=0.006)

    def test_refuses_fewer_than_2000_resamples(self):
        with pytest.raises(ValueError, match="2000"):
            bootstrap_interval([1, 2], lambda s: 0.0, n_resamples=500)

    def test_refuses_when_statistic_is_often_undefined(self):
        assert bootstrap_interval([1, 2, 3], lambda s: None) is None

    def test_empty_input(self):
        assert bootstrap_interval([], lambda s: 0.0) is None


class TestPairedDifference:
    def test_point_estimate_and_pairing(self):
        # (first correct, second correct) per question
        pairs = [(False, True)] * 6 + [(True, True)] * 4
        diff, interval = paired_accuracy_difference(pairs)
        assert diff == pytest.approx(0.6)
        assert interval.low > 0

    def test_identical_systems_have_zero_difference(self):
        pairs = [(True, True)] * 5 + [(False, False)] * 5
        diff, interval = paired_accuracy_difference(pairs)
        assert diff == 0
        assert interval == Interval(0.0, 0.0)

    def test_empty(self):
        assert paired_accuracy_difference([]) == (None, None)
