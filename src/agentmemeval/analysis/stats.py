"""Statistics used in reports (METHODOLOGY 1.7). Standard library only.

| situation                                  | method                               |
|--------------------------------------------|--------------------------------------|
| one proportion                             | Wilson score interval                |
| composite metrics, differences             | percentile bootstrap, by question    |
| two arms / before-after on the same items  | exact McNemar (discordant pairs)     |
| agreement                                  | Cohen's kappa + bootstrap interval   |

A slightly wrong statistic does not crash, it prints a normal-looking
interval, so every function here is pinned by hand-computed or textbook
values in tests/unit/test_stats.py.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import TypeVar

Z_95 = 1.959963984540054  # standard normal 0.975 quantile
DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 20251018
SMALL_SAMPLE = 30  # below this, results are qualitative only

T = TypeVar("T")


@dataclass(frozen=True)
class Interval:
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"interval low {self.low} > high {self.high}")


def wilson(successes: int, n: int, z: float = Z_95) -> Interval | None:
    """Wilson score interval for successes/n. None when n == 0 (nothing to estimate)."""
    if not 0 <= successes <= n:
        raise ValueError(f"need 0 <= successes <= n, got {successes}/{n}")
    if n == 0:
        return None
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    # Clamp floating-point noise at the boundaries (e.g. 0/n gives -1e-17).
    return Interval(max(0.0, center - half), min(1.0, center + half))


def _binom_cdf_half(k: int, n: int) -> float:
    """P(X <= k) for X ~ Binomial(n, 0.5), exact."""
    return sum(math.comb(n, i) for i in range(k + 1)) / 2**n


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c.

    b = pairs where only the first system is right, c = only the second. Pairs
    where both agree carry no information about the difference and are not inputs.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be >= 0")
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * _binom_cdf_half(min(b, c), n))


def min_discordant_for_significance(alpha: float = 0.05) -> int:
    """Smallest number of discordant pairs that can reach p < alpha at all (all one way)."""
    n = 1
    while mcnemar_exact(n, 0) >= alpha:
        n += 1
    return n


def bootstrap_interval(
    items: Sequence[T],
    statistic: Callable[[Sequence[T]], float | None],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> Interval | None:
    """Percentile bootstrap, resampling whole items (questions) with replacement.

    For paired comparisons pass items that hold both systems' results for one
    question, so the pairing survives resampling. Resamples where the statistic
    is undefined (None) are dropped; if more than 1% are, the interval is refused.
    """
    if n_resamples < 2000:
        raise ValueError("METHODOLOGY 1.7 requires at least 2000 resamples")
    if not items:
        return None
    rng = random.Random(seed)
    n = len(items)
    values = []
    undefined = 0
    for _ in range(n_resamples):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        v = statistic(sample)
        if v is None:
            undefined += 1
        else:
            values.append(v)
    if undefined > n_resamples // 100:
        return None
    values.sort()
    alpha = (1 - level) / 2
    return Interval(_quantile(values, alpha), _quantile(values, 1 - alpha))


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (numpy's default 'linear' method)."""
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float | None:
    """Cohen's kappa for two raters over the same items. None if chance agreement is 1."""
    if len(a) != len(b):
        raise ValueError("raters must label the same items")
    if not a:
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / (n * n)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def cohen_kappa_with_interval(
    a: Sequence[Hashable], b: Sequence[Hashable], **bootstrap_kwargs
) -> tuple[float | None, Interval | None]:
    pairs = list(zip(a, b, strict=True))
    kappa = cohen_kappa(a, b)

    def stat(sample: Sequence[tuple[Hashable, Hashable]]) -> float | None:
        return cohen_kappa([x for x, _ in sample], [y for _, y in sample])

    return kappa, bootstrap_interval(pairs, stat, **bootstrap_kwargs)


def paired_accuracy_difference(
    pairs: Sequence[tuple[bool, bool]], **bootstrap_kwargs
) -> tuple[float | None, Interval | None]:
    """Accuracy(second) - accuracy(first) on paired items, with a bootstrap interval."""

    def stat(sample: Sequence[tuple[bool, bool]]) -> float | None:
        return (sum(y for _, y in sample) - sum(x for x, _ in sample)) / len(sample)

    if not pairs:
        return None, None
    return stat(pairs), bootstrap_interval(pairs, stat, **bootstrap_kwargs)
