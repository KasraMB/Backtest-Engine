"""
Volume-at-price profile and Steidlmayer value area.

Uses volume (not TPO time) as the weight — modern futures-execution convention.
Bin count auto-selected via Freedman-Diaconis (ITFC patch from v9-final plan).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VolumeProfile:
    bin_edges: np.ndarray   # length B+1, price levels
    bin_centers: np.ndarray # length B
    bin_volumes: np.ndarray # length B, contracts at each price bin
    poc_idx: int            # index of point-of-control (max-volume bin)

    @property
    def poc(self) -> float:
        return float(self.bin_centers[self.poc_idx])


def _fd_bin_count(values: np.ndarray, tick_size: float) -> int:
    """Freedman-Diaconis bin count, floored at 1 tick width."""
    n = len(values)
    if n < 4:
        return max(1, int(np.ceil((values.max() - values.min()) / tick_size)))
    q75, q25 = np.percentile(values, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        return max(1, int(np.ceil((values.max() - values.min()) / tick_size)))
    h_fd = 2.0 * iqr / (n ** (1.0 / 3.0))
    h = max(h_fd, tick_size)  # never finer than one tick
    span = max(values.max() - values.min(), tick_size)
    return max(1, int(np.ceil(span / h)))


def volume_profile(
    prices: np.ndarray,
    volumes: np.ndarray,
    tick_size: float = 0.25,
    bins: Optional[int] = None,
) -> VolumeProfile:
    """Volume-at-price histogram.

    `prices` and `volumes` are per-bar arrays of equal length (typically close-price
    weighted by bar volume; for the ORB window pass close_1m and volume_1m sliced
    to the window).
    """
    if len(prices) != len(volumes):
        raise ValueError(f"prices ({len(prices)}) and volumes ({len(volumes)}) length mismatch")
    if len(prices) == 0:
        raise ValueError("empty inputs to volume_profile")
    p = np.asarray(prices, dtype=np.float64)
    v = np.asarray(volumes, dtype=np.float64)
    p_min, p_max = float(p.min()), float(p.max())
    if p_max == p_min:
        # Single-price window — one bin
        edges = np.array([p_min - tick_size / 2, p_min + tick_size / 2])
        centers = np.array([p_min])
        return VolumeProfile(edges, centers, np.array([v.sum()]), 0)
    n_bins = bins if bins is not None else _fd_bin_count(p, tick_size)
    edges = np.linspace(p_min, p_max, n_bins + 1)
    hist, _ = np.histogram(p, bins=edges, weights=v)
    centers = 0.5 * (edges[:-1] + edges[1:])
    poc_idx = int(np.argmax(hist))
    return VolumeProfile(edges, centers, hist, poc_idx)


def value_area(profile: VolumeProfile, va_pct: float = 0.70) -> tuple[float, float, float]:
    """Expand outward from POC adding the larger neighbor pair until ≥ va_pct of total.

    Returns (POC, VAH, VAL) as prices (bin centers at the inclusive edges).
    """
    vols = profile.bin_volumes
    total = vols.sum()
    if total <= 0:
        poc = profile.poc
        return poc, poc, poc
    target = total * va_pct
    poc = profile.poc_idx
    lo = hi = poc
    acc = vols[poc]
    n = len(vols)
    while acc < target and (lo > 0 or hi < n - 1):
        # Peek next pair (sum of next 1 above + next 1 below, then take whichever is bigger)
        up_vol  = vols[hi + 1] if hi < n - 1 else -1.0
        dn_vol  = vols[lo - 1] if lo > 0       else -1.0
        if up_vol < 0 and dn_vol < 0:
            break
        if up_vol >= dn_vol:
            hi += 1
            acc += up_vol
        else:
            lo -= 1
            acc += dn_vol
    return profile.poc, float(profile.bin_centers[hi]), float(profile.bin_centers[lo])


def orb_value_area(
    close_1m: np.ndarray,
    volume_1m: np.ndarray,
    bar_idx_start: int,
    bar_idx_end_exclusive: int,
    tick_size: float = 0.25,
    va_pct: float = 0.70,
    bins: Optional[int] = None,
) -> tuple[float, float, float]:
    """Compute (POC, VAH, VAL) over an ORB window slice of the 1m arrays."""
    sl = slice(bar_idx_start, bar_idx_end_exclusive)
    prof = volume_profile(close_1m[sl], volume_1m[sl], tick_size=tick_size, bins=bins)
    return value_area(prof, va_pct=va_pct)
