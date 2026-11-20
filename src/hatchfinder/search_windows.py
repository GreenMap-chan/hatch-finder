import math
import random
from typing import Literal

import numpy as np
from PIL import Image

from .config import SearchMaskGenerationSettings


SampleKind = Literal["positive", "absent", "outside"]
Box = tuple[int, int, int, int]


def sample_kind(settings: SearchMaskGenerationSettings, rng: random.Random) -> SampleKind:
    if rng.random() >= settings.negative_fraction:
        return "positive"
    if rng.random() < settings.outside_window_share_of_negatives:
        return "outside"
    return "absent"


def _window_dimensions(image_size: tuple[int, int], settings: SearchMaskGenerationSettings):
    """Return the margins and inclusive range of legal widths for each feasible height."""
    width, height = image_size
    margin_x = math.ceil(width * settings.min_margin_fraction - 1e-9)
    margin_y = math.ceil(height * settings.min_margin_fraction - 1e-9)
    inner_w, inner_h = width - 2 * margin_x, height - 2 * margin_y
    full_area = width * height
    min_area = math.ceil(full_area * settings.area_fraction_min - 1e-9)
    max_area = math.floor(full_area * settings.area_fraction_max + 1e-9)
    heights = np.arange(1, inner_h + 1)
    low = np.maximum((min_area + heights - 1) // heights,
                     np.ceil(heights * settings.aspect_ratio_min - 1e-9))
    high = np.minimum(max_area // heights, np.floor(heights * settings.aspect_ratio_max + 1e-9))
    high = np.minimum(high, inner_w)
    low = np.maximum(low, 1)
    feasible = np.flatnonzero(low <= high)
    return margin_x, margin_y, heights[feasible], low[feasible].astype(int), high[feasible].astype(int)


def _fallback_sizes(kind: SampleKind, heights, low, high):
    # Any positive rectangle fits inside a legal maximal one. Any empty
    # rectangle contains a legal minimal one. Skip dimensions dominated by
    # previously tested rectangles so the fallback needs fewer image scans.
    if kind == "positive":
        widest = 0
        for w, h in zip(high[::-1], heights[::-1]):
            if w > widest:
                yield int(w), int(h)
                widest = w
    else:
        narrowest = math.inf
        for w, h in zip(low, heights):
            if w < narrowest:
                yield int(w), int(h)
                narrowest = w


def choose_window(
    target: Image.Image, kind: SampleKind, settings: SearchMaskGenerationSettings,
    rng: random.Random, quantiles: tuple[float, float], full_tile: bool, *, exhaustive: bool = False,
) -> Box | None:
    width, height = target.size
    if full_tile:
        if kind == "positive" and sum(target.histogram()[128:]) < settings.min_positive_pixels:
            return None
        return (0, 0, width, height)

    mx, my, heights, low, high = _window_dimensions(target.size, settings)
    if not heights.size:
        return None
    i = int(quantiles[0] * len(heights))
    h = int(heights[i])
    w = int(low[i]) + int(quantiles[1] * (int(high[i]) - int(low[i]) + 1))
    if kind == "absent":
        x = rng.randint(mx, width - mx - w)
        y = rng.randint(my, height - my - h)
        return (x, y, x + w, y + h)

    pixels = np.asarray(target) > 127
    inner = pixels[my:height-my, mx:width-mx]
    if kind == "positive" and np.count_nonzero(inner) < settings.min_positive_pixels:
        return None
    if kind == "outside" and (not pixels.any() or inner.all()):
        return None
    table = np.zeros((inner.shape[0] + 1, inner.shape[1] + 1), dtype=np.int32)
    sums = table[1:, 1:]
    np.cumsum(inner, axis=0, dtype=np.int32, out=sums)
    np.cumsum(sums, axis=1, dtype=np.int32, out=sums)
    sizes = _fallback_sizes(kind, heights, low, high) if exhaustive else [(w, h)]

    for w, h in sizes:
        counts = table[h:, w:] - table[:-h, w:] - table[h:, :-w] + table[:-h, :-w]
        matches = counts >= settings.min_positive_pixels if kind == "positive" else counts == 0
        # Choose uniformly without materializing a coordinate for every match.
        row_ends = np.cumsum(np.count_nonzero(matches, axis=1))
        if not row_ends[-1]:
            continue
        rank = rng.randrange(int(row_ends[-1]))
        row = int(row_ends.searchsorted(rank, side="right"))
        offset = rank - (int(row_ends[row-1]) if row else 0)
        x = mx + int(np.flatnonzero(matches[row])[offset])
        y = my + row
        return (x, y, x + w, y + h)
    return None
