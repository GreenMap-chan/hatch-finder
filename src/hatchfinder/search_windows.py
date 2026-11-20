import math
import random
from contextlib import ExitStack
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


def _integral(pixels):
    table = np.zeros((pixels.shape[0] + 1, pixels.shape[1] + 1), dtype=np.int32)
    sums = table[1:, 1:]
    np.cumsum(pixels, axis=0, dtype=np.int32, out=sums)
    np.cumsum(sums, axis=1, dtype=np.int32, out=sums)
    return table


def _window_counts(table, w, h):
    return table[h:, w:] - table[:-h, w:] - table[h:, :-w] + table[:-h, :-w]


def _random_origin(matches, rng):
    row_ends = np.cumsum(np.count_nonzero(matches, axis=1))
    if not row_ends[-1]:
        return None
    rank = rng.randrange(int(row_ends[-1]))
    row = int(row_ends.searchsorted(rank, side="right"))
    offset = rank - (int(row_ends[row - 1]) if row else 0)
    return int(np.flatnonzero(matches[row])[offset]), row


def _positive_content_window(table, padding_table, heights, low, high, mx, my, minimum, rng):
    # After an affine transform the usable region can have sloping edges. For
    # each origin and height, try the widest legal rectangle up to the first
    # column containing padding. If it lacks matches, every narrower one does.
    columns = np.arange(table.shape[1] - 1, dtype=np.int32)
    for h, min_w, max_w in zip(heights, low, high):
        blocked = _window_counts(padding_table, 1, h) > 0
        stops = np.where(blocked, columns, len(columns))
        nearest_stop = np.minimum.accumulate(stops[:, ::-1], axis=1)[:, ::-1]
        right = np.minimum(nearest_stop, columns + int(max_w))
        strip = table[h:] - table[:-h]
        counts = np.take_along_axis(strip, right, axis=1) - strip[:, :-1]
        origin = _random_origin((right - columns >= min_w) & (counts >= minimum), rng)
        if origin is not None:
            x, y = origin
            return mx + x, my + y, mx + int(right[y, x]), my + y + int(h)
    return None


def choose_window(
    target: Image.Image, kind: SampleKind, settings: SearchMaskGenerationSettings,
    rng: random.Random, quantiles: tuple[float, float], full_tile: bool, *, exhaustive: bool = False,
    content: Image.Image | None = None,
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
    if kind == "absent" and content is None:
        x = rng.randint(mx, width - mx - w)
        y = rng.randint(my, height - my - h)
        return (x, y, x + w, y + h)

    pixels = np.asarray(target) > 127
    inner = pixels[my:height-my, mx:width-mx]
    if kind == "positive" and np.count_nonzero(inner) < settings.min_positive_pixels:
        return None
    if kind == "outside" and (not pixels.any() or inner.all()):
        return None
    table = _integral(inner)
    padding_table = None
    if content is not None:
        padding_table = _integral(np.asarray(content)[my:height-my, mx:width-mx] <= 127)
        if exhaustive and kind == "positive":
            return _positive_content_window(table, padding_table, heights, low, high, mx, my,
                                            settings.min_positive_pixels, rng)
    sizes = _fallback_sizes(kind, heights, low, high) if exhaustive else [(w, h)]

    for w, h in sizes:
        counts = _window_counts(table, w, h)
        matches = counts >= settings.min_positive_pixels if kind == "positive" else counts == 0
        if padding_table is not None:
            matches &= _window_counts(padding_table, w, h) == 0
        origin = _random_origin(matches, rng)
        if origin is None:
            continue
        x, y = mx + origin[0], my + origin[1]
        return (x, y, x + w, y + h)
    return None


def generate_window_mask(target, kind, settings, rng, quantiles, full_tile, *, exhaustive, content=None):
    # Fractions and margins refer to the content extent, including after augmentation.
    extent = content.getbbox() if content is not None else (0, 0, *target.size)
    if extent is None:
        return None
    with ExitStack() as images:
        region = target
        valid = content
        if extent != (0, 0, *target.size):
            region = images.enter_context(target.crop(extent))
            valid = images.enter_context(content.crop(extent))
        if valid is not None and valid.getextrema()[0] > 127:
            valid = None
        bounds = choose_window(region, kind, settings, rng, quantiles, full_tile,
                               exhaustive=exhaustive, content=valid)
    if bounds is None:
        return None
    if full_tile and content is not None:
        return content.copy()
    x, y = extent[:2]
    left, top, right, bottom = bounds
    mask = Image.new("L", target.size, 0)
    mask.paste(255, (x + left, y + top, x + right, y + bottom))
    return mask
