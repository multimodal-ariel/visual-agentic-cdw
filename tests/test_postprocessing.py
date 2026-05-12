"""Byte-identity regression tests for the bbox-cropped postprocessing.

The optimized ``keep_largest_component`` and ``fill_holes`` in
``processing/postprocessing.py`` operate on the tight nonzero bounding box of
the input mask instead of the full volume. These tests pin their output to
the unoptimized scipy reference so future refactors can't silently change
mask semantics.
"""

import numpy as np
import pytest
from scipy.ndimage import binary_fill_holes as _ref_bfh
from scipy.ndimage import label as _ref_label

from processing.postprocessing import (
    _nonzero_bbox,
    fill_holes,
    keep_largest_component,
)


def _ref_keep_largest(mask: np.ndarray) -> np.ndarray:
    """Full-volume reference implementation (pre-optimization)."""
    labeled, num_features = _ref_label(mask)
    if num_features <= 1:
        return mask
    sizes = np.bincount(labeled.ravel())[1:]
    largest = sizes.argmax() + 1
    return (labeled == largest).astype(mask.dtype)


def _ref_fill_holes(mask: np.ndarray) -> np.ndarray:
    """Full-volume reference implementation (pre-optimization)."""
    return _ref_bfh(mask).astype(mask.dtype)


# ──────────────────────────────────────────────────────────────────────────
# Mask fixtures covering the algorithmically interesting shapes
# ──────────────────────────────────────────────────────────────────────────

def _solid_blob() -> np.ndarray:
    m = np.zeros((64, 64, 64), np.uint8)
    m[20:30, 25:35, 30:40] = 1
    return m


def _blob_with_interior_hole() -> np.ndarray:
    m = np.zeros((64, 64, 64), np.uint8)
    m[10:50, 10:50, 10:50] = 1
    m[25:30, 25:30, 25:30] = 0
    return m


def _two_disconnected_blobs() -> np.ndarray:
    m = np.zeros((64, 64, 64), np.uint8)
    m[5:15, 5:15, 5:15] = 1
    m[30:50, 30:50, 30:50] = 1
    return m


def _blob_at_volume_boundary() -> np.ndarray:
    """Mask touching face z=0 with an interior hole — verifies that the
    1-voxel zero pad in fill_holes correctly substitutes for the outer
    volume zeros."""
    m = np.zeros((64, 64, 64), np.uint8)
    m[0:32, 0:32, 0:32] = 1
    m[5:10, 5:10, 5:10] = 0
    return m


def _u_shape_opening_to_face() -> np.ndarray:
    """U-shaped mask whose opening reaches the z=0 face — its 'hole' is
    actually connected to outside and must NOT be filled."""
    m = np.zeros((64, 64, 64), np.uint8)
    m[20:40, 20:40, 0:30] = 1
    m[25:35, 25:35, 0:25] = 0
    return m


def _empty() -> np.ndarray:
    return np.zeros((64, 64, 64), np.uint8)


def _single_voxel() -> np.ndarray:
    m = np.zeros((64, 64, 64), np.uint8)
    m[32, 32, 32] = 1
    return m


def _random_noise() -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.random((128, 128, 128)) > 0.7).astype(np.uint8)


def _big_512_organ() -> np.ndarray:
    """Realistic 512³ volume with a liver-sized organ + interior hole."""
    m = np.zeros((512, 512, 237), np.uint8)
    m[100:300, 150:380, 50:200] = 1
    m[200:220, 250:280, 100:120] = 0
    return m


def _dense_all_ones() -> np.ndarray:
    return np.ones((96, 96, 96), np.uint8)


def _fragmented_many_components() -> np.ndarray:
    m = np.zeros((64, 64, 64), np.uint8)
    for i, c in enumerate([(10, 10, 10), (40, 40, 40), (10, 40, 10), (40, 10, 40)]):
        # Larger blob first so we can verify largest-component selection.
        size = 6 + 2 * i
        m[c[0]:c[0] + size, c[1]:c[1] + size, c[2]:c[2] + size] = 1
    return m


_MASKS = {
    "solid_blob": _solid_blob,
    "blob_with_interior_hole": _blob_with_interior_hole,
    "two_disconnected_blobs": _two_disconnected_blobs,
    "blob_at_volume_boundary": _blob_at_volume_boundary,
    "u_shape_opening_to_face": _u_shape_opening_to_face,
    "empty": _empty,
    "single_voxel": _single_voxel,
    "random_noise": _random_noise,
    "big_512_organ": _big_512_organ,
    "dense_all_ones": _dense_all_ones,
    "fragmented_many_components": _fragmented_many_components,
}


@pytest.mark.parametrize("name", list(_MASKS.keys()))
def test_keep_largest_component_matches_reference(name):
    mask = _MASKS[name]()
    ref = _ref_keep_largest(mask)
    got = keep_largest_component(mask)
    assert np.array_equal(ref, got), f"keep_largest_component diverged on '{name}'"


@pytest.mark.parametrize("name", list(_MASKS.keys()))
def test_fill_holes_matches_reference(name):
    mask = _MASKS[name]()
    ref = _ref_fill_holes(mask)
    got = fill_holes(mask)
    assert np.array_equal(ref, got), f"fill_holes diverged on '{name}'"


# ──────────────────────────────────────────────────────────────────────────
# Bounding-box helper invariants
# ──────────────────────────────────────────────────────────────────────────

def test_nonzero_bbox_empty_returns_none():
    assert _nonzero_bbox(np.zeros((10, 10, 10), np.uint8)) is None


def test_nonzero_bbox_is_tight():
    m = np.zeros((20, 20, 20), np.uint8)
    m[3:7, 5:9, 11:15] = 1
    slc = _nonzero_bbox(m)
    assert slc == (slice(3, 7), slice(5, 9), slice(11, 15))


def test_nonzero_bbox_covers_full_volume_for_dense_mask():
    m = np.ones((8, 8, 8), np.uint8)
    slc = _nonzero_bbox(m)
    assert slc == (slice(0, 8), slice(0, 8), slice(0, 8))


def test_nonzero_bbox_handles_single_voxel():
    m = np.zeros((10, 10, 10), np.uint8)
    m[4, 5, 6] = 1
    slc = _nonzero_bbox(m)
    assert slc == (slice(4, 5), slice(5, 6), slice(6, 7))
