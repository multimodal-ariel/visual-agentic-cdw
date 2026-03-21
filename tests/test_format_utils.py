"""
Tests for processing/format_utils.py

Covers: save_organ_mask(), load_reference(), split_multilabel_nifti().
All tests use synthetic NIfTI data — no real images required.
"""

import os

import nibabel as nib
import numpy as np
import pytest

from processing.format_utils import (
    load_reference,
    save_organ_mask,
    split_multilabel_nifti,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SHAPE = (32, 32, 16)       # (X, Y, Z) in nibabel convention
AFFINE = np.eye(4)         # identity — 1mm isotropic, origin at (0,0,0)
ANISO_AFFINE = np.diag([1.5, 1.5, 3.0, 1.0])


@pytest.fixture
def reference_nifti(tmp_path):
    """Synthetic CT-like reference volume."""
    data = np.random.randint(-1000, 1000, SHAPE).astype(np.int16)
    img = nib.Nifti1Image(data, AFFINE)
    path = str(tmp_path / "image_nifti.nii.gz")
    nib.save(img, path)
    return path


@pytest.fixture
def seg_dir(tmp_path):
    d = str(tmp_path / "segmentations_test")
    os.makedirs(d)
    return d


# ---------------------------------------------------------------------------
# save_organ_mask
# ---------------------------------------------------------------------------

def test_save_creates_file(seg_dir):
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[10:20, 10:20, 5:10] = 1
    path = save_organ_mask(mask, "liver", seg_dir, AFFINE)
    assert os.path.isfile(path)
    assert os.path.basename(path) == "liver.nii.gz"


def test_save_output_is_binary(seg_dir):
    mask = np.ones(SHAPE, dtype=np.float32) * 0.9   # non-uint8 float input
    save_organ_mask(mask, "spleen", seg_dir, AFFINE)
    loaded = nib.load(os.path.join(seg_dir, "spleen.nii.gz"))
    vals = set(np.unique(loaded.get_fdata()))
    assert vals.issubset({0.0, 1.0})


def test_save_dtype_is_uint8(seg_dir):
    mask = np.zeros(SHAPE, dtype=np.uint8)
    save_organ_mask(mask, "pancreas", seg_dir, AFFINE)
    loaded = nib.load(os.path.join(seg_dir, "pancreas.nii.gz"))
    assert loaded.get_data_dtype() == np.uint8


def test_save_affine_preserved(seg_dir):
    save_organ_mask(np.zeros(SHAPE, dtype=np.uint8), "aorta", seg_dir, ANISO_AFFINE)
    loaded = nib.load(os.path.join(seg_dir, "aorta.nii.gz"))
    np.testing.assert_array_almost_equal(loaded.affine, ANISO_AFFINE)


def test_save_shape_preserved(seg_dir):
    mask = np.zeros(SHAPE, dtype=np.uint8)
    save_organ_mask(mask, "kidney_left", seg_dir, AFFINE)
    loaded = nib.load(os.path.join(seg_dir, "kidney_left.nii.gz"))
    assert loaded.shape == SHAPE


def test_save_creates_seg_dir_if_missing(tmp_path):
    new_dir = str(tmp_path / "deep" / "nested" / "seg")
    mask = np.zeros(SHAPE, dtype=np.uint8)
    save_organ_mask(mask, "gallbladder", new_dir, AFFINE)
    assert os.path.isfile(os.path.join(new_dir, "gallbladder.nii.gz"))


def test_save_returns_correct_path(seg_dir):
    mask = np.zeros(SHAPE, dtype=np.uint8)
    path = save_organ_mask(mask, "stomach", seg_dir, AFFINE)
    assert path == os.path.join(seg_dir, "stomach.nii.gz")


def test_save_values_round_trip(seg_dir):
    """Nonzero voxels in input should be 1 in output."""
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[5:10, 5:10, 2:6] = 1
    save_organ_mask(mask, "liver", seg_dir, AFFINE)
    loaded = nib.load(os.path.join(seg_dir, "liver.nii.gz")).get_fdata()
    np.testing.assert_array_equal(loaded[5:10, 5:10, 2:6], 1)
    np.testing.assert_array_equal(loaded[0:5, :, :], 0)


# ---------------------------------------------------------------------------
# load_reference
# ---------------------------------------------------------------------------

def test_load_reference_affine(reference_nifti):
    affine, header, shape = load_reference(reference_nifti)
    np.testing.assert_array_almost_equal(affine, AFFINE)


def test_load_reference_shape(reference_nifti):
    _, _, shape = load_reference(reference_nifti)
    assert shape[:3] == SHAPE


def test_load_reference_header_type(reference_nifti):
    _, header, _ = load_reference(reference_nifti)
    assert isinstance(header, nib.Nifti1Header)


def test_load_reference_missing_file():
    with pytest.raises(Exception):
        load_reference("/nonexistent/path/image.nii.gz")


# ---------------------------------------------------------------------------
# split_multilabel_nifti
# ---------------------------------------------------------------------------

@pytest.fixture
def multilabel_nifti(tmp_path):
    """
    Synthetic multilabel volume:
      label 1 = liver   (region A)
      label 2 = spleen  (region B)
      label 3 = absent  (all zeros)
    """
    data = np.zeros(SHAPE, dtype=np.int32)
    data[2:12, 2:12, 2:8] = 1   # liver
    data[14:24, 14:24, 2:8] = 2  # spleen
    # label 3 intentionally absent
    img = nib.Nifti1Image(data, AFFINE)
    path = str(tmp_path / "multilabel.nii.gz")
    nib.save(img, path)
    return path


LABEL_MAP = {1: "liver", 2: "spleen", 3: "kidney_left"}


def test_split_returns_present_organs(multilabel_nifti, seg_dir):
    saved = split_multilabel_nifti(multilabel_nifti, LABEL_MAP, seg_dir)
    assert set(saved) == {"liver", "spleen"}


def test_split_absent_label_not_saved(multilabel_nifti, seg_dir):
    split_multilabel_nifti(multilabel_nifti, LABEL_MAP, seg_dir)
    assert not os.path.isfile(os.path.join(seg_dir, "kidney_left.nii.gz"))


def test_split_files_created(multilabel_nifti, seg_dir):
    split_multilabel_nifti(multilabel_nifti, LABEL_MAP, seg_dir)
    assert os.path.isfile(os.path.join(seg_dir, "liver.nii.gz"))
    assert os.path.isfile(os.path.join(seg_dir, "spleen.nii.gz"))


def test_split_masks_are_binary(multilabel_nifti, seg_dir):
    split_multilabel_nifti(multilabel_nifti, LABEL_MAP, seg_dir)
    for organ in ("liver", "spleen"):
        data = nib.load(os.path.join(seg_dir, f"{organ}.nii.gz")).get_fdata()
        assert set(np.unique(data)).issubset({0.0, 1.0}), f"{organ} mask is not binary"


def test_split_masks_correct_regions(multilabel_nifti, seg_dir):
    """Verify the right voxels were extracted for each label."""
    split_multilabel_nifti(multilabel_nifti, LABEL_MAP, seg_dir)

    liver = nib.load(os.path.join(seg_dir, "liver.nii.gz")).get_fdata()
    assert liver[2:12, 2:12, 2:8].sum() == 10 * 10 * 6   # all ones in liver region
    assert liver[14:24, 14:24, 2:8].sum() == 0            # spleen region is zero

    spleen = nib.load(os.path.join(seg_dir, "spleen.nii.gz")).get_fdata()
    assert spleen[14:24, 14:24, 2:8].sum() == 10 * 10 * 6
    assert spleen[2:12, 2:12, 2:8].sum() == 0


def test_split_affine_preserved(multilabel_nifti, seg_dir):
    split_multilabel_nifti(multilabel_nifti, LABEL_MAP, seg_dir)
    loaded = nib.load(os.path.join(seg_dir, "liver.nii.gz"))
    np.testing.assert_array_almost_equal(loaded.affine, AFFINE)


def test_split_shape_preserved(multilabel_nifti, seg_dir):
    split_multilabel_nifti(multilabel_nifti, LABEL_MAP, seg_dir)
    loaded = nib.load(os.path.join(seg_dir, "liver.nii.gz"))
    assert loaded.shape == SHAPE


def test_split_empty_label_map(multilabel_nifti, seg_dir):
    """Empty label map → no files written, empty list returned."""
    saved = split_multilabel_nifti(multilabel_nifti, {}, seg_dir)
    assert saved == []


def test_split_all_absent_labels(multilabel_nifti, seg_dir):
    """Label map with indices that don't appear in the volume."""
    saved = split_multilabel_nifti(multilabel_nifti, {99: "brain", 100: "thyroid"}, seg_dir)
    assert saved == []
    assert not os.path.isfile(os.path.join(seg_dir, "brain.nii.gz"))


def test_split_creates_seg_dir_if_missing(multilabel_nifti, tmp_path):
    new_dir = str(tmp_path / "new" / "seg")
    split_multilabel_nifti(multilabel_nifti, {1: "liver"}, new_dir)
    assert os.path.isfile(os.path.join(new_dir, "liver.nii.gz"))


def test_split_anisotropic_affine(tmp_path, seg_dir):
    """Verify non-identity affine is preserved through split."""
    data = np.zeros(SHAPE, dtype=np.int32)
    data[5:15, 5:15, 3:9] = 1
    img = nib.Nifti1Image(data, ANISO_AFFINE)
    path = str(tmp_path / "aniso.nii.gz")
    nib.save(img, path)

    split_multilabel_nifti(path, {1: "pancreas"}, seg_dir)
    loaded = nib.load(os.path.join(seg_dir, "pancreas.nii.gz"))
    np.testing.assert_array_almost_equal(loaded.affine, ANISO_AFFINE)
