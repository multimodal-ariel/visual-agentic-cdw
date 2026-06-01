import os

import nibabel as nib
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from processing.dicom_to_nifti_3d import (
    convert_dicom_dir_to_nifti,
    convert_dicom_to_nifti_3d,
    ensure_nifti_for_case,
    load_and_sort_dicom_slices,
)


def _write_slice(path, *, instance, z_position, value, series_uid):
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "CT"
    ds.Rows = 2
    ds.Columns = 3
    ds.InstanceNumber = instance
    ds.ImagePositionPatient = [0.0, 0.0, float(z_position)]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelSpacing = [0.5, 0.75]
    ds.SliceThickness = 2.0
    ds.SpacingBetweenSlices = 2.0
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope = 1
    ds.RescaleIntercept = 0
    ds.PixelData = np.full((2, 3), value, dtype=np.int16).tobytes()
    ds.save_as(path)


def test_convert_dicom_to_nifti_sorts_by_physical_position(tmp_path):
    series_uid = generate_uid()
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    _write_slice(dicom_dir / "slice_3.dcm", instance=1, z_position=4, value=30, series_uid=series_uid)
    _write_slice(dicom_dir / "slice_1.dcm", instance=3, z_position=0, value=10, series_uid=series_uid)
    _write_slice(dicom_dir / "slice_2.dcm", instance=2, z_position=2, value=20, series_uid=series_uid)

    slices = load_and_sort_dicom_slices(dicom_dir)
    assert [s.sort_position for s in slices] == [0.0, 2.0, 4.0]

    volume, affine, _ = convert_dicom_to_nifti_3d(dicom_dir)

    assert volume.shape == (3, 2, 3)
    assert np.all(volume[:, :, 0] == 10)
    assert np.all(volume[:, :, 1] == 20)
    assert np.all(volume[:, :, 2] == 30)
    assert np.allclose(np.diag(affine)[:3], [-0.75, -0.5, 2.0])


def test_ensure_nifti_for_case_is_idempotent(tmp_path):
    series_uid = generate_uid()
    dicom_dir = tmp_path / "dicom"
    case_dir = tmp_path / "case"
    dicom_dir.mkdir()
    case_dir.mkdir()
    _write_slice(dicom_dir / "slice_1.dcm", instance=1, z_position=0, value=1, series_uid=series_uid)
    _write_slice(dicom_dir / "slice_2.dcm", instance=2, z_position=2, value=2, series_uid=series_uid)

    first = ensure_nifti_for_case(case_dir, dicom_dir)
    second = ensure_nifti_for_case(case_dir, dicom_dir)

    assert first["status"] == "converted"
    assert second["status"] == "exists"
    assert os.path.isfile(case_dir / "image_nifti.nii.gz")

    img = nib.load(case_dir / "image_nifti.nii.gz")
    assert img.shape == (3, 2, 2)


def test_convert_dicom_dir_to_nifti_selects_largest_valid_series(tmp_path):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    large_uid = generate_uid()
    small_uid = generate_uid()
    _write_slice(dicom_dir / "large_1.dcm", instance=1, z_position=0, value=10, series_uid=large_uid)
    _write_slice(dicom_dir / "large_2.dcm", instance=2, z_position=2, value=20, series_uid=large_uid)
    _write_slice(dicom_dir / "large_3.dcm", instance=3, z_position=4, value=30, series_uid=large_uid)
    _write_slice(dicom_dir / "small_1.dcm", instance=1, z_position=0, value=99, series_uid=small_uid)

    output = tmp_path / "out.nii.gz"
    convert_dicom_dir_to_nifti(dicom_dir, output)

    img = nib.load(output)
    data = np.asanyarray(img.dataobj)
    assert img.shape == (3, 2, 3)
    assert np.all(data[:, :, 0] == 10)
    assert np.all(data[:, :, 1] == 20)
    assert np.all(data[:, :, 2] == 30)


def test_convert_dicom_dir_to_nifti_resolves_same_geometry_series_tie(tmp_path):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    uid_1 = generate_uid()
    uid_2 = generate_uid()
    _write_slice(dicom_dir / "a_1.dcm", instance=1, z_position=0, value=1, series_uid=uid_1)
    _write_slice(dicom_dir / "a_2.dcm", instance=2, z_position=2, value=2, series_uid=uid_1)
    _write_slice(dicom_dir / "b_1.dcm", instance=1, z_position=0, value=3, series_uid=uid_2)
    _write_slice(dicom_dir / "b_2.dcm", instance=2, z_position=2, value=4, series_uid=uid_2)

    output = tmp_path / "out.nii.gz"
    convert_dicom_dir_to_nifti(dicom_dir, output)

    img = nib.load(output)
    assert img.shape == (3, 2, 2)


def test_convert_dicom_dir_to_nifti_rejects_ambiguous_series_tie(tmp_path):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    uid_1 = generate_uid()
    uid_2 = generate_uid()
    _write_slice(dicom_dir / "a_1.dcm", instance=1, z_position=0, value=1, series_uid=uid_1)
    _write_slice(dicom_dir / "a_2.dcm", instance=2, z_position=2, value=2, series_uid=uid_1)
    _write_slice(dicom_dir / "b_1.dcm", instance=1, z_position=10, value=3, series_uid=uid_2)
    _write_slice(dicom_dir / "b_2.dcm", instance=2, z_position=12, value=4, series_uid=uid_2)

    try:
        convert_dicom_dir_to_nifti(dicom_dir, tmp_path / "out.nii.gz")
    except ValueError as exc:
        assert "Ambiguous DICOM directory" in str(exc)
    else:
        raise AssertionError("ambiguous mixed DICOM series should be rejected")
