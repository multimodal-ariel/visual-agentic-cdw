"""Reusable 3D DICOM-to-NIfTI conversion utilities.

This module is import-safe: batch entry points call these functions directly
instead of relying on a hardcoded ``if __name__ == "__main__"`` script block.
The converter preserves DICOM geometry by sorting slices by their physical
position along the acquisition normal, then writing a RAS-oriented NIfTI.
"""

from __future__ import annotations

import json
import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from nibabel.nifti1 import Nifti1Extension, Nifti1Image
from pydicom import dcmread


LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])
DEFAULT_IMAGE_FILENAME = "image_nifti.nii.gz"


@dataclass(frozen=True)
class DicomSlice:
    """Header metadata needed to validate and sort one DICOM slice."""

    path: str
    position: np.ndarray
    row_cosines: np.ndarray
    col_cosines: np.ndarray
    normal: np.ndarray
    sort_position: float
    instance_number: int | None
    series_uid: str
    study_uid: str
    rows: int
    columns: int
    pixel_spacing: tuple[float, float]


def extract_selected_dicom_metadata(dicom) -> dict:
    """Extract compact provenance fields for the NIfTI header extension."""

    selected_keys = [
        "AccessionNumber",
        "Modality",
        "BodyPartExamined",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SeriesDescription",
        "ProtocolName",
    ]
    metadata = {}
    for key in selected_keys:
        if hasattr(dicom, key):
            metadata[key] = str(getattr(dicom, key))
    return metadata


def find_dicom_files(dicom_dir: str | os.PathLike) -> list[str]:
    """Return candidate DICOM files below ``dicom_dir``.

    Some exports omit conventional extensions or use vendor-specific ones.
    Header parsing later filters out non-DICOM files.
    """

    root = Path(dicom_dir)
    if not root.is_dir():
        raise ValueError(f"Path is not a directory: {dicom_dir}")

    paths: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.startswith("."):
                continue
            path = Path(dirpath) / name
            paths.append(str(path))
    if not paths:
        raise ValueError(f"No DICOM candidate files found in {dicom_dir}")
    return sorted(paths)


def _as_float_array(values: Iterable, field: str, path: str) -> np.ndarray:
    try:
        return np.asarray([float(v) for v in values], dtype=float)
    except Exception as exc:
        raise ValueError(f"Invalid {field} in {path}: {exc}") from exc


def _read_slice_header(path: str) -> DicomSlice | None:
    try:
        ds = dcmread(path, stop_before_pixels=True)
    except Exception:
        return None

    required = (
        "ImagePositionPatient",
        "ImageOrientationPatient",
        "PixelSpacing",
        "Rows",
        "Columns",
        "SeriesInstanceUID",
    )
    if any(not hasattr(ds, key) for key in required):
        return None

    position = _as_float_array(ds.ImagePositionPatient, "ImagePositionPatient", path)
    orientation = _as_float_array(ds.ImageOrientationPatient, "ImageOrientationPatient", path)
    if position.shape != (3,) or orientation.shape != (6,):
        raise ValueError(f"Invalid DICOM geometry vector lengths in {path}")

    row_cosines = orientation[:3]
    col_cosines = orientation[3:]
    normal = np.cross(row_cosines, col_cosines)
    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0:
        raise ValueError(f"Invalid ImageOrientationPatient with zero normal in {path}")
    normal = normal / normal_norm

    pixel_spacing = tuple(float(v) for v in ds.PixelSpacing)
    if len(pixel_spacing) != 2:
        raise ValueError(f"Invalid PixelSpacing in {path}: {ds.PixelSpacing}")

    instance_number = None
    if hasattr(ds, "InstanceNumber"):
        try:
            instance_number = int(ds.InstanceNumber)
        except Exception:
            instance_number = None

    return DicomSlice(
        path=path,
        position=position,
        row_cosines=row_cosines,
        col_cosines=col_cosines,
        normal=normal,
        sort_position=float(np.dot(position, normal)),
        instance_number=instance_number,
        series_uid=str(ds.SeriesInstanceUID),
        study_uid=str(getattr(ds, "StudyInstanceUID", "")),
        rows=int(ds.Rows),
        columns=int(ds.Columns),
        pixel_spacing=(float(pixel_spacing[0]), float(pixel_spacing[1])),
    )


def load_and_sort_dicom_slices(dicom_dir: str | os.PathLike) -> list[DicomSlice]:
    """Load headers, validate one 3D series, and sort by physical slice position."""

    slices = []
    for path in find_dicom_files(dicom_dir):
        header = _read_slice_header(path)
        if header is not None:
            slices.append(header)

    if not slices:
        raise ValueError(f"No valid 3D DICOM slices found in {dicom_dir}")

    series_uids = {s.series_uid for s in slices}
    if len(series_uids) != 1:
        raise ValueError(
            f"Expected one SeriesInstanceUID in {dicom_dir}, found {len(series_uids)}"
        )

    ref = slices[0]
    for s in slices[1:]:
        if (s.rows, s.columns) != (ref.rows, ref.columns):
            raise ValueError(f"Mixed slice matrix sizes in {dicom_dir}")
        if not np.allclose(s.row_cosines, ref.row_cosines, atol=1e-4) or not np.allclose(
            s.col_cosines, ref.col_cosines, atol=1e-4
        ):
            raise ValueError(f"Mixed slice orientations in {dicom_dir}")
        if not np.allclose(s.pixel_spacing, ref.pixel_spacing, atol=1e-4):
            raise ValueError(f"Mixed pixel spacing in {dicom_dir}")

    sorted_slices = sorted(
        slices,
        key=lambda s: (
            round(s.sort_position, 5),
            s.instance_number if s.instance_number is not None else 0,
            s.path,
        ),
    )
    positions = np.asarray([s.sort_position for s in sorted_slices], dtype=float)
    if len(positions) > 1:
        deltas = np.diff(positions)
        if np.any(np.isclose(deltas, 0.0, atol=1e-5)):
            raise ValueError(f"Duplicate slice positions in {dicom_dir}")
        if np.any(deltas < 0):
            raise ValueError(f"Slice sorting failed for {dicom_dir}")

    return sorted_slices


def _slice_spacing(sorted_slices: list[DicomSlice], ref_dicom) -> float:
    if len(sorted_slices) > 1:
        positions = np.asarray([s.sort_position for s in sorted_slices], dtype=float)
        spacings = np.abs(np.diff(positions))
        median_spacing = float(np.median(spacings))
        if median_spacing > 0:
            if np.max(np.abs(spacings - median_spacing)) > max(0.1, 0.05 * median_spacing):
                raise ValueError(f"Inconsistent slice spacing: {spacings.tolist()}")
            return median_spacing
    if hasattr(ref_dicom, "SpacingBetweenSlices"):
        return float(ref_dicom.SpacingBetweenSlices)
    return float(getattr(ref_dicom, "SliceThickness", 1.0))


def convert_dicom_to_nifti_3d(dicom_dir: str | os.PathLike):
    """Convert one validated DICOM series directory to ``(volume, affine, ext)``.

    The output volume is shaped ``(X, Y, Z)`` and the affine is in NIfTI RAS
    coordinates. CT rescale slope/intercept are applied when present.
    """

    sorted_slices = load_and_sort_dicom_slices(dicom_dir)
    sorted_paths = [s.path for s in sorted_slices]
    ref_slice = sorted_slices[0]
    ref_dicom = dcmread(sorted_paths[0])

    volume_zyx = np.zeros(
        (len(sorted_paths), ref_slice.rows, ref_slice.columns),
        dtype=np.float32,
    )
    for idx, fpath in enumerate(sorted_paths):
        ds = dcmread(fpath)
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        volume_zyx[idx, :, :] = ds.pixel_array.astype(np.float32) * slope + intercept

    volume = volume_zyx.transpose(2, 1, 0)

    row_spacing, col_spacing = ref_slice.pixel_spacing
    z_spacing = _slice_spacing(sorted_slices, ref_dicom)
    row_ras = LPS_TO_RAS @ ref_slice.row_cosines
    col_ras = LPS_TO_RAS @ ref_slice.col_cosines
    z_ras = LPS_TO_RAS @ ref_slice.normal
    origin_ras = LPS_TO_RAS @ ref_slice.position

    affine = np.eye(4)
    affine[:3, 0] = row_ras * col_spacing
    affine[:3, 1] = col_ras * row_spacing
    affine[:3, 2] = z_ras * z_spacing
    affine[:3, 3] = origin_ras

    metadata = extract_selected_dicom_metadata(ref_dicom)
    metadata.update(
        {
            "num_slices": len(sorted_slices),
            "slice_spacing": z_spacing,
            "sort_method": "ImagePositionPatient_projected_on_slice_normal",
        }
    )
    ext = Nifti1Extension(6, json.dumps(metadata, indent=2).encode("utf-8"))
    return volume, affine, ext


def save_nifti_with_metadata(volume: np.ndarray, affine: np.ndarray, ext, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nifti_img = Nifti1Image(volume, affine)
    nifti_img.header.extensions.append(ext)
    nib.save(nifti_img, output_path)


def convert_dicom_dir_to_nifti(
    dicom_dir: str | os.PathLike,
    output_path: str | os.PathLike,
    *,
    overwrite: bool = False,
) -> str:
    """Convert ``dicom_dir`` into ``output_path`` and return the output path."""

    output_path = str(output_path)
    if os.path.isfile(output_path) and not overwrite:
        return output_path
    volume, affine, ext = convert_dicom_to_nifti_3d(dicom_dir)
    save_nifti_with_metadata(volume, affine, ext, output_path)
    return output_path


def ensure_nifti_for_case(
    case_path: str | os.PathLike,
    dicom_path: str | os.PathLike,
    *,
    image_filename: str = DEFAULT_IMAGE_FILENAME,
    overwrite: bool = False,
) -> dict:
    """Ensure ``case_path/image_filename`` exists, converting DICOM if needed.

    Returns a small structured status dict for batch logs and tests.
    """

    case_path = str(case_path)
    dicom_path = str(dicom_path)
    output_path = os.path.join(case_path, image_filename)

    if os.path.isfile(output_path) and not overwrite:
        return {
            "status": "exists",
            "image_path": output_path,
            "dicom_path": dicom_path,
        }

    if not os.path.isdir(dicom_path):
        return {
            "status": "missing_dicom",
            "image_path": output_path,
            "dicom_path": dicom_path,
            "error": f"DICOM directory not found: {dicom_path}",
        }

    try:
        convert_dicom_dir_to_nifti(dicom_path, output_path, overwrite=overwrite)
    except Exception as exc:
        return {
            "status": "conversion_failed",
            "image_path": output_path,
            "dicom_path": dicom_path,
            "error": str(exc),
        }

    return {
        "status": "converted",
        "image_path": output_path,
        "dicom_path": dicom_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one 3D DICOM series to NIfTI.")
    parser.add_argument("dicom_dir", help="Directory containing one DICOM series")
    parser.add_argument("output_path", help="Output .nii or .nii.gz path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output")
    args = parser.parse_args()

    out = convert_dicom_dir_to_nifti(
        args.dicom_dir,
        args.output_path,
        overwrite=args.overwrite,
    )
    print(out)


if __name__ == "__main__":
    main()