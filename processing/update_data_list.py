#!/usr/bin/env python3
"""
Fast DICOM directory scanner.

Key optimizations over the original script:
  - discovers DICOM-containing directories via *.dcm files, not leaf-dir walks
  - skips previously processed patients before expensive DICOM reads
  - reads DICOM headers only; never loads pixel arrays
  - classifies dimensionality after removing singleton axes wherever they occur
"""

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


DEFAULT_ROOT_DIR = "/data/RAD/"
DEFAULT_OUTPUT_DIR = "/data/soumitri/new_data_paths"
DEFAULT_MAX_WORKERS = max(1, multiprocessing.cpu_count() - 2)
RAD_COHORT_INDEX = 0
RAD_PATIENT_ID_INDEX = 1
RAD_COHORTS = {"CONTR", "RHEUM", "LUPUS"}

OUTPUT_FILES = {
    "3d": "3d_scans_list.json",
    "2d": "2d_scans_list.json",
    "4d_5d": "4d_or_5d_scans_list.json",
    "processed": "dicom_dir_list.json",
    "errors": "dicom_error_list.json",
    "patients": "processed_patient_ids.json",
}


def dump_to_json(data: Any, filename: str, output_dir: str) -> None:
    filepath = filename if os.path.isabs(filename) else os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def extract_record_path(record: Any) -> str | None:
    if isinstance(record, str):
        return record
    if isinstance(record, (list, tuple)) and record:
        return str(record[0])
    return None


def patient_id_from_rad_path(path: str, root_dir: str) -> str:
    """Extract patient ID from RAD paths.

    Supported layouts:
      - root_dir=/data/RAD, path=/data/RAD/<cohort>/<patient-id>/...
      - root_dir=/data/RAD/<cohort>, path=/data/RAD/<cohort>/<patient-id>/...
    """
    try:
        root_path = Path(root_dir).resolve()
        parts = Path(path).resolve().relative_to(root_path).parts
    except ValueError as exc:
        raise ValueError(f"path is not under root_dir: {path}") from exc

    root_cohort = root_path.name
    if root_cohort in RAD_COHORTS:
        if not parts:
            raise ValueError(
                "expected path layout /data/RAD/<CONTR|RHEUM|LUPUS>/<patient-id>/..., "
                f"got: {path}"
            )
        return parts[0]

    if len(parts) <= RAD_PATIENT_ID_INDEX:
        raise ValueError(
            "expected path layout /data/RAD/<CONTR|RHEUM|LUPUS>/<patient-id>/..., "
            f"got: {path}"
        )

    cohort = parts[RAD_COHORT_INDEX]
    if cohort not in RAD_COHORTS:
        raise ValueError(
            f"expected cohort to be one of {sorted(RAD_COHORTS)}, got {cohort!r}: {path}"
        )

    return parts[RAD_PATIENT_ID_INDEX]


def cohort_from_rad_path(path: str, root_dir: str) -> str:
    """Extract cohort from RAD paths."""
    try:
        root_path = Path(root_dir).resolve()
        parts = Path(path).resolve().relative_to(root_path).parts
    except ValueError as exc:
        raise ValueError(f"path is not under root_dir: {path}") from exc

    root_cohort = root_path.name
    if root_cohort in RAD_COHORTS:
        return root_cohort

    if not parts:
        raise ValueError(
            "expected path layout /data/RAD/<CONTR|RHEUM|LUPUS>/<patient-id>/..., "
            f"got: {path}"
        )

    cohort = parts[RAD_COHORT_INDEX]
    if cohort not in RAD_COHORTS:
        raise ValueError(
            f"expected cohort to be one of {sorted(RAD_COHORTS)}, got {cohort!r}: {path}"
        )

    return cohort


def load_json_list(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def load_existing(output_dir: str, root_dir: str) -> tuple[dict, set[str], dict[str, set[str]]]:
    data_stores = {
        key: load_json_list(os.path.join(output_dir, filename))
        for key, filename in OUTPUT_FILES.items()
        if key != "patients"
    }

    patient_file = os.path.join(output_dir, OUTPUT_FILES["patients"])
    processed_patient_ids = set(load_json_list(patient_file))
    processed_by_cohort = {cohort: set() for cohort in RAD_COHORTS}

    # Backfill patient IDs from older JSONs that only stored paths.
    for records in data_stores.values():
        for record in records:
            path = extract_record_path(record)
            if path:
                try:
                    patient_id = patient_id_from_rad_path(path, root_dir)
                    cohort = cohort_from_rad_path(path, root_dir)
                except ValueError:
                    continue
                processed_patient_ids.add(patient_id)
                processed_by_cohort[cohort].add(patient_id)

    return data_stores, processed_patient_ids, processed_by_cohort


def scan_roots(root_dir: str, cohorts: list[str] | None = None) -> list[Path]:
    root = Path(root_dir)
    root_cohort = root.name

    if root_cohort in RAD_COHORTS:
        if cohorts and any(cohort != root_cohort for cohort in cohorts):
            raise ValueError(
                f"--root-dir is already cohort {root_cohort!r}; "
                f"cannot also scan cohorts {cohorts!r}"
            )
        return [root]

    if cohorts:
        return [root / cohort for cohort in cohorts]

    return [root]


def iter_dicom_dirs(root_dir: str, cohorts: list[str] | None = None) -> list[Path]:
    """Return unique directories containing at least one .dcm file."""
    dirs: set[Path] = set()

    # This still traverses the tree, but only materializes directories that
    # actually contain DICOM-looking files.
    for scan_root in scan_roots(root_dir, cohorts):
        for pattern in ("*.dcm", "*.DCM"):
            matches = scan_root.rglob(pattern)
            for dcm in tqdm(
                matches,
                desc=f"Finding {pattern} under {scan_root}",
                unit="file",
                dynamic_ncols=True,
            ):
                dirs.add(dcm.parent)

    return sorted(dirs)


def first_dicom_file(series_dir: Path) -> Path | None:
    for child in series_dir.iterdir():
        if child.is_file() and child.suffix.lower() == ".dcm":
            return child
    return None


def read_header(path: Path):
    if pydicom is not None:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)

    if sitk is None:
        raise RuntimeError("Install either pydicom or SimpleITK to read DICOM headers.")

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return reader


def header_value(header, keyword: str, sitk_tag: str, default=None):
    if pydicom is not None and header.__class__.__module__.startswith("pydicom"):
        return getattr(header, keyword, default)

    if hasattr(header, "HasMetaDataKey") and header.HasMetaDataKey(sitk_tag):
        return header.GetMetaData(sitk_tag).strip()
    return default


def int_header_value(header, keyword: str, sitk_tag: str, default: int) -> int:
    value = header_value(header, keyword, sitk_tag, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def same_series_file_count(series_dir: Path, series_uid: str | None) -> int:
    count = 0
    for dcm in series_dir.iterdir():
        if not dcm.is_file() or dcm.suffix.lower() != ".dcm":
            continue

        if series_uid is None:
            count += 1
            continue

        try:
            header = read_header(dcm)
            if header_value(header, "SeriesInstanceUID", "0020|000e") == series_uid:
                count += 1
        except Exception:
            continue

    return max(1, count)


def infer_shape_from_headers(series_dir: Path, first_header) -> tuple[int, ...]:
    rows = int_header_value(first_header, "Rows", "0028|0010", 0)
    cols = int_header_value(first_header, "Columns", "0028|0011", 0)
    frames = int_header_value(first_header, "NumberOfFrames", "0028|0008", 1)

    if rows <= 0 or cols <= 0:
        raise ValueError("missing_rows_or_columns")

    series_uid = header_value(first_header, "SeriesInstanceUID", "0020|000e")
    slices = same_series_file_count(series_dir, series_uid)

    # Numpy-style shape. Singleton axes may be anywhere in real data, so
    # classify_shape removes all dimensions <= 1 before deciding modality.
    if frames > 1 and slices > 1:
        return (frames, slices, rows, cols)
    if frames > 1:
        return (frames, rows, cols)
    return (slices, rows, cols)


def classify_shape(shape: tuple[int, ...]) -> str:
    """Classify true dimensionality after removing singleton axes anywhere."""
    meaningful_dims = tuple(dim for dim in shape if dim > 1)

    if len(meaningful_dims) <= 2:
        return "2d"
    if len(meaningful_dims) == 3:
        return "3d"
    return "4d_5d"


def process_dicom_dir(task: tuple[str, str]):
    series_dir_str, root_dir = task
    series_dir = Path(series_dir_str)
    path_str = str(series_dir)
    patient_id = patient_id_from_rad_path(path_str, root_dir)

    try:
        first_file = first_dicom_file(series_dir)
        if first_file is None:
            return path_str, patient_id, None, "error"

        header = read_header(first_file)
        shape = infer_shape_from_headers(series_dir, header)
        return path_str, patient_id, shape, classify_shape(shape)
    except Exception:
        return path_str, patient_id, None, "error"


def add_result(
    path_str: str,
    patient_id: str,
    shape: tuple[int, ...] | None,
    status: str,
    seen_patient_ids: set[str],
    new_processed: list,
    new_errors: list,
    new_2d: list,
    new_3d: list,
    new_4d: list,
) -> None:
    if patient_id in seen_patient_ids:
        return

    seen_patient_ids.add(patient_id)

    if status == "2d":
        new_processed.append(path_str)
        new_2d.append((path_str, shape))
    elif status == "3d":
        new_processed.append(path_str)
        new_3d.append((path_str, shape))
    elif status == "4d_5d":
        new_processed.append(path_str)
        new_4d.append((path_str, shape))
    else:
        new_errors.append(path_str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast DICOM scanner")
    parser.add_argument("--root-dir", default=DEFAULT_ROOT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument(
        "--cohort",
        choices=sorted(RAD_COHORTS),
        action="append",
        default=None,
        help=(
            "Scan only this cohort. Repeat for multiple cohorts. "
            "Omit to scan every cohort under --root-dir."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"--- Fast DICOM Scanner ({args.max_workers} workers) ---")
    data_stores, processed_patient_ids, processed_by_cohort = load_existing(
        args.output_dir, args.root_dir
    )
    print(f"Loaded {len(processed_patient_ids)} previously processed patient IDs.")
    for cohort in sorted(RAD_COHORTS):
        print(f"  {cohort}: {len(processed_by_cohort[cohort])}")
    categorized_count = len(set().union(*processed_by_cohort.values()))
    uncategorized_count = len(processed_patient_ids) - categorized_count
    if uncategorized_count > 0:
        print(f"  uncategorized: {uncategorized_count}")

    scan_label = ", ".join(str(path) for path in scan_roots(args.root_dir, args.cohort))
    print(f"Scanning DICOM files under {scan_label} ...")
    dicom_dirs = iter_dicom_dirs(args.root_dir, args.cohort)
    candidates = [
        str(path)
        for path in dicom_dirs
        if patient_id_from_rad_path(str(path), args.root_dir) not in processed_patient_ids
    ]
    print(f"Found {len(dicom_dirs)} DICOM dirs. {len(candidates)} remain after patient skip.")

    new_2d: list = []
    new_3d: list = []
    new_4d: list = []
    new_processed: list = []
    new_errors: list = []
    seen_patient_ids = set(processed_patient_ids)

    tasks = [(path, args.root_dir) for path in candidates]
    if args.max_workers <= 1:
        iterator = map(process_dicom_dir, tasks)
        for result in tqdm(iterator, total=len(tasks), desc="Processing DICOMs"):
            add_result(*result, seen_patient_ids, new_processed, new_errors, new_2d, new_3d, new_4d)
    elif tasks:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(process_dicom_dir, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing DICOMs"):
                add_result(
                    *future.result(),
                    seen_patient_ids,
                    new_processed,
                    new_errors,
                    new_2d,
                    new_3d,
                    new_4d,
                )

    print("\n--- Summary of New Scans ---")
    print(f"3D Volumes: {len(new_3d)}")
    print(f"2D Images:  {len(new_2d)}")
    print(f"4D/Other:   {len(new_4d)}")
    print(f"Errors:     {len(new_errors)}")

    data_stores["3d"].extend(new_3d)
    data_stores["2d"].extend(new_2d)
    data_stores["4d_5d"].extend(new_4d)
    data_stores["processed"].extend(new_processed)
    data_stores["errors"].extend(new_errors)

    dump_to_json(data_stores["3d"], OUTPUT_FILES["3d"], args.output_dir)
    dump_to_json(data_stores["2d"], OUTPUT_FILES["2d"], args.output_dir)
    dump_to_json(data_stores["4d_5d"], OUTPUT_FILES["4d_5d"], args.output_dir)
    dump_to_json(data_stores["processed"], OUTPUT_FILES["processed"], args.output_dir)
    dump_to_json(data_stores["errors"], OUTPUT_FILES["errors"], args.output_dir)
    dump_to_json(sorted(seen_patient_ids), OUTPUT_FILES["patients"], args.output_dir)


if __name__ == "__main__":
    main()
