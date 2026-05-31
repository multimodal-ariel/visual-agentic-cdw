#!/usr/bin/env python3
"""
Debug SynthSeg installation independently of the clinical batch pipeline.

This script:
  1. Downloads a public MNI152 T1 NIfTI template unless --input-image is given.
  2. Runs upstream SynthSeg_predict.py directly in cdw_synthseg.
  3. Runs the CDW SynthSegTool wrapper on the same image.
  4. Prints/saves a compact report with command, return codes, outputs, shapes.

Use this to separate SynthSeg environment/setup problems from clinical-data
or batch-runner problems.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import nibabel as nib


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.base_tool import ToolInput
from tools.synthseg import SynthSegTool


MNI_URL = "https://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_sym_09a_nifti.zip"
MNI_T1_HINTS = ("t1", "nlin", "sym", "09a")


def resolve_conda_executable() -> str:
    candidates = [
        os.environ.get("CONDA_EXE", ""),
        shutil.which("conda") or "",
        "/home/soumitri/env/miniconda3/bin/conda",
        "/home/sochattopadhyay/miniconda3/bin/conda",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("Could not find conda executable. Set CONDA_EXE or run from a shell with conda on PATH.")


def download_mni_template(work_dir: Path) -> Path:
    archive = work_dir / "mni_icbm152_nlin_sym_09a_nifti.zip"
    extract_dir = work_dir / "mni_icbm152_nlin_sym_09a"
    if not archive.is_file():
        print(f"Downloading MNI template: {MNI_URL}")
        urllib.request.urlretrieve(MNI_URL, archive)
    if not extract_dir.is_dir():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_dir)
    matches = [
        path
        for path in extract_dir.rglob("*.nii*")
        if all(hint in path.name.lower() for hint in MNI_T1_HINTS)
        and "mask" not in path.name.lower()
        and "eye" not in path.name.lower()
        and "face" not in path.name.lower()
    ]
    if not matches:
        matches = [
            path
            for path in extract_dir.rglob("*.nii*")
            if "t1" in path.name.lower()
            and "mask" not in path.name.lower()
            and "eye" not in path.name.lower()
            and "face" not in path.name.lower()
        ]
    if not matches:
        available = [str(path.relative_to(extract_dir)) for path in extract_dir.rglob("*.nii*")]
        raise FileNotFoundError(f"Could not find a T1 NIfTI inside {archive}; available={available[:20]}")
    return sorted(matches, key=lambda p: (len(p.name), p.name))[0]


def run_command(cmd: list[str], *, cwd: Path | None, timeout: int | None) -> dict[str, Any]:
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    return {
        "cmd": cmd,
        "cwd": str(cwd) if cwd else "",
        "returncode": proc.returncode,
        "elapsed_s": round(time.time() - t0, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def nifti_info(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "path": str(path)}
    img = nib.load(str(path))
    return {
        "exists": True,
        "path": str(path),
        "shape": tuple(int(x) for x in img.shape[:3]),
        "zooms": tuple(float(x) for x in img.header.get_zooms()[:3]),
    }


def run_upstream(
    *,
    image_path: Path,
    out_dir: Path,
    synthseg_dir: Path,
    conda_env: str,
    robust: bool,
    cpu: bool,
    write_qc: bool,
    crop: int | None,
    timeout: int | None,
) -> dict[str, Any]:
    image_path = image_path.resolve()
    out_dir = out_dir.resolve()
    synthseg_dir = synthseg_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_seg = (out_dir / "upstream_synthseg_raw_1mm.nii.gz").resolve()
    volumes_csv = (out_dir / "upstream_volumes.csv").resolve()
    qc_csv = (out_dir / "upstream_qc.csv").resolve()
    runner = (synthseg_dir / "scripts" / "commands" / "SynthSeg_predict.py").resolve()

    cmd = [
        resolve_conda_executable(),
        "run",
        "-n",
        conda_env,
        "python",
        str(runner),
        "--i",
        str(image_path),
        "--o",
        str(raw_seg),
        "--vol",
        str(volumes_csv),
    ]
    if write_qc:
        cmd.extend(["--qc", str(qc_csv)])
    if robust:
        cmd.append("--robust")
    if cpu:
        cmd.append("--cpu")
    if crop:
        cmd.extend(["--crop", str(crop)])

    result = run_command(cmd, cwd=synthseg_dir, timeout=timeout)
    result["outputs"] = {
        "raw_seg": nifti_info(raw_seg),
        "volumes_csv_exists": volumes_csv.is_file(),
        "qc_csv_exists": qc_csv.is_file(),
    }
    return result


def run_wrapper(
    *,
    image_path: Path,
    out_dir: Path,
    modality: str,
    synthseg_dir: Path,
    robust: bool,
    cpu: bool,
    write_qc: bool,
    timeout: int | None,
) -> dict[str, Any]:
    image_path = image_path.resolve()
    out_dir = out_dir.resolve()
    synthseg_dir = synthseg_dir.resolve()
    case_dir = out_dir / "wrapper_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    case_image = case_dir / "image_nifti.nii.gz"
    if not case_image.exists():
        try:
            os.symlink(image_path, case_image)
        except OSError:
            shutil.copy2(image_path, case_image)

    tool = SynthSegTool(
        dry_run=False,
        synthseg_dir=str(synthseg_dir),
        robust=robust,
        write_qc=write_qc,
    )
    t0 = time.time()
    output = tool.run(
        ToolInput(
            image_path=str(case_image),
            case_path=str(case_dir),
            modality=modality,
            anatomy="head",
            target_organs=["brain"],
            device="cpu" if cpu else "gpu:0",
            timeout_s=timeout,
        )
    )
    seg_dir = Path(output.seg_dir)
    return {
        "success": output.success,
        "error": output.error,
        "elapsed_s": round(time.time() - t0, 3),
        "organs_segmented": output.organs_segmented,
        "statistics": output.statistics,
        "outputs": {
            "brain": nifti_info(seg_dir / "brain.nii.gz"),
            "labels_resampled": nifti_info(seg_dir / "synthseg_labels_resampled.nii.gz"),
            "raw_seg": nifti_info(seg_dir / "synthseg_raw_1mm.nii.gz"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-image", default="", help="Optional existing brain NIfTI to test")
    parser.add_argument("--work-dir", default="debug/synthseg_install", help="Working/output directory")
    parser.add_argument("--synthseg-dir", default=str(REPO_ROOT / "external" / "SynthSeg"))
    parser.add_argument("--conda-env", default="cdw_synthseg")
    parser.add_argument("--modality", choices=["MRI", "CT"], default="MRI")
    parser.add_argument("--no-robust", action="store_true", help="Do not pass --robust")
    parser.add_argument("--cpu", action="store_true", help="Run SynthSeg with --cpu")
    parser.add_argument("--write-qc", action="store_true", help="Also request upstream --qc")
    parser.add_argument("--crop", type=int, default=None, help="Optional upstream --crop size")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--report", default="", help="Optional report JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    synthseg_dir = Path(args.synthseg_dir).resolve()

    image_path = Path(args.input_image).resolve() if args.input_image else download_mni_template(work_dir).resolve()
    report = {
        "image": nifti_info(image_path),
        "options": {
            "robust": not args.no_robust,
            "cpu": args.cpu,
            "write_qc": args.write_qc,
            "crop": args.crop,
            "modality": args.modality,
            "conda_env": args.conda_env,
            "synthseg_dir": str(synthseg_dir),
        },
        "upstream": run_upstream(
            image_path=image_path,
            out_dir=work_dir / "upstream",
            synthseg_dir=synthseg_dir,
            conda_env=args.conda_env,
            robust=not args.no_robust,
            cpu=args.cpu,
            write_qc=args.write_qc,
            crop=args.crop,
            timeout=args.timeout,
        ),
        "wrapper": run_wrapper(
            image_path=image_path,
            out_dir=work_dir,
            modality=args.modality,
            synthseg_dir=synthseg_dir,
            robust=not args.no_robust,
            cpu=args.cpu,
            write_qc=args.write_qc,
            timeout=args.timeout,
        ),
    }

    report_path = Path(args.report) if args.report else work_dir / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return 0 if report["upstream"]["outputs"]["raw_seg"]["exists"] and report["wrapper"]["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
