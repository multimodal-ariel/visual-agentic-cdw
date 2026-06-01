import json
import os

import orchestrator.batch_runner as batch_runner_module
from config.constants import DICOM_ROOT, SEGMENTATION_ROOT
from orchestrator.batch_runner import BatchRunner


def _runner(tmp_path):
    filelist = tmp_path / "filelist.json"
    filelist.write_text(json.dumps([]))
    return BatchRunner(
        filelist=str(filelist),
        state_file=str(tmp_path / "state.json"),
        output_csv=str(tmp_path / "results.csv"),
        dry_run=True,
        no_llm=True,
    )


def test_dicom_path_for_case_maps_segmentation_root_to_raw_root():
    case_path = f"{SEGMENTATION_ROOT}/LUPUS/PATIENT/20240101/CT_HEAD/Head__5_0"

    dicom_path = BatchRunner._dicom_path_for_case(case_path)

    assert dicom_path == f"{DICOM_ROOT}/LUPUS/PATIENT/20240101/CT_HEAD/Head__5_0"


def test_ensure_case_image_uses_lazy_materialization(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    case_path = tmp_path / "case"
    case_path.mkdir()
    calls = {}

    def fake_ensure(case_path_arg, dicom_path_arg, *, image_filename):
        calls["case_path"] = str(case_path_arg)
        calls["dicom_path"] = str(dicom_path_arg)
        image_path = os.path.join(case_path_arg, image_filename)
        open(image_path, "wb").close()
        return {
            "status": "converted",
            "image_path": image_path,
            "dicom_path": str(dicom_path_arg),
        }

    monkeypatch.setattr(batch_runner_module, "ensure_nifti_for_case", fake_ensure)

    result = runner._ensure_case_image("CASE", str(case_path))

    assert result.status == "completed"
    assert calls["case_path"] == str(case_path)
    assert calls["dicom_path"] == str(case_path)
    assert os.path.isfile(case_path / "image_nifti.nii.gz")


def test_ensure_case_image_reports_missing_dicom_as_failed(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    case_path = tmp_path / "case"
    case_path.mkdir()

    def fake_ensure(case_path_arg, dicom_path_arg, *, image_filename):
        return {
            "status": "missing_dicom",
            "image_path": os.path.join(case_path_arg, image_filename),
            "dicom_path": str(dicom_path_arg),
            "error": "DICOM directory not found",
        }

    monkeypatch.setattr(batch_runner_module, "ensure_nifti_for_case", fake_ensure)

    result = runner._ensure_case_image("CASE", str(case_path))

    assert result.status == "failed"
    assert "DICOM directory not found" in result.error
