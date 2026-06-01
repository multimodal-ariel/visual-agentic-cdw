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
            "source_series_count": 2,
            "selected_series_uid": "1.2.3",
            "series_selection_reason": "largest_valid_series",
        }

    monkeypatch.setattr(batch_runner_module, "ensure_nifti_for_case", fake_ensure)

    result = runner._ensure_case_image("CASE", str(case_path))

    assert result.status == "completed"
    assert calls["case_path"] == str(case_path)
    assert calls["dicom_path"] == str(case_path)
    assert os.path.isfile(case_path / "image_nifti.nii.gz")
    assert result.metadata["nifti_materialization"]["status"] == "converted"
    assert result.metadata["nifti_materialization"]["selected_series_uid"] == "1.2.3"


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


def test_batch_runner_defaults_to_v3_state_and_output(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_runner_module, "LOG_DIR", str(tmp_path / "logs"))
    runner = BatchRunner(
        filelist=str(tmp_path / "filelist.json"),
        dry_run=True,
        no_llm=True,
    )

    assert runner.state_file.endswith("pipeline_state_v3.json")
    assert runner.output_csv.endswith("pipeline_results_v3.csv")


def test_batch_runner_reprocesses_stale_running_cases(tmp_path, monkeypatch):
    filelist = tmp_path / "filelist.json"
    case_path = tmp_path / "case"
    case_path.mkdir()
    filelist.write_text(json.dumps([str(case_path)]))
    runner = BatchRunner(
        filelist=str(filelist),
        state_file=str(tmp_path / "state.json"),
        output_csv=str(tmp_path / "results.csv"),
        dry_run=True,
        no_llm=True,
        progress_every=0,
    )

    case_id = next(iter(runner.tracker._state["cases"])) if runner.tracker._state.get("cases") else None
    if case_id is None:
        from orchestrator.case_id import derive_case_id

        case_id = derive_case_id(str(case_path))
        runner.tracker.register_cases([case_id])
    runner.tracker.set_status(case_id, "running")

    called = {}

    def fake_run_one(cid, path):
        called["cid"] = cid
        from orchestrator.pipeline import CaseResult

        runner.tracker.set_status(cid, "skipped")
        return CaseResult(case_id=cid, case_path=path, status="skipped")

    monkeypatch.setattr(runner, "_run_one_case", fake_run_one)

    summary = runner.run()

    assert called["cid"] == case_id
    assert summary["skipped"] == 1
