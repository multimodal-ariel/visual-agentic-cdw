import json

from orchestrator.case_tracker import CaseTracker


def test_case_tracker_persists_extra_metadata(tmp_path):
    state_path = tmp_path / "state.json"
    tracker = CaseTracker(str(state_path))
    tracker.register_cases(["CASE"])

    tracker.set_status(
        "CASE",
        "completed",
        tools_run=["SynthSeg"],
        extra={
            "nifti_materialization": {
                "status": "converted",
                "source_series_count": 2,
                "selected_series_uid": "1.2.3",
                "series_selection_reason": "largest_valid_series",
            }
        },
    )

    state = json.loads(state_path.read_text())
    entry = state["cases"]["CASE"]
    assert entry["status"] == "completed"
    assert entry["tools_run"] == ["SynthSeg"]
    assert entry["nifti_materialization"]["status"] == "converted"
    assert entry["nifti_materialization"]["selected_series_uid"] == "1.2.3"


def test_case_tracker_clears_stale_error_on_successful_retry(tmp_path):
    state_path = tmp_path / "state.json"
    tracker = CaseTracker(str(state_path))
    tracker.register_cases(["CASE"])

    tracker.set_status("CASE", "failed", error="old failure")
    tracker.set_status("CASE", "completed", tools_run=["SynthSeg"])

    state = json.loads(state_path.read_text())
    entry = state["cases"]["CASE"]
    assert entry["status"] == "completed"
    assert entry["error"] == ""
    assert entry["tools_run"] == ["SynthSeg"]
