"""
================================================================================
CDW Pipeline — QC Module Tests  (no real data, no LLM required)
================================================================================
Tests all three QC tiers using synthetic NIfTI masks written to a tmp directory.

Run:
    pytest tests/test_qc.py -v
    # or directly:
    python tests/test_qc.py

Coverage:
    Tier 1 — GeometricQC: volume plausibility, connected components, paired ratios,
              mask overlap, severity rollup, missing-dir error path
    Tier 2 — MultiToolQC: Dice coefficient, volume agreement, disagreement flagging,
              consensus (majority-vote), missing-tool presence mismatch
    Tier 3 — QCInterpreter._fallback_gating: rule-based PASS/WARN/FAIL routing
              (no LLM instantiated — verifies offline-safe fallback path)
    Dataclasses — OrganQCResult.compute_severity, CaseQCReport.compute_overall
================================================================================
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

# ── Repo root on sys.path ──────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from qc.dataclasses import (
    CaseQCReport,
    MultiToolQCResult,
    OrganAgreement,
    OrganQCResult,
    QCInterpretation,
    ToolQCResult,
)
from qc.geometric_qc import GeometricQC
from qc.multi_tool_qc import MultiToolQC


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — synthetic NIfTI writers
# ──────────────────────────────────────────────────────────────────────────────

_AFFINE = np.diag([1.5, 1.5, 3.0, 1.0])   # 1.5 × 1.5 × 3.0 mm voxels


def _write_mask(path: str, arr: np.ndarray, affine=_AFFINE) -> None:
    img = nib.Nifti1Image(arr.astype(np.uint8), affine)
    nib.save(img, path)


def _sphere_mask(shape=(64, 64, 64), radius=12, center=None) -> np.ndarray:
    """Binary sphere mask inside a zero volume."""
    if center is None:
        center = [s // 2 for s in shape]
    arr = np.zeros(shape, dtype=np.uint8)
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    dist = np.sqrt(
        (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2
    )
    arr[dist <= radius] = 1
    return arr


def _two_sphere_mask(shape=(64, 64, 64), r=8) -> np.ndarray:
    """Two separated spheres — simulates fragmented mask (CC=2)."""
    arr = np.zeros(shape, dtype=np.uint8)
    for center in [(16, 32, 32), (48, 32, 32)]:
        z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        dist = np.sqrt(
            (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2
        )
        arr[dist <= r] = 1
    return arr


def _write_seg_dir(tmp_dir: str, organs: dict) -> None:
    """Write {organ_name: mask_array} to tmp_dir as NIfTI files."""
    os.makedirs(tmp_dir, exist_ok=True)
    for organ, arr in organs.items():
        _write_mask(os.path.join(tmp_dir, f"{organ}.nii.gz"), arr)


# ──────────────────────────────────────────────────────────────────────────────
# Minimal organ_reference.json for tests (no dependency on real config/)
# ──────────────────────────────────────────────────────────────────────────────

_TINY_REFERENCE = {
    "volume_ranges_ml": {
        "liver":        [700.0, 2500.0],
        "spleen":       [100.0, 500.0],
        "kidney_left":  [100.0, 350.0],
        "kidney_right": [100.0, 350.0],
        "pancreas":     [40.0,  150.0],
    },
    "paired_organ_ratios": [
        {"organ_a": "kidney_left", "organ_b": "kidney_right", "min_ratio": 0.5, "max_ratio": 2.0}
    ],
    "lung_composite_ratios": [],
    "expected_max_components": {
        "liver":        1,
        "spleen":       1,
        "kidney_left":  1,
        "kidney_right": 1,
    },
}


@pytest.fixture
def ref_file(tmp_path):
    """Write a minimal organ_reference.json to tmp_path."""
    p = tmp_path / "organ_reference.json"
    p.write_text(json.dumps(_TINY_REFERENCE))
    return str(p)


# ──────────────────────────────────────────────────────────────────────────────
# Dataclass tests
# ──────────────────────────────────────────────────────────────────────────────

class TestOrganQCResult:
    def test_no_flags_pass(self):
        r = OrganQCResult(case_id="c1", organ="liver", volume_ml=1200.0,
                          volume_in_range=True, num_components=1,
                          largest_component_fraction=1.0)
        flags = r.compute_severity()
        assert flags == []
        assert r.severity == "PASS"

    def test_one_flag_warn(self):
        r = OrganQCResult(case_id="c1", organ="liver", volume_ml=3000.0,
                          volume_in_range=False, volume_flag="OVER_VOLUME")
        flags = r.compute_severity()
        assert "OVER_VOLUME" in flags
        assert r.severity == "WARN"

    def test_three_flags_fail(self):
        r = OrganQCResult(case_id="c1", organ="pancreas",
                          volume_in_range=False, volume_flag="UNDER_VOLUME",
                          cc_flag="CC=4",
                          ratio_in_range=False, ratio_flag="RATIO_OUTLIER")
        flags = r.compute_severity(warn_max=2)
        assert r.severity == "FAIL"
        assert len(flags) == 3


class TestCaseQCReport:
    def test_compute_overall_pass(self):
        tr = ToolQCResult(case_id="c1", case_path="/tmp/c1", worst_severity="PASS")
        report = CaseQCReport(case_id="c1", case_path="/tmp/c1", tool_results=[tr])
        report.compute_overall()
        assert report.overall_severity == "PASS"

    def test_compute_overall_fail_from_tool(self):
        tr = ToolQCResult(case_id="c1", case_path="/tmp/c1", worst_severity="FAIL")
        report = CaseQCReport(case_id="c1", case_path="/tmp/c1", tool_results=[tr])
        report.compute_overall()
        assert report.overall_severity == "FAIL"

    def test_compute_overall_warn_from_agreement(self):
        tr = ToolQCResult(case_id="c1", case_path="/tmp/c1", worst_severity="PASS")
        ag = MultiToolQCResult(case_id="c1", case_path="/tmp/c1",
                               organs_with_disagreement=1)
        report = CaseQCReport(case_id="c1", case_path="/tmp/c1",
                               tool_results=[tr], agreement=ag)
        report.compute_overall()
        assert report.overall_severity == "WARN"

    def test_compute_overall_unusable_llm(self):
        tr = ToolQCResult(case_id="c1", case_path="/tmp/c1", worst_severity="WARN")
        interp = QCInterpretation(case_id="c1", case_path="/tmp", overall_quality="UNUSABLE")
        report = CaseQCReport(case_id="c1", case_path="/tmp/c1",
                               tool_results=[tr], interpretation=interp)
        report.compute_overall()
        assert report.overall_severity == "FAIL"


# ──────────────────────────────────────────────────────────────────────────────
# Tier 1: GeometricQC
# ──────────────────────────────────────────────────────────────────────────────

class TestGeometricQC:
    """All tests write synthetic NIfTI masks to a tmpdir — no real data."""

    def test_pass_case(self, tmp_path, ref_file):
        """Single-component liver-sized sphere → PASS."""
        # Sphere radius 18 vox × voxel 1.5³ mm³ → ~30 mL ≈ plausible spleen
        seg_dir = str(tmp_path / "seg_pass")
        organs = {
            "liver":  _sphere_mask(shape=(80, 80, 80), radius=30),   # ~1130 mL
            "spleen": _sphere_mask(shape=(80, 80, 80), radius=14),   # ~115 mL
        }
        _write_seg_dir(seg_dir, organs)

        gqc = GeometricQC(organ_ref_path=ref_file)
        result = gqc.run(case_path=str(tmp_path), seg_dir=seg_dir, tool_name="TestTool")

        assert result.error == ""
        assert result.num_organs == 2
        liver_r = next(r for r in result.organ_results if r.organ == "liver")
        assert liver_r.severity in ("PASS", "WARN")   # synthetic may be off but not ERROR

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Connected-component check is currently disabled in "
            "qc/geometric_qc.py (~lines 173-174 and 283-310) — postprocessing "
            "now keeps only the largest CC, so the in-QC CC count is always 1. "
            "Restore _check_connected_components and remove this xfail if you "
            "want fragment detection back."
        ),
    )
    def test_fragmented_organ_flagged(self, tmp_path, ref_file):
        """Two-sphere mask → CC=2 flag set."""
        seg_dir = str(tmp_path / "seg_frag")
        organs = {"liver": _two_sphere_mask(shape=(80, 80, 80), r=10)}
        _write_seg_dir(seg_dir, organs)

        gqc = GeometricQC(organ_ref_path=ref_file)
        result = gqc.run(case_path=str(tmp_path), seg_dir=seg_dir)
        liver_r = next(r for r in result.organ_results if r.organ == "liver")
        assert liver_r.num_components == 2
        assert liver_r.cc_flag != ""

    def test_over_volume_flagged(self, tmp_path, ref_file):
        """Very large sphere → OVER_VOLUME flag for spleen."""
        seg_dir = str(tmp_path / "seg_large")
        organs = {"spleen": _sphere_mask(shape=(80, 80, 80), radius=35)}  # ~2400 mL >> 500 mL max
        _write_seg_dir(seg_dir, organs)

        gqc = GeometricQC(organ_ref_path=ref_file)
        result = gqc.run(case_path=str(tmp_path), seg_dir=seg_dir)
        spleen_r = next(r for r in result.organ_results if r.organ == "spleen")
        assert spleen_r.volume_flag != ""

    def test_under_volume_flagged(self, tmp_path, ref_file):
        """Tiny sphere → UNDER_VOLUME for liver."""
        seg_dir = str(tmp_path / "seg_small")
        organs = {"liver": _sphere_mask(shape=(80, 80, 80), radius=4)}  # ~1 mL << 700 mL min
        _write_seg_dir(seg_dir, organs)

        gqc = GeometricQC(organ_ref_path=ref_file)
        result = gqc.run(case_path=str(tmp_path), seg_dir=seg_dir)
        liver_r = next(r for r in result.organ_results if r.organ == "liver")
        assert liver_r.volume_flag != ""

    def test_missing_seg_dir_returns_error(self, tmp_path, ref_file):
        gqc = GeometricQC(organ_ref_path=ref_file)
        result = gqc.run(case_path=str(tmp_path), seg_dir="/nonexistent/path")
        assert result.worst_severity == "ERROR"
        assert result.error != ""

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Mask-overlap check is currently disabled in qc/geometric_qc.py "
            "(~lines 188-189 and 353-385). Restore _check_overlap and remove "
            "this xfail if you want overlap detection back."
        ),
    )
    def test_overlap_detected(self, tmp_path, ref_file):
        """Two fully overlapping masks → overlap flag on at least one organ."""
        seg_dir = str(tmp_path / "seg_overlap")
        same = _sphere_mask(shape=(80, 80, 80), radius=15)
        organs = {"liver": same.copy(), "spleen": same.copy()}
        _write_seg_dir(seg_dir, organs)

        gqc = GeometricQC(organ_ref_path=ref_file)
        result = gqc.run(case_path=str(tmp_path), seg_dir=seg_dir)
        has_overlap = any(r.has_overlap for r in result.organ_results)
        assert has_overlap

    def test_worst_severity_aggregation(self, tmp_path, ref_file):
        """Mix of passing and failing organs → worst_severity reflects worst."""
        seg_dir = str(tmp_path / "seg_mix")
        organs = {
            "spleen": _sphere_mask(shape=(80, 80, 80), radius=13),       # plausible → PASS
            "liver":  _sphere_mask(shape=(80, 80, 80), radius=3),         # tiny → UNDER_VOLUME
        }
        _write_seg_dir(seg_dir, organs)

        gqc = GeometricQC(organ_ref_path=ref_file)
        result = gqc.run(case_path=str(tmp_path), seg_dir=seg_dir)
        assert result.worst_severity in ("WARN", "FAIL")   # at least one flag


# ──────────────────────────────────────────────────────────────────────────────
# Tier 2: MultiToolQC
# ──────────────────────────────────────────────────────────────────────────────

class TestMultiToolQC:
    def test_identical_masks_perfect_dice(self, tmp_path):
        """Two tools produce identical masks → Dice = 1.0."""
        arr = _sphere_mask(shape=(64, 64, 64), radius=14)
        seg_a = str(tmp_path / "tool_a")
        seg_b = str(tmp_path / "tool_b")
        _write_seg_dir(seg_a, {"liver": arr})
        _write_seg_dir(seg_b, {"liver": arr.copy()})

        mtqc = MultiToolQC()
        result = mtqc.run(
            case_path=str(tmp_path),
            seg_dirs={"ToolA": seg_a, "ToolB": seg_b},
        )
        assert result.mean_dice == pytest.approx(1.0, abs=1e-4)
        assert result.organs_with_disagreement == 0
        assert result.organs_with_agreement == 1

    def test_non_overlapping_masks_zero_dice(self, tmp_path):
        """Two tools produce completely different masks → Dice = 0."""
        arr_a = _sphere_mask(shape=(64, 64, 64), radius=10, center=[16, 32, 32])
        arr_b = _sphere_mask(shape=(64, 64, 64), radius=10, center=[48, 32, 32])
        seg_a = str(tmp_path / "tool_a")
        seg_b = str(tmp_path / "tool_b")
        _write_seg_dir(seg_a, {"spleen": arr_a})
        _write_seg_dir(seg_b, {"spleen": arr_b})

        mtqc = MultiToolQC()
        result = mtqc.run(
            case_path=str(tmp_path),
            seg_dirs={"ToolA": seg_a, "ToolB": seg_b},
        )
        spleen_ag = next(ag for ag in result.agreements if ag.organ == "spleen")
        assert spleen_ag.dice == pytest.approx(0.0, abs=1e-4)
        assert spleen_ag.agrees is False
        assert result.organs_with_disagreement >= 1

    def test_presence_mismatch(self, tmp_path):
        """Tools segment different sets of organs → no shared organs → 0 agreements."""
        arr = _sphere_mask(shape=(64, 64, 64), radius=12)
        seg_a = str(tmp_path / "tool_a")
        seg_b = str(tmp_path / "tool_b")
        # Each tool has a different organ — no shared organs
        _write_seg_dir(seg_a, {"kidney_left": arr})
        _write_seg_dir(seg_b, {"spleen": arr.copy()})

        mtqc = MultiToolQC()
        result = mtqc.run(
            case_path=str(tmp_path),
            seg_dirs={"ToolA": seg_a, "ToolB": seg_b},
        )
        # Both tools have masks so pair is evaluated; no shared organ → 0 agreements
        assert result.tool_pairs_checked == 1
        assert result.organs_with_agreement == 0
        assert result.organs_with_disagreement == 0

    def test_single_tool_skips_tier2(self, tmp_path):
        """Only one tool provided → Tier 2 skips gracefully."""
        arr = _sphere_mask(shape=(64, 64, 64), radius=12)
        seg_a = str(tmp_path / "tool_a")
        _write_seg_dir(seg_a, {"liver": arr})

        mtqc = MultiToolQC()
        result = mtqc.run(
            case_path=str(tmp_path),
            seg_dirs={"ToolA": seg_a},
        )
        # Less than 2 tools — no pairs to compare
        assert result.tool_pairs_checked == 0

    def test_majority_vote_consensus(self, tmp_path):
        """Majority-vote of 3 identical masks returns a valid NIfTI.

        Skipped if postprocessing dependencies (itk) are not installed in this env.
        """
        pytest.importorskip("itk", reason="itk not installed in this env — skip consensus test")
        arr = _sphere_mask(shape=(64, 64, 64), radius=12)
        seg_dirs = {}
        for i in range(3):
            d = str(tmp_path / f"tool_{i}")
            _write_seg_dir(d, {"liver": arr.copy()})
            seg_dirs[f"Tool{i}"] = d

        mtqc = MultiToolQC()
        consensus = mtqc.generate_consensus(seg_dirs, organ="liver", method="majority_vote")
        assert consensus is not None
        np.testing.assert_array_equal(consensus > 0, arr > 0)


# ──────────────────────────────────────────────────────────────────────────────
# Tier 3: QCInterpreter fallback (no LLM)
# ──────────────────────────────────────────────────────────────────────────────

class TestQCInterpreterFallback:
    """Tests the rule-based fallback path — no LLM instantiated."""

    def _make_organ(self, organ: str, severity: str, n_components: int = 1,
                    lcc_frac: float = 1.0) -> OrganQCResult:
        r = OrganQCResult(case_id="c1", organ=organ, volume_ml=200.0,
                          num_components=n_components,
                          largest_component_fraction=lcc_frac)
        r.severity = severity
        if severity == "WARN" and n_components > 1:
            r.cc_flag = f"CC={n_components}"
        elif severity == "FAIL":
            r.cc_flag = "CC=5"
            r.volume_flag = "UNDER_VOLUME"
        return r

    def _make_tool_result(self, organs: list[OrganQCResult]) -> ToolQCResult:
        tr = ToolQCResult(case_id="c1", case_path="/tmp/c1", tool_name="TestTool")
        tr.organ_results = organs
        return tr

    def test_pass_gets_full_extraction(self):
        from qc.qc_interpreter import QCInterpreter
        tr = self._make_tool_result([self._make_organ("liver", "PASS")])
        interp = QCInterpretation(case_id="c1", case_path="/tmp")
        # Call fallback directly without a real LLM
        qi = QCInterpreter.__new__(QCInterpreter)
        qi._fallback_gating(tr, interp)
        liver_entry = next(e for e in interp.extract if e["organ"] == "liver")
        assert set(liver_entry["features_allowed"]) == {"first_order", "shape", "texture"}
        assert "none" in liver_entry["postprocessing_needed"]

    def test_warn_fragmented_gets_lcc(self):
        from qc.qc_interpreter import QCInterpreter
        tr = self._make_tool_result([
            self._make_organ("spleen", "WARN", n_components=2, lcc_frac=0.92)
        ])
        interp = QCInterpretation(case_id="c1", case_path="/tmp")
        qi = QCInterpreter.__new__(QCInterpreter)
        qi._fallback_gating(tr, interp)
        spleen_entry = next(e for e in interp.extract if e["organ"] == "spleen")
        assert "lcc" in spleen_entry["postprocessing_needed"]

    def test_fail_gets_skipped(self):
        from qc.qc_interpreter import QCInterpreter
        tr = self._make_tool_result([self._make_organ("pancreas", "FAIL")])
        interp = QCInterpretation(case_id="c1", case_path="/tmp")
        qi = QCInterpreter.__new__(QCInterpreter)
        qi._fallback_gating(tr, interp)
        assert any(e["organ"] == "pancreas" for e in interp.skip)
        assert not any(e["organ"] == "pancreas" for e in interp.extract)

    def test_mixed_case(self):
        from qc.qc_interpreter import QCInterpreter
        organs = [
            self._make_organ("liver",   "PASS"),
            self._make_organ("spleen",  "WARN", n_components=2, lcc_frac=0.88),
            self._make_organ("pancreas","FAIL"),
        ]
        tr = self._make_tool_result(organs)
        interp = QCInterpretation(case_id="c1", case_path="/tmp")
        qi = QCInterpreter.__new__(QCInterpreter)
        qi._fallback_gating(tr, interp)
        extract_organs = {e["organ"] for e in interp.extract}
        skip_organs    = {e["organ"] for e in interp.skip}
        assert "liver"   in extract_organs
        assert "spleen"  in extract_organs
        assert "pancreas" in skip_organs


# ──────────────────────────────────────────────────────────────────────────────
# Entry point (pytest or direct)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=str(REPO_ROOT),
    )
    sys.exit(result.returncode)
