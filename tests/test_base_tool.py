"""
Tests for tools/base_tool.py

Covers: ToolInput, ToolOutput, is_applicable(), _write_manifest(), _timed_run(),
        postprocess() dispatch. No real images or GPU required.
"""

import json
import os
from datetime import datetime

import numpy as np
import pytest

from tools.base_tool import BaseSegmentationTool, ToolInput, ToolOutput


# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing
# ---------------------------------------------------------------------------

class DummyTool(BaseSegmentationTool):
    name = "DummyTool"
    supported_modalities = ["CT", "MRI"]
    supported_anatomies = ["abdomen", "chest", "pelvis"]
    supports_2d = False
    supports_3d = True

    def __init__(self, succeed: bool = True):
        self.succeed = succeed

    def run(self, inp: ToolInput) -> ToolOutput:
        seg_dir = inp.output_dir or os.path.join(inp.case_path, "segmentations_dummy")
        os.makedirs(seg_dir, exist_ok=True)
        if self.succeed:
            return ToolOutput(
                tool_name=self.name,
                case_path=inp.case_path,
                seg_dir=seg_dir,
                organs_segmented=["liver", "spleen"],
                statistics={"liver": {"volume_mm3": 1_500_000}},
                success=True,
            )
        return ToolOutput(
            tool_name=self.name,
            case_path=inp.case_path,
            seg_dir=seg_dir,
            success=False,
            error="Simulated failure",
        )


class AllAnatomyTool(BaseSegmentationTool):
    """Tool that accepts all anatomies — like TotalSegmentator."""
    name = "AllAnatomyTool"
    supported_modalities = ["CT"]
    supported_anatomies = ["all"]
    supports_2d = False
    supports_3d = True

    def run(self, inp: ToolInput) -> ToolOutput:
        return ToolOutput(tool_name=self.name, case_path=inp.case_path, seg_dir="")


class Supports2DTool(BaseSegmentationTool):
    name = "Supports2DTool"
    supported_modalities = ["CT"]
    supported_anatomies = ["all"]
    supports_2d = True
    supports_3d = False

    def run(self, inp: ToolInput) -> ToolOutput:
        return ToolOutput(tool_name=self.name, case_path=inp.case_path, seg_dir="")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tool():
    return DummyTool(succeed=True)


@pytest.fixture
def failing_tool():
    return DummyTool(succeed=False)


@pytest.fixture
def tool_input(tmp_path):
    seg_dir = str(tmp_path / "segmentations_dummy")
    return ToolInput(
        image_path=str(tmp_path / "image_nifti.nii.gz"),
        case_path=str(tmp_path),
        modality="CT",
        anatomy="abdomen",
        target_organs=["liver", "spleen"],
        output_dir=seg_dir,
        device="gpu:0",
    )


# ---------------------------------------------------------------------------
# ToolOutput defaults
# ---------------------------------------------------------------------------

def test_tool_output_defaults():
    out = ToolOutput(tool_name="X", case_path="/tmp", seg_dir="/tmp/seg")
    assert out.success is True
    assert out.error == ""
    assert out.runtime_seconds == 0.0
    assert out.timestamp == ""
    assert out.organs_segmented == []
    assert out.statistics == {}


# ---------------------------------------------------------------------------
# is_applicable
# ---------------------------------------------------------------------------

def test_is_applicable_match(tool):
    assert tool.is_applicable("CT", "abdomen", "3D") is True
    assert tool.is_applicable("MRI", "chest", "3D") is True
    assert tool.is_applicable("MRI", "pelvis", "3D") is True


def test_is_applicable_wrong_modality(tool):
    assert tool.is_applicable("PET", "abdomen", "3D") is False
    assert tool.is_applicable("US", "abdomen", "3D") is False


def test_is_applicable_wrong_anatomy(tool):
    assert tool.is_applicable("CT", "head", "3D") is False
    assert tool.is_applicable("CT", "whole_body", "3D") is False


def test_is_applicable_2d_not_supported(tool):
    assert tool.is_applicable("CT", "abdomen", "2D") is False


def test_is_applicable_4d_treated_as_3d(tool):
    assert tool.is_applicable("CT", "abdomen", "4D") is True


def test_is_applicable_case_insensitive(tool):
    assert tool.is_applicable("ct", "Abdomen", "3D") is True
    assert tool.is_applicable("CT", "CHEST", "3D") is True


def test_is_applicable_all_anatomy():
    t = AllAnatomyTool()
    assert t.is_applicable("CT", "head", "3D") is True
    assert t.is_applicable("CT", "whole_body", "3D") is True
    assert t.is_applicable("CT", "anything", "3D") is True


def test_is_applicable_2d_tool():
    t = Supports2DTool()
    assert t.is_applicable("CT", "chest", "2D") is True
    assert t.is_applicable("CT", "chest", "3D") is False


# ---------------------------------------------------------------------------
# _write_manifest
# ---------------------------------------------------------------------------

def test_write_manifest_creates_file(tool, tmp_path):
    seg_dir = str(tmp_path / "seg")
    os.makedirs(seg_dir)
    out = ToolOutput(
        tool_name="DummyTool",
        case_path=str(tmp_path),
        seg_dir=seg_dir,
        organs_segmented=["liver"],
        success=True,
        runtime_seconds=10.0,
        timestamp="2026-03-20T03:00:00",
    )
    tool._write_manifest(out)
    assert os.path.isfile(os.path.join(seg_dir, "manifest.json"))


def test_write_manifest_content(tool, tmp_path):
    seg_dir = str(tmp_path / "seg")
    os.makedirs(seg_dir)
    out = ToolOutput(
        tool_name="DummyTool",
        case_path=str(tmp_path),
        seg_dir=seg_dir,
        organs_segmented=["liver", "spleen"],
        statistics={"liver": {"volume_mm3": 1500}},
        success=True,
        runtime_seconds=42.7,
        timestamp="2026-03-20T03:00:00",
        error="",
    )
    tool._write_manifest(out)
    with open(os.path.join(seg_dir, "manifest.json")) as f:
        m = json.load(f)
    assert m["tool_name"] == "DummyTool"
    assert m["success"] is True
    assert m["organs_segmented"] == ["liver", "spleen"]
    assert m["runtime_seconds"] == 42.7
    assert m["timestamp"] == "2026-03-20T03:00:00"
    assert m["native_stats"] == {"liver": {"volume_mm3": 1500}}
    assert m["error"] == ""


def test_write_manifest_failure_case(tool, tmp_path):
    seg_dir = str(tmp_path / "seg")
    os.makedirs(seg_dir)
    out = ToolOutput(
        tool_name="DummyTool",
        case_path=str(tmp_path),
        seg_dir=seg_dir,
        success=False,
        error="GPU out of memory",
        timestamp="2026-03-20T03:00:00",
    )
    tool._write_manifest(out)
    with open(os.path.join(seg_dir, "manifest.json")) as f:
        m = json.load(f)
    assert m["success"] is False
    assert m["error"] == "GPU out of memory"
    assert m["organs_segmented"] == []


def test_write_manifest_creates_seg_dir_if_missing(tool, tmp_path):
    """Even for failed runs where seg_dir was never created."""
    seg_dir = str(tmp_path / "nonexistent" / "seg")
    out = ToolOutput(
        tool_name="DummyTool",
        case_path=str(tmp_path),
        seg_dir=seg_dir,
        success=False,
        error="Tool crashed before mkdir",
        timestamp="2026-03-20T03:00:00",
    )
    tool._write_manifest(out)
    assert os.path.isfile(os.path.join(seg_dir, "manifest.json"))


def test_write_manifest_empty_seg_dir_is_noop(tool):
    """seg_dir='' should not raise."""
    out = ToolOutput(tool_name="X", case_path="/tmp", seg_dir="")
    tool._write_manifest(out)  # should not raise


# ---------------------------------------------------------------------------
# _timed_run
# ---------------------------------------------------------------------------

def test_timed_run_sets_runtime(tool, tool_input):
    out = tool._timed_run(tool_input)
    assert out.runtime_seconds > 0.0


def test_timed_run_sets_timestamp(tool, tool_input):
    out = tool._timed_run(tool_input)
    assert out.timestamp != ""
    datetime.fromisoformat(out.timestamp)  # raises ValueError if invalid format


def test_timed_run_writes_manifest(tool, tool_input):
    out = tool._timed_run(tool_input)
    assert os.path.isfile(os.path.join(out.seg_dir, "manifest.json"))


def test_timed_run_manifest_runtime_matches(tool, tool_input):
    out = tool._timed_run(tool_input)
    with open(os.path.join(out.seg_dir, "manifest.json")) as f:
        m = json.load(f)
    assert m["runtime_seconds"] == round(out.runtime_seconds, 2)


def test_timed_run_failing_tool_writes_manifest(failing_tool, tool_input):
    out = failing_tool._timed_run(tool_input)
    assert out.success is False
    assert os.path.isfile(os.path.join(out.seg_dir, "manifest.json"))
    with open(os.path.join(out.seg_dir, "manifest.json")) as f:
        m = json.load(f)
    assert m["success"] is False
    assert "Simulated failure" in m["error"]


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

def test_repr_contains_name(tool):
    r = repr(tool)
    assert "DummyTool" in r
    assert "CT" in r
    assert "MRI" in r
