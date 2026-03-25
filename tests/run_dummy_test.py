"""
Dummy Data Test Runner
=======================
Runs one or more segmentation tool wrappers on the dummy_outputs/ cases
and prints a summary table of results.

Usage:
    python scripts/run_dummy_test.py --tool totalseg_ct
    python scripts/run_dummy_test.py --tool totalseg_mr
    python scripts/run_dummy_test.py --tool mrseg
    python scripts/run_dummy_test.py --tool all --modality CT
    python scripts/run_dummy_test.py --tool totalseg_ct --cases 0002 0005

The script derives modality from case naming: 00X = MRI, 00XX = CT.
Outputs land in dummy_outputs/<case_id>/segmentations_<tool_name>/.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import List, Optional

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from tools.base_tool import ToolInput, ToolOutput

DUMMY_ROOT = os.path.join(REPO_ROOT, "dummy_outputs")


def modality_from_id(case_id: str) -> str:
    """00X → MRI, 00XX → CT (based on naming convention in dummy_data/)."""
    return "MRI" if len(case_id) == 3 else "CT"


def anatomy_from_modality(modality: str) -> str:
    return "abdomen_pelvis"  # all dummy cases are abdominal


def get_all_cases() -> List[str]:
    return sorted(d for d in os.listdir(DUMMY_ROOT)
                  if os.path.isdir(os.path.join(DUMMY_ROOT, d)))


def build_tool(tool_name: str, dry_run: bool = False):
    if tool_name == "totalseg_ct":
        from tools.totalsegmentator_ct import TotalSegmentatorCTTool
        return TotalSegmentatorCTTool(dry_run=dry_run)
    elif tool_name == "totalseg_mr":
        from tools.totalsegmentator_mr import TotalSegmentatorMRTool
        return TotalSegmentatorMRTool(dry_run=dry_run)
    elif tool_name == "mrseg":
        from tools.mrsegmentator import MRSegmentatorTool
        return MRSegmentatorTool(dry_run=dry_run)
    elif tool_name == "mriseg":
        from tools.mrisegmenter import MRISegmenterTool
        return MRISegmenterTool(dry_run=dry_run)
    elif tool_name == "vibeseg":
        from tools.vibesegmentator import VIBESegmentatorTool
        return VIBESegmentatorTool(dry_run=dry_run)
    elif tool_name == "voxtell":
        from tools.voxtell import VoxTellTool
        return VoxTellTool(dry_run=dry_run)
    elif tool_name == "biomedparse3d":
        from tools.biomedparse3d import BiomedParse3DTool
        return BiomedParse3DTool(dry_run=dry_run)
    elif tool_name == "textmedseg3d":
        from tools.textmedseg3d import TextMedSeg3DTool
        return TextMedSeg3DTool(dry_run=dry_run)
    elif tool_name == "vista3d":
        from tools.vista3d import VISTA3DTool
        return VISTA3DTool(dry_run=dry_run, use_modality_preset=True)
    else:
        raise ValueError(f"Unknown tool: {tool_name}")


TOOL_MODALITY = {
    "totalseg_ct": ["CT"],
    "totalseg_mr": ["MRI"],
    "mrseg":       ["CT", "MRI"],
    "mriseg":      ["MRI"],
    "vibeseg":     ["CT", "MRI"],
    "voxtell":     ["CT", "MRI", "PET_CT"],
    "biomedparse3d": ["CT", "MRI", "PET_CT", "US"],
    "textmedseg3d":  ["CT", "MRI", "PET_CT"],
    "vista3d":       ["CT", "MRI"],
}

ALL_TOOLS = list(TOOL_MODALITY.keys())

# Default target organs for text-promptable tools
DEFAULT_ORGANS = ["liver", "spleen", "kidney_left", "kidney_right", "pancreas",
                  "gallbladder", "aorta", "stomach"]


def run_cases(
    tool_name: str,
    cases: List[str],
    device: str = "gpu:0",
    dry_run: bool = False,
    modality_filter: Optional[str] = None,
):
    tool = build_tool(tool_name, dry_run=dry_run)
    supported_mods = TOOL_MODALITY[tool_name]

    print(f"\n{'='*70}")
    print(f"Tool: {tool.name}  |  conda_env: {tool.conda_env}  |  dry_run={dry_run}")
    print(f"{'='*70}")

    results = []
    for case_id in cases:
        modality = modality_from_id(case_id)

        # Skip if modality not supported
        if modality not in supported_mods:
            print(f"  SKIP  {case_id}  [{modality}] — not supported by {tool.name}")
            continue

        if modality_filter and modality != modality_filter:
            continue

        case_path = os.path.join(DUMMY_ROOT, case_id)
        image_path = os.path.join(case_path, "image_nifti.nii.gz")

        if not os.path.exists(image_path):
            print(f"  ERROR {case_id}  image_nifti.nii.gz not found")
            continue

        inp = ToolInput(
            image_path=image_path,
            case_path=case_path,
            modality=modality,
            anatomy=anatomy_from_modality(modality),
            target_organs=DEFAULT_ORGANS,
            device=device,
        )

        print(f"  Running {case_id} [{modality}] ...", end="", flush=True)
        t0 = time.time()
        output = tool._timed_run(inp)
        elapsed = time.time() - t0

        status = "OK  " if output.success else "FAIL"
        organs_n = len(output.organs_segmented)
        err_str = f"  ERR: {output.error[:80]}" if not output.success else ""
        print(f"  {status}  {elapsed:.1f}s  {organs_n} organs{err_str}")

        results.append({
            "case": case_id,
            "modality": modality,
            "success": output.success,
            "n_organs": organs_n,
            "runtime_s": round(elapsed, 1),
            "error": output.error,
        })

    # Summary
    print(f"\n--- Summary ({tool.name}) ---")
    ok = sum(1 for r in results if r["success"])
    print(f"  {ok}/{len(results)} cases succeeded")
    for r in results:
        tag = "✓" if r["success"] else "✗"
        print(f"  {tag} {r['case']:8s} [{r['modality']}]  {r['n_organs']} organs  {r['runtime_s']}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run tool wrappers on dummy CT/MRI data")
    parser.add_argument(
        "--tool", default="totalseg_ct",
        choices=ALL_TOOLS + ["all"],
        help="Which tool to run (default: totalseg_ct)",
    )
    parser.add_argument(
        "--cases", nargs="+", default=None,
        help="Specific case IDs to process (default: all in dummy_outputs/)",
    )
    parser.add_argument(
        "--modality", default=None, choices=["CT", "MRI"],
        help="Filter by modality (default: run both)",
    )
    parser.add_argument(
        "--device", default="gpu:0",
        help="GPU device string (default: gpu:0)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Dry run — simulate outputs without calling the tool",
    )
    args = parser.parse_args()

    cases = args.cases or get_all_cases()
    tools_to_run = ALL_TOOLS if args.tool == "all" else [args.tool]

    for tool_name in tools_to_run:
        run_cases(
            tool_name=tool_name,
            cases=cases,
            device=args.device,
            dry_run=args.dry_run,
            modality_filter=args.modality,
        )


if __name__ == "__main__":
    main()
