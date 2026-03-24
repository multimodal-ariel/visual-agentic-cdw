"""
Batch QC report generator.

Iterates cases in a directory, generates PNG figures, and produces a single
index.html sorted by QC severity (FAIL first) — designed for the reviewer
workflow: open worst cases first, spot-check WARNs, skip PASSes.

CLI
---
    # Figures only (no QC JSON — all cases treated as UNKNOWN severity)
    conda run -n cdw_radiomics python -m visualizations.report \\
        --cases dummy_outputs/ --output viz_output/

    # Severity-sorted with QC data from pipeline_results.json
    conda run -n cdw_radiomics python -m visualizations.report \\
        --cases dummy_outputs/ --output viz_output/ \\
        --qc-json logs/pipeline_results.json

    # Disagreement overlays for multi-tool cases
    conda run -n cdw_radiomics python -m visualizations.report \\
        --cases dummy_outputs/ --output viz_output/ \\
        --qc-json logs/pipeline_results.json --disagreements
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from visualizations.utils import (
    SEVERITY_HEX, SEVERITY_RANK, TOOL_DIR_MAP,
    list_masks, list_tool_dirs,
)
from visualizations.slice_viewer import plot_case_overview, plot_disagreement_overlay

logger = logging.getLogger(__name__)


# ── QC JSON parsing ───────────────────────────────────────────────────────────

def _load_qc_json(qc_json_path: str | Path) -> dict[str, dict]:
    """
    Parse a pipeline results JSON into {case_id: qc_info}.

    Handles two formats:
      1. pipeline_results.json from orchestrator (list of case dicts)
      2. e2e_report.json from tests/run_e2e.py (image_pipeline list)

    Returns:
        {case_id: {
            "overall_severity": "PASS"/"WARN"/"FAIL"/"UNKNOWN",
            "organ_severities": {organ: severity},
            "flagged_organs":   [organ, ...],
            "anatomy":          str,
            "modality":         str,
            "tools":            [tool_dir, ...],
        }}
    """
    path = Path(qc_json_path)
    raw = json.loads(path.read_text())

    result: dict[str, dict] = {}

    # Detect format
    cases_list: list[dict] = []
    if isinstance(raw, list):
        cases_list = raw
    elif "image_pipeline" in raw:
        cases_list = raw["image_pipeline"]
    elif "cases" in raw:
        cases_list = raw["cases"]

    for case in cases_list:
        case_id = case.get("case_id") or case.get("id", "")
        if not case_id:
            continue

        # Overall severity
        sev = (
            case.get("qc_overall_severity")
            or case.get("overall_severity")
            or "UNKNOWN"
        ).upper()

        # Per-organ severities from qc_per_tool
        organ_severities: dict[str, str] = {}
        flagged: list[str] = []
        tools_used: list[str] = []

        qc_per_tool = case.get("qc_per_tool", {})
        for tool_dir, tool_qc in qc_per_tool.items():
            if tool_dir not in TOOL_DIR_MAP:
                # Try matching by display name
                tool_dir = next(
                    (k for k, v in TOOL_DIR_MAP.items() if v == tool_dir), tool_dir
                )
            tools_used.append(tool_dir)

            for organ_result in tool_qc.get("organ_results", []):
                name = organ_result.get("organ", "")
                s = (organ_result.get("severity") or "UNKNOWN").upper()
                if name:
                    # Keep worst severity across tools
                    prev = organ_severities.get(name, "PASS")
                    organ_severities[name] = (
                        s if SEVERITY_RANK.get(s, 3) < SEVERITY_RANK.get(prev, 3) else prev
                    )
                    if s in ("FAIL", "WARN") and name not in flagged:
                        flagged.append(name)

        result[case_id] = {
            "overall_severity": sev,
            "organ_severities": organ_severities,
            "flagged_organs":   flagged,
            "anatomy":          case.get("anatomy") or case.get("metadata", {}).get("anatomy", ""),
            "modality":         case.get("modality") or case.get("metadata", {}).get("modality", ""),
            "tools":            tools_used or list(qc_per_tool.keys()),
        }

    return result


# ── Figure generation ─────────────────────────────────────────────────────────

def generate_case_figures(
    case_path: Path,
    output_dir: Path,
    qc_info: Optional[dict] = None,
    with_disagreements: bool = False,
) -> list[dict]:
    """
    Generate overview PNGs for every tool in a case.

    Returns list of figure metadata dicts for the HTML report.
    """
    figures = []
    tool_dirs = list_tool_dirs(case_path)

    if not tool_dirs:
        return figures

    organ_sevs = qc_info.get("organ_severities", {}) if qc_info else {}
    overall_sev = qc_info.get("overall_severity", "UNKNOWN") if qc_info else "UNKNOWN"

    for display, seg_dir in tool_dirs.items():
        tool_dir_name = seg_dir.name
        png_name = f"{case_path.name}_{display}_overview.png"
        png_path = output_dir / png_name

        saved = plot_case_overview(
            case_path=case_path,
            tool_dir=tool_dir_name,
            output_path=png_path,
            qc_severities=organ_sevs if organ_sevs else None,
        )

        if saved:
            mask_count = sum(1 for _ in seg_dir.glob("*.nii.gz"))
            figures.append({
                "case_id":   case_path.name,
                "tool":      display,
                "severity":  overall_sev,
                "flagged":   qc_info.get("flagged_organs", []) if qc_info else [],
                "anatomy":   qc_info.get("anatomy", "") if qc_info else "",
                "modality":  qc_info.get("modality", "") if qc_info else "",
                "masks":     mask_count,
                "png":       png_name,
            })

    # Disagreement overlays for tools with shared organs
    if with_disagreements and len(tool_dirs) >= 2:
        tool_list = list(tool_dirs.items())
        for i in range(len(tool_list)):
            for j in range(i + 1, len(tool_list)):
                disp_a, dir_a = tool_list[i]
                disp_b, dir_b = tool_list[j]
                masks_a = list_masks(case_path, dir_a.name)
                masks_b = list_masks(case_path, dir_b.name)
                shared = sorted(set(masks_a) & set(masks_b))

                # Only produce disagreement figures for flagged organs
                flagged = set(qc_info.get("flagged_organs", [])) if qc_info else set(shared[:5])
                target_organs = [o for o in shared if o in flagged] or shared[:3]

                for organ in target_organs:
                    png_name = f"{case_path.name}_{organ}_{disp_a}_vs_{disp_b}.png"
                    png_path = output_dir / png_name
                    saved = plot_disagreement_overlay(
                        case_path=case_path,
                        organ=organ,
                        tool_a_dir=dir_a.name,
                        tool_b_dir=dir_b.name,
                        output_path=png_path,
                    )
                    if saved:
                        figures.append({
                            "case_id":   case_path.name,
                            "tool":      f"{disp_a} vs {disp_b}",
                            "severity":  organ_sevs.get(organ, "UNKNOWN"),
                            "flagged":   [organ],
                            "anatomy":   qc_info.get("anatomy", "") if qc_info else "",
                            "modality":  qc_info.get("modality", "") if qc_info else "",
                            "masks":     1,
                            "png":       png_name,
                            "is_disagreement": True,
                        })

    return figures


# ── HTML report ───────────────────────────────────────────────────────────────

def _sev_badge(sev: str) -> str:
    color = SEVERITY_HEX.get(sev, "#888888")
    return f'<span class="badge" style="background:{color}">{sev}</span>'


def build_html_report(
    figures: list[dict],
    output_path: Path,
    title: str = "CDW Segmentation QC Report",
) -> Path:
    """
    Write index.html with figures sorted by QC severity (FAIL → WARN → PASS → UNKNOWN).

    Layout: summary stats header + sortable table with thumbnails.
    Click any thumbnail to view full-size in a new tab.
    """
    figures_sorted = sorted(
        figures,
        key=lambda f: (
            SEVERITY_RANK.get(f["severity"], 3),
            f["case_id"],
            f["tool"],
        ),
    )

    counts = {"FAIL": 0, "WARN": 0, "PASS": 0, "UNKNOWN": 0}
    for f in figures_sorted:
        counts[f.get("severity", "UNKNOWN")] = counts.get(f.get("severity", "UNKNOWN"), 0) + 1

    rows = []
    for f in figures_sorted:
        sev = f.get("severity", "UNKNOWN")
        sev_color = SEVERITY_HEX.get(sev, "#888888")
        flagged_str = ", ".join(f.get("flagged", [])[:8])
        if len(f.get("flagged", [])) > 8:
            flagged_str += f" (+{len(f['flagged']) - 8} more)"
        badge = _sev_badge(sev)
        is_dis = f.get("is_disagreement", False)
        row_style = f'style="border-left: 4px solid {sev_color};"'
        rows.append(f"""
        <tr {row_style}>
          <td><code>{f['case_id']}</code></td>
          <td>{f.get('modality','')}</td>
          <td>{f.get('anatomy','')}</td>
          <td>{f['tool']}{'&nbsp;<em>(diff)</em>' if is_dis else ''}</td>
          <td>{badge}</td>
          <td class="flagged">{flagged_str}</td>
          <td>{f.get('masks', '')}</td>
          <td>
            <a href="{f['png']}" target="_blank">
              <img src="{f['png']}" loading="lazy" />
            </a>
          </td>
        </tr>""")

    rows_html = "\n".join(rows)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ font-family: sans-serif; background: #1a1a1a; color: #ddd; margin: 20px; }}
  h1   {{ color: #eee; font-size: 1.4em; }}
  .summary {{ display: flex; gap: 20px; margin: 12px 0 20px 0; }}
  .stat {{ padding: 8px 16px; border-radius: 4px; font-weight: bold; font-size: 1.1em; }}
  .stat-FAIL    {{ background: #5c1010; color: #ff8888; }}
  .stat-WARN    {{ background: #5c3300; color: #ffbb44; }}
  .stat-PASS    {{ background: #0a3d10; color: #66ee88; }}
  .stat-UNKNOWN {{ background: #333;    color: #aaa; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.85em; }}
  th {{ background: #2a2a2a; color: #bbb; padding: 8px; text-align: left;
        position: sticky; top: 0; z-index: 10; border-bottom: 2px solid #444; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid #2a2a2a; vertical-align: middle; }}
  tr:hover {{ background: #252525; }}
  .badge {{ padding: 2px 8px; border-radius: 3px; font-size: 0.8em;
             font-weight: bold; color: #111; }}
  .flagged {{ font-size: 0.78em; color: #bbb; max-width: 260px; }}
  img {{ width: 220px; border-radius: 3px; border: 1px solid #333;
         transition: transform 0.1s; cursor: pointer; }}
  img:hover {{ transform: scale(1.04); }}
  code {{ background: #2a2a2a; padding: 1px 4px; border-radius: 2px; font-size: 0.9em; }}
  em {{ color: #888; }}
</style>
</head>
<body>
<h1>&#128302; {title}</h1>
<p style="color:#777;font-size:0.85em">Generated {ts} &mdash; {len(figures_sorted)} figure(s) across {len(set(f['case_id'] for f in figures_sorted))} case(s). Sorted by severity (FAIL first).</p>

<div class="summary">
  <div class="stat stat-FAIL">&#10060; FAIL: {counts['FAIL']}</div>
  <div class="stat stat-WARN">&#9888; WARN: {counts['WARN']}</div>
  <div class="stat stat-PASS">&#10003; PASS: {counts['PASS']}</div>
  <div class="stat stat-UNKNOWN">&#63; UNKNOWN: {counts['UNKNOWN']}</div>
</div>

<table>
<thead>
  <tr>
    <th>Case ID</th><th>Modality</th><th>Anatomy</th><th>Tool</th>
    <th>Severity</th><th>Flagged Organs</th><th>Masks</th><th>Preview</th>
  </tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</body>
</html>
"""

    output_path.write_text(html)
    return output_path


# ── Batch entry point ─────────────────────────────────────────────────────────

def generate_batch_report(
    cases_dir: str | Path,
    output_dir: str | Path,
    qc_json: Optional[str | Path] = None,
    with_disagreements: bool = False,
) -> Path:
    """
    Generate PNGs for all cases and write index.html.

    Args:
        cases_dir:          Root directory containing per-case subdirs.
        output_dir:         Where to write PNGs and index.html.
        qc_json:            Optional pipeline results JSON for severity sorting.
        with_disagreements: Also generate tool-comparison diff figures.

    Returns:
        Path to index.html.
    """
    cases_dir = Path(cases_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qc_data: dict[str, dict] = {}
    if qc_json:
        logger.info("Loading QC data from %s", qc_json)
        qc_data = _load_qc_json(qc_json)
        logger.info("Loaded QC info for %d cases", len(qc_data))

    # Discover case directories (has at least one segmentations_* subdir)
    case_dirs = sorted([
        d for d in cases_dir.iterdir()
        if d.is_dir() and any(
            sd.is_dir() and sd.name.startswith("segmentations_")
            for sd in d.iterdir()
        )
    ])

    logger.info("Found %d cases in %s", len(case_dirs), cases_dir)

    all_figures: list[dict] = []
    for case_path in case_dirs:
        qc_info = qc_data.get(case_path.name)
        logger.info("Processing %s  (severity=%s)",
                    case_path.name, qc_info.get("overall_severity", "?") if qc_info else "?")
        figs = generate_case_figures(
            case_path=case_path,
            output_dir=output_dir,
            qc_info=qc_info,
            with_disagreements=with_disagreements,
        )
        all_figures.extend(figs)
        logger.info("  → %d figure(s)", len(figs))

    index_path = output_dir / "index.html"
    build_html_report(all_figures, index_path)
    logger.info("Report written: %s  (%d figures)", index_path, len(all_figures))
    return index_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate CDW segmentation QC visualization report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cases",   required=True, help="Directory containing case subdirs")
    parser.add_argument("--output",  required=True, help="Output directory for PNGs + index.html")
    parser.add_argument("--qc-json", default=None,
                        help="Pipeline results JSON for severity-sorted output")
    parser.add_argument("--disagreements", action="store_true",
                        help="Also generate tool-comparison disagreement figures")
    args = parser.parse_args()

    index = generate_batch_report(
        cases_dir=args.cases,
        output_dir=args.output,
        qc_json=args.qc_json,
        with_disagreements=args.disagreements,
    )
    print(f"\nReport ready: {index}")


if __name__ == "__main__":
    main()
