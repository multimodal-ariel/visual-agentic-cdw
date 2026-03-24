"""
CDW Segmentation Visualization
================================
Headless (Agg/matplotlib) slice-level visualization for QC review.

Primary entry point:
    python -m visualizations.report --cases dummy_outputs/ --output viz_output/

Or with a QC JSON for severity-sorted output:
    python -m visualizations.report --cases dummy_outputs/ --output viz_output/ \
        --qc-json logs/pipeline_results.json
"""
