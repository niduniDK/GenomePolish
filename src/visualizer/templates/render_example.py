# ! This is an example to view the visualization dashboard for a single site. It is not part of the main pipeline.

"""
Minimal example of how to feed data into dashboard_template.html.j2.
Adapt this to pull real values from your PolishML pipeline / SHAP explainer.

Usage:
    python render_example.py > out.html
    # then serve the folder (dashboard.css / dashboard.js must sit alongside out.html)
"""
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader("."))
template = env.get_template("dashboard.html.j2")

# --- raw vs polished alignment row ---------------------------------------
def cell(base, cls="plain"):
    return {"base": base, "cls": cls}

alignment = {
    "raw": [cell(b) for b in "AGCTGATCC"] + [cell("X", "error")] + [cell(b) for b in "GTTTTTTTTA"] + [cell("C"), cell("G"), cell("G"), cell("A"), cell("T"), cell("T")],
    "connection": ["|"] + ["."] * 8 + ["X"] + ["."] * 9 + ["|"],
    "polished": [cell(b) for b in "AGCTGATCC"] + [cell("G", "applied")] + [cell(b) for b in "TTTTTTTT"] + [cell("-", "hp")] + [cell(b) for b in "ACGGATT"],
}

# --- SHAP features (from your explainer.shap_values(x)) ------------------
shap_features = [
    {"feature": "Mismatch frac", "value": 0.342},
    {"feature": "HP length",     "value": 0.218},
    {"feature": "Strand bias",   "value": 0.156},
    {"feature": "Mean BQ (alt)", "value": 0.128},
    {"feature": "Coverage z",    "value": 0.089},
    {"feature": "Mean MAPQ",     "value": -0.072},
    {"feature": "Soft-clip rate","value": 0.064},
    {"feature": "Alt entropy",   "value": 0.058},
    {"feature": "Indel frac",    "value": -0.045},
    {"feature": "Local GC%",     "value": -0.022},
]

# --- pileup reads (46 G / 4 A, truncated for the visible rows) -----------
pileup_reads = [
    {"base": "G", "strand": "+", "bq": 38, "mq": 60, "range": [28, 39]},
    {"base": "G", "strand": "+", "bq": 34, "mq": 60, "range": [28, 39]},
    {"base": "A", "strand": "+", "bq": 18, "mq": 60, "range": [6, 13]},
    {"base": "G", "strand": "-", "bq": 37, "mq": 60, "range": [28, 39]},
    {"base": "G", "strand": "+", "bq": 22, "mq": 60, "range": [28, 39]},
    {"base": "A", "strand": "-", "bq": 15, "mq": 60, "range": [6, 13]},
    {"base": "G", "strand": "+", "bq": 39, "mq": 60, "range": [28, 39]},
    {"base": "G", "strand": "+", "bq": 24, "mq": 60, "range": [28, 39]},
]

context = {
    "site": {
        "contig": "1",
        "position": 14531,
        "range_start": 14525,
        "range_end": 14549,
        "zoom_bp": 10,
        "assembly_base": "A",
        "corrected_base": "G",
        "p_error": 0.963,
        "decision": "Flag",
        "threshold": 0.995,
        "status": "Substitution Detected",
        "status_level": "warn",
    },
    "alignment": alignment,
    "pileup": {
        "n_reads": 50,
        "n_g": 46,
        "n_a": 4,
        "depth": 50,
        "reads": pileup_reads,
        "more_reads": 26,
        "more_breakdown": "25x G, 1x A",
        "strand_bias": 0.04,
        "mean_mq": 59.1,
        "af": "92%",
    },
    "shap": {
        "base_value": "0.050",
        "shap_sum": "+0.942",
        "final_value": "0.963",
        "features": shap_features,
    },
    "rationale": {
        "summary": (
            "Flagged for review — 92% of reads (46/50) support G over the "
            "assembly's A. All G-supporting reads have high base quality "
            "(mean 34.8) and map uniquely (MQ 59.1)."
        ),
        "points": [
            "The 4 A-supporting reads have low BQ (mean 10.5).",
            "Calibrated confidence (0.963) falls below the auto-correct threshold (0.995).",
        ],
    },
}

print(template.render(**context))

import webbrowser
from pathlib import Path

out_path = Path(__file__).parent / "out.html"
out_path.write_text(template.render(**context))
# webbrowser.open(out_path.resolve().as_uri())
