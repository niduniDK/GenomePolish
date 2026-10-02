"""Render an HTML dashboard from the corrected FASTA produced by correct.py."""

from __future__ import annotations

import argparse
import json
import shutil
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape



def read_fasta(path: Path) -> list[dict[str, str | int]]:
	records: list[dict[str, str | int]] = []
	name: str | None = None
	sequence: list[str] = []
	for line in path.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if not line:
			continue
		if line.startswith(">"):
			if name is not None:
				records.append({"name": name, "sequence": "".join(sequence), "length": sum(map(len, sequence))})
			name, sequence = line[1:].split(None, 1)[0], []
		elif name is not None:
			sequence.append(line)
	if name is not None:
		records.append({"name": name, "sequence": "".join(sequence), "length": sum(map(len, sequence))})
	return records


def render_dashboard(
	fasta: Path,
	output: Path,
	template: Path | None = None,
	raw_fasta: Path | None = None,
	prediction_dataframe=None,
	model=None,
	feature_columns=None,
	alignment_window_bases: int = 50,
) -> None:
	if alignment_window_bases < 1:
		raise ValueError("alignment_window_bases must be at least 1")
	records = read_fasta(fasta)
	raw_records = read_fasta(raw_fasta) if raw_fasta is not None else []
	raw_by_name = {record["name"]: record for record in raw_records}
	row = prediction_dataframe.iloc[0] if prediction_dataframe is not None and len(prediction_dataframe) else {}
	contig = str(row.get("contig", records[0]["name"] if records else "unknown"))
	position = int(row.get("position", row.get("assembly_position", 1)))
	p_error = float(row.get("error_proba", row.get("p_error", 0.5)))
	assembly_base = str(row.get("assembly_base", "N"))
	corrected_base = str(row.get("corrected_base", row.get("predicted_base", assembly_base)))
	window_rows = []
	feature_baseline = []
	feature_names = list(feature_columns or [])
	if prediction_dataframe is not None and feature_names:
		feature_values = prediction_dataframe.loc[:, feature_names].astype(float)
		feature_baseline = [float(value) for value in feature_values.mean(axis=0)]
		for _, prediction in prediction_dataframe.iterrows():
			window_rows.append({
				"position": int(prediction.get("position", prediction.get("assembly_position", 1))),
				"features": [float(prediction[name]) for name in feature_names],
				"p_error": float(prediction.get("error_proba", prediction.get("p_error", 0.5))),
			})
	shap_features = [
		{"feature": name, "value": 0.0}
		for name in feature_names
	]
	genome_alignment = []
	for corrected_record in records:
		contig_name = str(corrected_record["name"])
		corrected_sequence = str(corrected_record["sequence"])
		raw_record = raw_by_name.get(contig_name, {})
		raw_sequence = str(raw_record.get("sequence", ""))
		length = max(len(raw_sequence), len(corrected_sequence))
		genome_alignment.append({
			"contig": contig_name,
			"length": length,
		})
	context_row = {
		"contig": contig,
		"position": position,
		"range_start": max(1, position - 10),
		"range_end": position + 10,
		"zoom_bp": 10,
		"assembly_base": assembly_base,
		"corrected_base": corrected_base,
		"p_error": p_error,
		"decision": str(row.get("predicted_label", "Review")),
		"threshold": 0.5,
		"status": "Prediction available",
		"status_level": "warn",
	}
	context = {
		"site": context_row,
		"sequences": records,
		"genome_alignment": genome_alignment,
		"alignment_window_bases": alignment_window_bases,
		"pileup": {"n_reads": 0, "n_g": 0, "n_a": 0, "depth": 0, "reads": [], "more_reads": 0, "more_breakdown": "", "strand_bias": 0, "mean_mq": 0, "af": "0%"},
		"shap": {"base_value": "0.000", "shap_sum": f"{sum(item['value'] for item in shap_features):+.3f}", "final_value": f"{p_error:.3f}", "features": shap_features},
		"rationale": {"summary": "Prediction data loaded for this assembly site.", "points": []},
		"window_rows": window_rows,
		"feature_names": feature_names,
		"feature_baseline": feature_baseline,
	}
	template_path = template or Path(__file__).with_name("templates") / "dashboard.html.j2"
	environment = Environment(
		loader=FileSystemLoader(str(template_path.parent)),
		autoescape=select_autoescape(),
	)
	html = environment.get_template(template_path.name).render(**context)
	output.parent.mkdir(parents=True, exist_ok=True)
	output.write_text(html, encoding="utf-8")
	manifest = []
	for contig_index, contig_alignment in enumerate(genome_alignment):
		contig_name = contig_alignment["contig"]
		corrected_sequence = str(records[contig_index]["sequence"])
		raw_sequence = str(raw_by_name.get(contig_name, {}).get("sequence", ""))
		chunk_files = []
		for chunk_start in range(0, contig_alignment["length"], alignment_window_bases):
			chunk_name = f"alignment-{contig_index}-{chunk_start}.json"
			chunk_path = output.parent / chunk_name
			chunk_end = chunk_start + alignment_window_bases
			chunk_path.write_text(json.dumps({
				"start": chunk_start,
				"raw": raw_sequence[chunk_start:chunk_end],
				"corrected": corrected_sequence[chunk_start:chunk_end],
			}), encoding="utf-8")
			chunk_files.append(chunk_name)
		manifest.append({
			"contig": contig_name,
			"length": contig_alignment["length"],
			"chunks": chunk_files,
		})
	(output.parent / "alignment-manifest.json").write_text(
		json.dumps({"windowBases": alignment_window_bases, "contigs": manifest}),
		encoding="utf-8",
	)
	for asset_name in ("dashboard.css", "dashboard.js", "shap_wasm.js"):
		asset = template_path.parent / asset_name
		if asset.is_file():
			shutil.copy2(asset, output.parent / asset_name)


