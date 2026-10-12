"""Render an HTML dashboard from the corrected FASTA produced by correct.py."""

from __future__ import annotations

import argparse
from functools import partial
import json
import shutil
import subprocess
import threading
import webbrowser
from collections import Counter
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .feature_contribution import calculate_shap_features

from jinja2 import Environment, FileSystemLoader, select_autoescape
import pysam

MAX_DASHBOARD_ROWS = 5000


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args,
        bam_paths=None,
        prediction_dataframe=None,
        feature_names=None,
        shap_model=None,
        **kwargs,
    ):
        self.bam_paths = bam_paths or {}
        self.prediction_dataframe = prediction_dataframe
        self.feature_names = feature_names or []
        self.shap_model = shap_model
        super().__init__(*args, **kwargs)

    def do_GET(self):
        request = urlparse(self.path)
        if request.path not in ("/api/pileup", "/api/shap"):
            return super().do_GET()

        query = parse_qs(request.query)
        contig = query.get("contig", [""])[0]
        try:
            position = int(query.get("position", [""])[0])
        except (TypeError, ValueError):
            self.send_error(400, "A valid position is required")
            return
        if not contig or position < 1:
            self.send_error(400, "A contig and positive position are required")
            return
        payload = {}
        if request.path == "/api/pileup":
            payload = summarize_pileup(self.bam_paths, contig, position)
        elif request.path == "/api/shap":
            if self.prediction_dataframe is None or self.shap_model is None:
                self.send_error(503, "SHAP is not configured")
                return
            matches = self.prediction_dataframe[
                (self.prediction_dataframe["contig"].astype(str) == contig)
                & (self.prediction_dataframe["position"] == position)
            ]
            if matches.empty:
                self.send_error(404, "No prediction found for this position")
                return
            values = calculate_shap_features(
                self.shap_model,
                matches.iloc[[0]],
                self.feature_names,
            )
            payload = {
                "features": values,
                "shap_sum": sum(item["value"] for item in values),
                "p_error": float(matches.iloc[0].get("error_proba", 0.0)),
            }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def start_dashboard_server(
    path: Path,
    bam_paths=None,
    prediction_dataframe=None,
    feature_names=None,
    shap_model=None,
) -> tuple[ThreadingHTTPServer, str]:
    handler = partial(
        DashboardRequestHandler,
        directory=str(path.parent.resolve()),
        bam_paths=bam_paths,
        prediction_dataframe=prediction_dataframe,
        feature_names=feature_names,
        shap_model=shap_model,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/{path.name}"


def find_bam_index(bam: Path) -> Path | None:
    for candidate in (Path(f"{bam}.bai"), bam.with_suffix(".bai")):
        if candidate.is_file():
            return candidate
    return None


def ensure_bam_index(bam: Path) -> Path:
    index = find_bam_index(bam)
    if index is not None:
        return index
    if not bam.is_file():
        raise FileNotFoundError(f"BAM file for pileup was not found: {bam}")
    samtools = shutil.which("samtools")
    if samtools is None:
        raise FileNotFoundError(
            f"BAM index was not found for {bam}; install samtools or provide {bam}.bai"
        )
    subprocess.run([samtools, "index", str(bam)], check=True)
    index = find_bam_index(bam)
    if index is None:
        raise FileNotFoundError(f"samtools did not create a BAM index for {bam}")
    return index


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


# def summarize_pileup(bam_paths, contig: str, position: int) -> dict:
#     support = Counter()
#     platform_depth = {}
#     forward = reverse = total = 0
#     mapping_quality = []
#     for platform, bam_value in (bam_paths or {}).items():
#         try:
#             with pysam.AlignmentFile(bam_value, "rb") as bam:
#                 platform_count = 0
#                 for column in bam.pileup(
#                     contig,
#                     position - 1,
#                     position,
#                     truncate=True,
#                     min_base_quality=0,
#                     min_mapping_quality=0,
#                 ):
#                     if column.reference_pos != position - 1:
#                         continue
#                     for pileup_read in column.pileups:
#                         if (
#                             pileup_read.is_del
#                             or pileup_read.is_refskip
#                             or pileup_read.query_position is None
#                         ):
#                             continue
#                         read = pileup_read.alignment
#                         base = read.query_sequence[pileup_read.query_position].upper()
#                         support[base] += 1
#                         platform_count += 1
#                         total += 1
#                         mapping_quality.append(read.mapping_quality)
#                         if read.is_reverse:
#                             reverse += 1
#                         else:
#                             forward += 1
#                 platform_depth[platform] = platform_count
#         except (OSError, ValueError, KeyError):
#             continue
#     dominant_base, dominant_count = support.most_common(1)[0] if support else ("N", 0)
#     return {
#         "depth": total,
#         "support": dict(support),
#         "dominant_base": dominant_base,
#         "dominant_count": dominant_count,
#         "platform_depth": platform_depth,
#         "mean_mq": sum(mapping_quality) / len(mapping_quality)
#         if mapping_quality
#         else 0.0,
#         "strand_bias": abs(forward - reverse) / total if total else 0.0,
#         "af": f"{(dominant_count / total * 100):.1f}%" if total else "0%",
#     }


def fetch_site_reads(bam, platform: str, contig: str, position: int) -> list[dict]:
    reads = []
    for column in bam.pileup(
        contig,
        position - 1,
        position,
        truncate=True,
        min_base_quality=0,
        min_mapping_quality=0,
    ):
        if column.reference_pos != position - 1:
            continue
        for p in column.pileups:
            if p.is_del or p.is_refskip or p.query_position is None:
                continue
            a = p.alignment
            reads.append(
                {
                    "platform": platform,
                    "base": a.query_sequence[p.query_position].upper(),
                    "strand": "-" if a.is_reverse else "+",
                    "bq": a.query_qualities[p.query_position],
                    "mq": a.mapping_quality,
                    "offset": p.query_position,
                }
            )
    return reads


def summarize_pileup(bam_paths, contig: str, position: int) -> dict:
    reads = []
    for platform, bam_value in (bam_paths or {}).items():
        try:
            with pysam.AlignmentFile(bam_value, "rb") as bam:
                reads.extend(fetch_site_reads(bam, platform, contig, position))
        except (OSError, ValueError, KeyError):
            continue
    support = Counter(r["base"] for r in reads)
    total = len(reads)
    reverse = sum(r["strand"] == "-" for r in reads)
    forward = total - reverse
    mapping_quality = [r["mq"] for r in reads]
    dominant_base, dominant_count = support.most_common(1)[0] if support else ("N", 0)
    # print({
    #     "reads": reads,
    #     "depth": total,
    #     "support": dict(support),
    #     "dominant_base": dominant_base,
    #     "dominant_count": dominant_count,
    #     "platform_depth": dict(Counter(r["platform"] for r in reads)),
    #     "mean_mq": sum(mapping_quality) / total if total else 0.0,
    #     "strand_bias": abs(forward - reverse) / total if total else 0.0,
    #     "af": f"{(dominant_count / total * 100):.1f}%" if total else "0%",
    # })
    return {
        "reads": reads,
        "depth": total,
        "support": dict(support),
        "dominant_base": dominant_base,
        "dominant_count": dominant_count,
        "platform_depth": dict(Counter(r["platform"] for r in reads)),
        "mean_mq": sum(mapping_quality) / total if total else 0.0,
        "strand_bias": abs(forward - reverse) / total if total else 0.0,
        "af": f"{(dominant_count / total * 100):.1f}%" if total else "0%",
    }


def get_rationale(
    pileup, p_error: float, decision: str, assembly_base: str, corrected_base: str
) -> dict[str, str]:
    # pileup = summarize_pileup(bam_paths, contig, position)
    print("pileup: ", pileup)
    if pileup["depth"]:
        support = ", ".join(
            f"{base}: {count} ({count / pileup['depth'] * 100:.1f}%)"
            for base, count in sorted(
                pileup["support"].items(), key=lambda item: (-item[1], item[0])
            )
        )
        pileup_argument = (
            f"Pileup support: depth {pileup['depth']}; {support}; "
            f"mean MAPQ {pileup['mean_mq']:.1f}; strand bias {pileup['strand_bias']:.3f}."
        )
    else:
        pileup_argument = "Pileup support: no usable reads were found at this position."

    return {
        "summary": (
            f"The model predicts {decision} with error confidence {p_error:.3f}. "
            f"The draft base is {assembly_base} and the corrected base is {corrected_base}."
        ),
        "pileup_argument": pileup_argument,
    }

def render_dashboard(
    fasta: Path,
    output: Path,
    template: Path | None = None,
    raw_fasta: Path | None = None,
    prediction_dataframe=None,
    model=None,
    feature_columns=None,
    alignment_window_bases: int = 50,
    chunk_bases: int | None = None,
    bam_paths: dict[str, str | Path] | None = None,
) -> None:

    if alignment_window_bases < 1:
        raise ValueError("alignment_window_bases must be at least 1")
    chunk_bases = (
        chunk_bases if chunk_bases is not None else max(4096, alignment_window_bases)
    )
    if chunk_bases < alignment_window_bases:
        raise ValueError("chunk_bases must be at least alignment_window_bases")

    records = read_fasta(fasta)
    raw_records = read_fasta(raw_fasta) if raw_fasta is not None else []
    raw_by_name = {record["name"]: record for record in raw_records}
    row = (
        prediction_dataframe.iloc[0]
        if prediction_dataframe is not None and len(prediction_dataframe)
        else {}
    )
    contig = str(row.get("contig", records[0]["name"] if records else "unknown"))
    position = int(row.get("position", row.get("assembly_position", 1)))
    p_error = float(row.get("error_proba", row.get("p_error", 0.5)))
    assembly_base = str(row.get("assembly_base", "N"))
    corrected_base = str(
        row.get("corrected_base", row.get("predicted_base", assembly_base))
    )
    window_rows = []
    feature_baseline = []
    feature_names = list(feature_columns or [])

    print(
        f"rendering dashboard for {len(records)} contigs, {len(raw_records)} raw contigs, {len(prediction_dataframe) if prediction_dataframe is not None else 0} predictions, {len(feature_names)} features"
    )

    prediction_by_contig = {}
    if prediction_dataframe is not None:
        if feature_names:
            for contig_name, contig_predictions in prediction_dataframe.groupby(
                "contig", sort=False
            ):
                prediction_by_contig[str(contig_name)] = contig_predictions.sort_values(
                    "position"
                ).set_index("position")
            feature_baseline = [
                float(value)
                for value in prediction_dataframe.loc[:, feature_names].mean(axis=0)
            ]
        row_step = max(
            1,
            (len(prediction_dataframe) + MAX_DASHBOARD_ROWS - 1) // MAX_DASHBOARD_ROWS,
        )
        dashboard_predictions = prediction_dataframe.iloc[::row_step]
        for _, prediction in dashboard_predictions.iterrows():
            window_rows.append(
                {
                    "position": int(
                        prediction.get(
                            "position", prediction.get("assembly_position", 1)
                        )
                    ),
                    "features": [float(prediction[name]) for name in feature_names],
                    "p_error": float(
                        prediction.get("error_proba", prediction.get("p_error", 0.5))
                    ),
                }
            )

    shap_features = [{"feature": name, "value": 0.0} for name in feature_names]

    print(
        f"rendering dashboard with {len(window_rows)} window rows, {len(shap_features)} shap features, {len(feature_baseline)} feature baseline values"
    )

    genome_alignment = []
    for corrected_record in records:
        contig_name = str(corrected_record["name"])
        corrected_sequence = str(corrected_record["sequence"])
        raw_record = raw_by_name.get(contig_name, {})
        raw_sequence = str(raw_record.get("sequence", ""))
        length = max(len(raw_sequence), len(corrected_sequence))
        genome_alignment.append(
            {
                "contig": contig_name,
                "length": length,
            }
        )

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

    output.parent.mkdir(parents=True, exist_ok=True)
    for bam_value in (bam_paths or {}).values():
        ensure_bam_index(Path(bam_value))  # pileup() needs the .bai; no symlinks needed now

    pileup = summarize_pileup(bam_paths, contig, position)
    rationale = get_rationale(
        pileup,
        p_error,
        context_row["decision"],
        assembly_base,
        corrected_base,
    )
    # pileup_context = {
    #     "n_reads": pileup["depth"],
    #     "n_g": pileup["support"].get("G", 0),
    #     "n_a": pileup["support"].get("A", 0),
    #     "n_t": pileup["support"].get("T", 0),
    #     "n_c": pileup["support"].get("C", 0),
    #     "depth": pileup["depth"],
    #     "reads": [],
    #     "more_reads": 0,
    #     "more_breakdown": "",
    #     "strand_bias": pileup["strand_bias"],
    #     "mean_mq": pileup["mean_mq"],
    #     "af": pileup["af"],
    # }
    # print("Pileup context:", pileup_context)
    
    context = {
        "site": context_row,
        "sequences": records,
        "genome_alignment": genome_alignment,
        "alignment_window_bases": alignment_window_bases,
        "pileup": {
			"n_reads": pileup["depth"],
			"n_g": pileup["support"].get("G", 0),
			"n_a": pileup["support"].get("A", 0),
			"n_t": pileup["support"].get("T", 0),
			"n_c": pileup["support"].get("C", 0),
			"depth": pileup["depth"],
			"reads": [],
			"more_reads": 0,
			"more_breakdown": "",
			"strand_bias": pileup["strand_bias"],
			"mean_mq": pileup["mean_mq"],
			"af": pileup["af"],
		},
        # SHAP is calculated on demand by /api/shap. The model passed here is
        # an ONNX Runtime wrapper, which is not a TreeExplainer-compatible model.
        "shap": {
            "base_value": "0.000",
            "shap_sum": "+0.000",
            "final_value": f"{p_error:.3f}",
            "features": shap_features,
        },
        "rationale": rationale,
        "window_rows": window_rows,
        "feature_names": feature_names,
        "feature_baseline": feature_baseline,
    }

    print(
        f"rendering dashboard to {output} with {len(context['window_rows'])} window rows, {len(context['feature_names'])} feature names, {len(context['feature_baseline'])} feature baseline values"
    )

    template_path = (
        template or Path(__file__).with_name("templates") / "dashboard.html.j2"
    )
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
        contig_predictions = prediction_by_contig.get(contig_name)
        for chunk_start in range(0, contig_alignment["length"], chunk_bases):
            chunk_name = f"alignment-{contig_index}-{chunk_start}.json"
            chunk_path = output.parent / chunk_name
            chunk_end = min(chunk_start + chunk_bases, contig_alignment["length"])
            chunk_metadata = []
            if contig_predictions is not None:
                for position in range(chunk_start + 1, chunk_end + 1):
                    prediction = (
                        contig_predictions.loc[position]
                        if position in contig_predictions.index
                        else None
                    )
                    if prediction is None:
                        chunk_metadata.append(None)
                        continue
                    chunk_metadata.append(
                        {
                            "position": position,
                            "assembly_base": str(
                                prediction.get(
                                    "assembly_base",
                                    raw_sequence[position - 1 : position] or "N",
                                )
                            ),
                            "corrected_base": corrected_sequence[
                                position - 1 : position
                            ]
                            or str(prediction.get("assembly_base", "N")),
                            "p_error": float(prediction.get("error_proba", 0.5)),
                            "decision": str(
                                prediction.get("predicted_label", "Review")
                            ),
                            "illumina_coverage": float(
                                prediction.get("illumina_coverage", 0.0)
                            ),
                            "ont_coverage": float(prediction.get("ont_coverage", 0.0)),
                            "features": [
                                float(prediction[name]) for name in feature_names
                            ],
                        }
                    )
            else:
                chunk_metadata = [None] * (chunk_end - chunk_start)
            chunk_path.write_text(
                json.dumps(
                    {
                        "start": chunk_start,
                        "raw": raw_sequence[chunk_start:chunk_end],
                        "corrected": corrected_sequence[chunk_start:chunk_end],
                        "metadata": chunk_metadata,
                    }
                ),
                encoding="utf-8",
            )
            chunk_files.append(chunk_name)
        manifest.append(
            {
                "contig": contig_name,
                "length": contig_alignment["length"],
                "chunks": chunk_files,
            }
        )
        print(
            f"wrote {len(manifest)} records in alignment manifest for {len(genome_alignment)} contigs"
        )
    (output.parent / "alignment-manifest.json").write_text(
        json.dumps(
            {
                "windowBases": alignment_window_bases,
                "chunkBases": chunk_bases,
                "contigs": manifest,
            }
        ),
        encoding="utf-8",
    )
    for asset_name in ("dashboard.css", "dashboard.js", "shap_wasm.js"):
        asset = template_path.parent / asset_name
        if asset.is_file():
            shutil.copy2(asset, output.parent / asset_name)
