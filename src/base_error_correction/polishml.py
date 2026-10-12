import argparse
import os
import webbrowser
from pathlib import Path

import pandas as pd
import onnx

from .utils.correction_utils import (
    correct_assemblies,
    write_fasta,
)
from .inference import ErrorDetectionModel, FEATURE_COLUMNS
from .aligner import Aligner, get_alignments
from visualizer.feature_contribution import onnx_to_tree_dict
from visualizer.visualize import render_dashboard, start_dashboard_server

# from jinja2 import Environment, FileSystemLoader


def main():
    parser = argparse.ArgumentParser(prog="base-correct", description=__doc__)
    parser.add_argument(
        "--illumina-bam",
        help="Illumina BAM aligned to the assembly",
    )
    parser.add_argument(
        "--ont-bam",
        help="ONT BAM aligned to the assembly",
    )
    parser.add_argument(
        "--assembly-fasta",
        required=True,
        help="original assembly FASTA being corrected",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="path for the directory containing all output files",
    )
    parser.add_argument("--illumina-r1", help="Illumina R1 FASTQ for alignment")
    parser.add_argument("--illumina-r2", help="Illumina R2 FASTQ for alignment")
    parser.add_argument("--ont-fastq", help="ONT FASTQ for alignment")
    parser.add_argument("--error-type-col", default="predicted_label")
    parser.add_argument("--proba-col", default="error_proba")
    parser.add_argument("--correction-threshold", type=float, default=0.5)
    parser.add_argument(
        "--alignment-window-bases",
        type=int,
        default=256,
        help="number of bases visible in the dashboard window",
    )
    parser.add_argument(
        "--chunk-bases",
        type=int,
        default=256,
        help="number of bases written per cached JSON chunk; must be >= --alignment-window-bases",
    )
    parser.add_argument(
        "--aligner",
        action="store_true",
        help="align FASTQ reads to the draft assembly",
    )
    args = parser.parse_args()
    output_directory = Path(args.output_dir)
    output_fasta = output_directory / "corrected.fasta"
    dashboard_path = output_directory / "dashboard.html"

    bam_paths = {}

    if not args.illumina_bam and not args.ont_bam and not args.aligner:
        parser.error("at least one of --illumina-bam / --ont-bam is required")

    if not args.illumina_bam and not args.ont_bam and args.aligner:
        if all(
            value is not None
            for value in (
                args.assembly_fasta,
                args.illumina_r1,
                args.illumina_r2,
                args.ont_fastq,
            )
        ):
            illumina_bam, ont_bam = get_alignments(
                args.assembly_fasta,
                args.illumina_r1,
                args.illumina_r2,
                args.ont_fastq,
                output_directory,
            )
            bam_paths["illumina"] = illumina_bam
            bam_paths["ont"] = ont_bam
        else:
            parser.error(
                "Provide assembly FASTA, Illumina R1/R2 and ONT FASTQ files for alignment"
            )

    if args.illumina_bam:
        bam_paths["illumina"] = args.illumina_bam
    if args.ont_bam:
        bam_paths["ont"] = args.ont_bam

    if "illumina" not in bam_paths or "ont" not in bam_paths:
        parser.error(
            "--predictions-csv or both --illumina-bam and --ont-bam are required"
        )
    Aligner().ensure_bwa_index(args.assembly_fasta)
    model_path = os.path.join(
        os.path.dirname(__file__), "models", "lightgbm_model_base_error.onnx"
    )
    base_model = ErrorDetectionModel(model_path)
    predictions = base_model.predict(
        assembly_fasta=args.assembly_fasta,
        illumina_bam=bam_paths["illumina"],
        ont_bam=bam_paths["ont"],
    )

    corrected = correct_assemblies(
        predictions,
        bam_paths,
        args.assembly_fasta,
        error_type_col=args.error_type_col,
        proba_col=args.proba_col,
        correction_threshold=args.correction_threshold,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    write_fasta(output_fasta, corrected)
    print(f"wrote {output_fasta}")
    render_dashboard(
        output_fasta,
        dashboard_path,
        raw_fasta=Path(args.assembly_fasta),
        prediction_dataframe=predictions,
        model=base_model,
        feature_columns=FEATURE_COLUMNS,
        alignment_window_bases=args.alignment_window_bases,
        chunk_bases=args.chunk_bases,
        bam_paths=bam_paths,
    )
    print(f"dashboard written to {dashboard_path}")
    server, dashboard_url = start_dashboard_server(
        dashboard_path,
        bam_paths,
        prediction_dataframe=predictions,
        feature_names=FEATURE_COLUMNS,
        shap_model=onnx_to_tree_dict(
            model_path,
            predictions.loc[:, list(FEATURE_COLUMNS)],
        ),
    )
    print(f"dashboard available at {dashboard_url} (press Ctrl+click to open in browser)")
    webbrowser.open(dashboard_url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    # env = Environment(loader=FileSystemLoader("/home/cse_g2/Niduni/genome-polish-tool/src/visualizer/templates"))
    # template = env.get_template("dashboard.html.j2")
    # out_path = Path(__file__).parent / "out.html"
    # out_path.write_text(template.render(**context))


if __name__ == "__main__":
    main()
