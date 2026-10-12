from collections import Counter
import math

import numpy as np
import pandas as pd
import onnxruntime as ort
import pysam
from pathlib import Path
from tempfile import TemporaryDirectory

from .aligner import Aligner
from .utils.extract_illumina_features import extract_illumina_features
from .utils.extract_ont_features import extract_ont_features


FEATURE_COLUMNS = (
    "distance_to_contig_end",
    "illumina_coverage",
    "illumina_mismatch_rate",
    "illumina_insertion_rate",
    "illumina_deletion_rate",
    "illumina_mean_mapq",
    "illumina_strand_bias",
    "illumina_softclip_rate",
    "illumina_A_fraction",
    "illumina_C_fraction",
    "illumina_G_fraction",
    "illumina_T_fraction",
    "illumina_alt_mean_bq",
    "ont_coverage",
    "ont_mismatch_rate",
    "ont_insertion_rate",
    "ont_deletion_rate",
    "ont_mean_mapq",
    "ont_strand_bias",
    "ont_softclip_rate",
    "ont_A_fraction",
    "ont_C_fraction",
    "ont_G_fraction",
    "ont_T_fraction",
    "ont_alt_mean_bq",
    "gc_content",
    "homopolymer_length",
    "is_homopolymer",
    "homopolymer_position",
    "distance_to_homopolymer",
    "sequence_complexity",
)


def preprocess(feature_table, feature_columns=FEATURE_COLUMNS):
    """Return the complete ordered numeric feature matrix for inference."""
    if isinstance(feature_table, (str, bytes)):
        table = pd.read_csv(feature_table)
    elif isinstance(feature_table, pd.DataFrame):
        table = feature_table.copy()
    else:
        raise TypeError("feature_table must be a CSV path or pandas DataFrame")

    if "contig" not in table.columns:
        raise ValueError("Feature table must contain 'contig'")

    if "position" not in table.columns:
        if "assembly_position" in table.columns:
            table = table.rename(columns={"assembly_position": "position"})
        else:
            raise ValueError(
                "Feature table must contain 'position' or 'assembly_position'"
            )

    missing = [column for column in feature_columns if column not in table]
    if missing:
        raise ValueError(f"Feature table is missing columns: {missing}")

    values = table.loc[:, list(feature_columns)].apply(
        pd.to_numeric,
        errors="raise",
    )
    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise ValueError("Feature table contains NaN or infinite feature values")

    return values.to_numpy(dtype=np.float32)


def _sequence_context(sequence, position, window_size=21):
    base = sequence[position]
    left = position
    right = position
    while left > 0 and sequence[left - 1] == base:
        left -= 1
    while right < len(sequence) - 1 and sequence[right + 1] == base:
        right += 1
    run_length = right - left + 1
    is_homopolymer = int(base in "ACGT" and run_length >= 4)
    homopolymer_position = position - left + 1 if is_homopolymer else 0
    if is_homopolymer:
        distance_to_homopolymer = 0
    else:
        distance_to_homopolymer = -1
        for index in range(len(sequence)):
            candidate = sequence[index]
            start = index
            while start > 0 and sequence[start - 1] == candidate:
                start -= 1
            end = index
            while end < len(sequence) - 1 and sequence[end + 1] == candidate:
                end += 1
            if end - start + 1 >= 4:
                distance_to_homopolymer = min(
                    abs(position - start), abs(position - end)
                )
                break

    half_window = window_size // 2
    window = sequence[
        max(0, position - half_window) : min(len(sequence), position + half_window + 1)
    ]
    valid_bases = [item for item in window if item in "ACGT"]
    gc_content = (
        sum(item in "GC" for item in valid_bases) / len(valid_bases)
        if valid_bases
        else 0.0
    )
    counts = Counter(window)
    sequence_complexity = (
        -sum(
            (count / len(window)) * math.log2(count / len(window))
            for count in counts.values()
        )
        if window
        else 0.0
    )
    return {
        "distance_to_contig_end": min(position, len(sequence) - position - 1),
        "gc_content": gc_content,
        "homopolymer_length": run_length,
        "is_homopolymer": is_homopolymer,
        "homopolymer_position": homopolymer_position,
        "distance_to_homopolymer": distance_to_homopolymer,
        "sequence_complexity": sequence_complexity,
    }


def build_sequence_feature_table(assembly_fasta, window_size=21):
    """Build the assembly and sequence features expected by the model."""
    assembly_fasta = Path(assembly_fasta)
    if not assembly_fasta.is_file():
        raise FileNotFoundError(f"Assembly FASTA not found: {assembly_fasta}")
    rows = []
    fasta = pysam.FastaFile(str(assembly_fasta))
    try:
        for contig in fasta.references:
            sequence = fasta.fetch(contig).upper()
            for position, base in enumerate(sequence):
                row = {
                    "contig": contig,
                    "position": position + 1,
                    "assembly_base": base,
                }
                row.update(_sequence_context(sequence, position, window_size))
                rows.append(row)
    finally:
        fasta.close()
    return pd.DataFrame(rows)


def build_feature_table(illumina_bam, ont_bam, assembly_fasta, output_csv=None):
    """Extract and merge the complete feature table from two BAM files.

    The assembly provides the seven sequence-context features required by the
    trained ONNX model in addition to the 24 read-alignment features.
    """
    illumina_bam = Path(illumina_bam)
    ont_bam = Path(ont_bam)
    if not illumina_bam.is_file():
        raise FileNotFoundError(f"Illumina BAM not found: {illumina_bam}")
    if not ont_bam.is_file():
        raise FileNotFoundError(f"ONT BAM not found: {ont_bam}")

    with TemporaryDirectory() as temporary_directory:
        temporary_directory = Path(temporary_directory)
        illumina_csv = temporary_directory / "illumina_features.csv"
        ont_csv = temporary_directory / "ont_features.csv"

        extract_illumina_features(illumina_bam, illumina_csv)
        extract_ont_features(ont_bam, ont_csv)

        illumina = pd.read_csv(illumina_csv)
        ont = pd.read_csv(ont_csv)
        table = ont.merge(
            illumina,
            on=["contig", "position"],
            how="outer",
            validate="one_to_one",
            sort=False,
        )
    sequence = build_sequence_feature_table(assembly_fasta)
    key_columns = ["contig", "position"]
    sequence_indexed = sequence.set_index(key_columns)
    table_indexed = table.set_index(key_columns).reindex(sequence_indexed.index)
    table = pd.concat([sequence_indexed, table_indexed], axis=1).reset_index()

    table = table[["contig", "position", "assembly_base", *FEATURE_COLUMNS]]
    table.loc[:, list(FEATURE_COLUMNS)] = table.loc[:, list(FEATURE_COLUMNS)].fillna(
        0.0
    )

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_csv, index=False)

    return table


class ErrorDetectionModel:
    def __init__(self, model_path, providers=["CPUExecutionProvider"]):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(
        self,
        feature_table=None,
        *,
        assembly_fasta=None,
        illumina_bam=None,
        ont_bam=None,
        output_csv=None,
        threads=4,
    ):

        if feature_table is None:
            feature_table = build_feature_table(
                illumina_bam,
                ont_bam,
                assembly_fasta,
                output_csv=output_csv,
            )

        X = preprocess_for_base_err_detection(
            feature_table,
            assembly_fasta=assembly_fasta,
            illumina_bam=illumina_bam,
            ont_bam=ont_bam,
            output_csv=output_csv,
            threads=threads,
        )
        model_width = self.session.get_inputs()[0].shape[1]
        if model_width != X.shape[1]:
            raise ValueError(
                f"Model expects {model_width} features, but inference produced {X.shape[1]}"
            )
        outputs = self.session.run(None, {self.input_name: X})
        labels = np.asarray(outputs[0]).reshape(-1)
        probability_maps = outputs[1]
        probabilities = np.asarray(
            [
                float(probability_map.get(int(label), 0.0))
                for label, probability_map in zip(labels, probability_maps)
            ]
        )
        label_names = {
            0: "correct",
            1: "mismatch",
            2: "insertion",
            3: "deletion",
        }
        metadata = feature_table.reset_index(drop=True).copy()
        metadata["assembly_position"] = metadata["position"]
        metadata["predicted_label"] = [
            label_names.get(int(label), "correct") for label in labels
        ]
        metadata["error_proba"] = probabilities
        result = metadata
        # result.to_csv(output_csv, index=False)
        return result


def preprocess_for_base_err_detection(
    feature_table=None,
    *,
    assembly_fasta=None,
    illumina_bam=None,
    ont_bam=None,
    output_csv=None,
    threads=4,
):
    if feature_table is None:
        feature_table = build_feature_table(
            illumina_bam,
            ont_bam,
            assembly_fasta,
            output_csv=output_csv,
        )
    return preprocess(feature_table)
