from collections import Counter

import numpy as np
import pysam


def _consensus_from_observed_insertions(observed_insertions, min_supporting_reads=3):
    """Build an insertion consensus from supported reads."""
    if len(observed_insertions) < min_supporting_reads:
        return ""

    lengths = [length for length, _, _ in observed_insertions]
    consensus_length = Counter(lengths).most_common(1)[0][0]
    matching = [
        (sequence, qualities)
        for length, sequence, qualities in observed_insertions
        if length == consensus_length
    ]
    if len(matching) < min_supporting_reads:
        return ""

    position_votes = [Counter() for _ in range(consensus_length)]
    for sequence, qualities in matching:
        for index in range(consensus_length):
            position_votes[index][sequence[index]] += qualities[index]
    return "".join(votes.most_common(1)[0][0] for votes in position_votes)


def get_mismatch_consensus_base(bam_path: str, contig: str, position: int,
                                min_base_quality: int = 20,
                                min_mapping_quality: int = 20,
                                platform_tag: str = None) -> str:
    """Return the quality-weighted base consensus at an assembly position."""
    bam = pysam.AlignmentFile(bam_path, "rb")
    votes = Counter()
    try:
        for pileupcolumn in bam.pileup(
            contig,
            position,
            position + 1,
            min_base_quality=min_base_quality,
            min_mapping_quality=min_mapping_quality,
            truncate=True,
        ):
            for pileupread in pileupcolumn.pileups:
                if pileupread.is_del or pileupread.is_refskip:
                    continue
                read = pileupread.alignment
                if platform_tag and not read.has_tag(platform_tag):
                    continue
                query_position = pileupread.query_position
                base = read.query_sequence[query_position]
                base_quality = read.query_qualities[query_position]
                votes[base] += base_quality
    finally:
        bam.close()

    return votes.most_common(1)[0][0] if votes else ""


def get_mismatch_consensus_base_multi(
    bam_paths: dict,
    contig: str,
    position: int,
    min_base_quality: int = 20,
    min_mapping_quality: int = 20,
    platform_priority=("illumina", "ont"),
) -> str:
    """Try each platform BAM in priority order for a confident consensus."""
    for platform in platform_priority:
        bam_path = bam_paths.get(platform)
        if not bam_path:
            continue
        base = get_mismatch_consensus_base(
            bam_path,
            contig,
            position,
            min_base_quality=min_base_quality,
            min_mapping_quality=min_mapping_quality,
        )
        if base:
            return base
    return ""


def _collect_observed_insertions(bam_path: str, contig: str,
                                 anchor_position: int,
                                 min_base_quality: int = 15,
                                 min_mapping_quality: int = 20) -> list:
    """Collect supported CIGAR insertions anchored at a reference position."""
    bam = pysam.AlignmentFile(bam_path, "rb")
    observed_insertions = []
    try:
        for read in bam.fetch(contig, anchor_position, anchor_position + 1):
            if read.is_unmapped or read.is_secondary:
                continue
            if read.mapping_quality < min_mapping_quality:
                continue

            reference_position = read.reference_start
            query_position = 0
            for operation, length in read.cigartuples:
                if operation in (0, 7, 8):
                    reference_position += length
                    query_position += length
                elif operation == 1:
                    if reference_position - 1 == anchor_position:
                        sequence = read.query_sequence[
                            query_position:query_position + length
                        ]
                        qualities = read.query_qualities[
                            query_position:query_position + length
                        ]
                        if any(
                            quality < min_base_quality for quality in qualities
                        ):
                            query_position += length
                            continue
                        observed_insertions.append((length, sequence, qualities))
                    query_position += length
                elif operation in (2, 3):
                    reference_position += length
                elif operation == 4:
                    query_position += length
    finally:
        bam.close()
    return observed_insertions


def get_deletion_insert_sequence(bam_path: str, contig: str,
                                 anchor_position: int,
                                 min_base_quality: int = 15,
                                 min_mapping_quality: int = 20,
                                 min_supporting_reads: int = 3) -> str:
    """Recover the inserted sequence that replaces a deletion."""
    observed_insertions = _collect_observed_insertions(
        bam_path,
        contig,
        anchor_position,
        min_base_quality=min_base_quality,
        min_mapping_quality=min_mapping_quality,
    )
    return _consensus_from_observed_insertions(
        observed_insertions,
        min_supporting_reads,
    )


def get_deletion_insert_sequence_multi(
    bam_paths: dict,
    contig: str,
    anchor_position: int,
    min_base_quality: int = 15,
    min_mapping_quality: int = 20,
    min_supporting_reads: int = 3,
    platform_priority=("illumina", "ont"),
) -> str:
    """Try platform-specific insertion consensuses, then pool platform reads."""
    for platform in platform_priority:
        bam_path = bam_paths.get(platform)
        if not bam_path:
            continue
        sequence = get_deletion_insert_sequence(
            bam_path,
            contig,
            anchor_position,
            min_base_quality=min_base_quality,
            min_mapping_quality=min_mapping_quality,
            min_supporting_reads=min_supporting_reads,
        )
        if sequence:
            return sequence

    pooled = []
    for bam_path in bam_paths.values():
        pooled.extend(
            _collect_observed_insertions(
                bam_path,
                contig,
                anchor_position,
                min_base_quality=min_base_quality,
                min_mapping_quality=min_mapping_quality,
            )
        )
    return _consensus_from_observed_insertions(pooled, min_supporting_reads)


def apply_corrections(assembly_seq: str, corrections: list) -> str:
    """Apply correction records from highest position to lowest position."""
    sequence = list(assembly_seq)
    for correction in sorted(
        corrections,
        key=lambda item: item["position"],
        reverse=True,
    ):
        position = correction["position"]
        if correction["type"] == "mismatch":
            sequence[position] = correction["value"]
        elif correction["type"] == "insertion":
            del sequence[position]
        elif correction["type"] == "deletion":
            sequence[position + 1:position + 1] = list(correction["value"])
        else:
            raise ValueError(f"unknown correction type {correction['type']}")
    return "".join(sequence)


def attach_predictions_from_multiclass_proba(df, proba: np.ndarray,
                                              class_labels):
    """Attach winning class names and probabilities to a prediction table."""
    class_labels = np.asarray(class_labels)
    argmax_indices = proba.argmax(axis=1)
    result = df.copy()
    result["predicted_error_type"] = class_labels[argmax_indices]
    result["error_proba"] = proba[
        np.arange(len(proba)),
        argmax_indices,
    ]
    return result


def build_corrections_from_predictions(
    df,
    bam_paths: dict,
    contig: str,
    error_type_col: str = "predicted_label",
    proba_col: str = "error_proba",
    correction_threshold: float = 0.5,
) -> list:
    """Build correction records for one contig."""
    confident = df[
        (df[error_type_col] != "correct")
        & (df[proba_col] >= correction_threshold)
    ]
    print(
        f"    [{contig}] {len(confident)}/{len(df)} flagged positions pass "
        f"threshold {correction_threshold}"
    )

    corrections = []
    for _, row in confident.iterrows():
        position = int(row["assembly_position"]) - 1
        error_type = row[error_type_col]
        if error_type == "mismatch":
            base = get_mismatch_consensus_base_multi(bam_paths, contig, position)
            if base and base != row["assembly_base"]:
                corrections.append({
                    "position": position,
                    "type": "mismatch",
                    "value": base,
                })
        elif error_type == "insertion":
            corrections.append({
                "position": position,
                "type": "insertion",
                "value": None,
            })
        elif error_type == "deletion":
            sequence = get_deletion_insert_sequence_multi(
                bam_paths,
                contig,
                position,
            )
            if sequence:
                corrections.append({
                    "position": position,
                    "type": "deletion",
                    "value": sequence,
                })
    return corrections


def build_corrections_for_all_assemblies(
    df,
    bam_paths: dict,
    error_type_col: str = "predicted_label",
    proba_col: str = "error_proba",
    correction_threshold: float = 0.5,
) -> dict:
    """Build corrections grouped by contig for one assembly."""
    all_corrections = {}
    for contig, group in df.groupby("contig"):
        all_corrections[contig] = build_corrections_from_predictions(
            group,
            bam_paths,
            contig,
            error_type_col=error_type_col,
            proba_col=proba_col,
            correction_threshold=correction_threshold,
        )
    return all_corrections


def load_assembly_sequences(fasta_path: str) -> dict:
    """Load every contig sequence from a FASTA file."""
    fasta = pysam.FastaFile(fasta_path)
    try:
        return {name: fasta.fetch(name) for name in fasta.references}
    finally:
        fasta.close()


def write_fasta(path: str, sequences: dict, line_width: int = 70) -> None:
    """Write contig sequences as a wrapped multi-FASTA."""
    with open(path, "w") as handle:
        for name, sequence in sequences.items():
            handle.write(f">{name}\n")
            for index in range(0, len(sequence), line_width):
                handle.write(sequence[index:index + line_width] + "\n")


def correct_assemblies(
    df,
    bam_paths: dict,
    assembly_path: str,
    error_type_col: str = "predicted_label",
    proba_col: str = "error_proba",
    correction_threshold: float = 0.5,
) -> dict:
    """Apply predictions and read-backed corrections to one assembly."""
    all_corrections = build_corrections_for_all_assemblies(
        df,
        bam_paths,
        error_type_col=error_type_col,
        proba_col=proba_col,
        correction_threshold=correction_threshold,
    )

    sequences = load_assembly_sequences(assembly_path)
    for contig, corrections in all_corrections.items():
        if contig not in sequences:
            raise KeyError(f"contig {contig!r} is not present in {assembly_path!r}")
        if corrections:
            sequences[contig] = apply_corrections(sequences[contig], corrections)
    return sequences