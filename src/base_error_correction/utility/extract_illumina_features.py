#!/usr/bin/env python3

import csv
import pysam


# ============================================================================
# ILLUMINA PER-BASE FEATURE EXTRACTION
# ============================================================================
#
# Soft-clipping implementation:
#
#   LEFT soft clipping  -> first aligned reference position
#   RIGHT soft clipping -> last aligned reference position
#
# Examples:
#
#   5S95M       -> first_ref += 1
#   95M5S       -> last_ref += 1
#   5S90M5S     -> first_ref += 1 AND last_ref += 1
#   95M         -> no soft-clipping count
#
# CIGAR operation:
#   4 = S = soft clipping
#
# The softclip_rate is calculated as:
#
#   softclip_read_count[position] / coverage[position]
#
# ============================================================================


def extract_illumina_features(bam_path, output_csv):

    print("=" * 70)
    print("ILLUMINA PER-BASE FEATURE EXTRACTION")
    print("=" * 70)

    print("\nOpening BAM...")
    bam = pysam.AlignmentFile(bam_path, "rb")

    with open(output_csv, "w", newline="") as out:

        writer = csv.writer(out)

        # ====================================================================
        # OUTPUT HEADER
        # ====================================================================

        writer.writerow([
            "contig",
            "position",
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
            "illumina_alt_mean_bq"
        ])

        # ====================================================================
        # PROCESS EACH CONTIG
        # ====================================================================

        for contig in bam.references:

            length = bam.get_reference_length(contig)

            print(
                f"\nProcessing {contig} "
                f"({length:,} bp)..."
            )

            # =================================================================
            # PER-BASE ARRAYS
            # =================================================================

            coverage = [0] * length

            mismatch_count = [0] * length

            deletion_count = [0] * length

            insertion_count = [0] * length

            forward_count = [0] * length

            reverse_count = [0] * length

            # Number of reads whose LEFT or RIGHT soft clipping
            # is associated with this reference position.
            softclip_read_count = [0] * length

            mapq_sum = [0] * length

            mapq_count = [0] * length

            base_counts = {
                "A": [0] * length,
                "C": [0] * length,
                "G": [0] * length,
                "T": [0] * length
            }

            alt_bq_sum = [0.0] * length

            alt_bq_count = [0] * length

            # =================================================================
            # PROCESS READS
            # =================================================================

            for read in bam.fetch(contig):

                # -------------------------------------------------------------
                # Ignore reads that should not contribute to features
                # -------------------------------------------------------------

                if read.is_unmapped:
                    continue

                if read.is_secondary:
                    continue

                if read.is_supplementary:
                    continue

                mapq = read.mapping_quality

                # -------------------------------------------------------------
                # Get alignment pairs
                # -------------------------------------------------------------

                aligned_pairs = read.get_aligned_pairs(
                    matches_only=False,
                    with_seq=True
                )

                reference_positions = [
                    ref_pos
                    for query_pos, ref_pos, reference_base in aligned_pairs
                    if ref_pos is not None
                ]

                if not reference_positions:
                    continue

                first_ref = min(reference_positions)

                last_ref = max(reference_positions)

                # =============================================================
                # SOFT CLIPPING
                # =============================================================
                #
                # IMPORTANT:
                #
                # We DO NOT ask:
                #
                #     "Does this read have soft clipping anywhere?"
                #
                # and then mark both endpoints.
                #
                # Instead:
                #
                #     LEFT soft clip  -> first aligned position
                #     RIGHT soft clip -> last aligned position
                #
                # =============================================================

                if read.cigartuples:

                    # ---------------------------------------------------------
                    # First CIGAR operation
                    # ---------------------------------------------------------

                    first_cigar_op, first_cigar_len = (
                        read.cigartuples[0]
                    )

                    # ---------------------------------------------------------
                    # Last CIGAR operation
                    # ---------------------------------------------------------

                    last_cigar_op, last_cigar_len = (
                        read.cigartuples[-1]
                    )

                    # ---------------------------------------------------------
                    # LEFT SOFT CLIPPING
                    #
                    # CIGAR operation 4 = S
                    #
                    # Example:
                    #
                    #     5S95M
                    #
                    # The soft clipping is on the LEFT side.
                    # Associate it with first_ref.
                    # ---------------------------------------------------------

                    if (
                        first_cigar_op == 4
                        and first_cigar_len > 0
                    ):

                        if (
                            0 <= first_ref < length
                        ):

                            softclip_read_count[
                                first_ref
                            ] += 1

                    # ---------------------------------------------------------
                    # RIGHT SOFT CLIPPING
                    #
                    # Example:
                    #
                    #     95M5S
                    #
                    # The soft clipping is on the RIGHT side.
                    # Associate it with last_ref.
                    # ---------------------------------------------------------

                    if (
                        last_cigar_op == 4
                        and last_cigar_len > 0
                    ):

                        if (
                            last_ref != first_ref
                            and 0 <= last_ref < length
                        ):

                            softclip_read_count[
                                last_ref
                            ] += 1

                # =============================================================
                # PROCESS ALIGNMENT
                # =============================================================

                previous_reference_position = None

                for query_pos, ref_pos, reference_base in aligned_pairs:

                    # ---------------------------------------------------------
                    # INSERTION
                    #
                    # ref_pos is None for inserted query bases.
                    #
                    # The insertion is assigned to the reference position
                    # immediately before the insertion.
                    # ---------------------------------------------------------

                    if ref_pos is None:

                        if (
                            query_pos is not None
                            and previous_reference_position is not None
                        ):

                            if (
                                0 <= previous_reference_position < length
                            ):

                                insertion_count[
                                    previous_reference_position
                                ] += 1

                        continue

                    # ---------------------------------------------------------
                    # Check reference position
                    # ---------------------------------------------------------

                    if ref_pos < 0 or ref_pos >= length:
                        continue

                    # Update previous reference position
                    previous_reference_position = ref_pos

                    # ---------------------------------------------------------
                    # DELETION
                    #
                    # query_pos is None when a reference base is deleted
                    # from the read.
                    # ---------------------------------------------------------

                    if query_pos is None:

                        deletion_count[ref_pos] += 1

                        continue

                    # ---------------------------------------------------------
                    # NORMAL ALIGNED BASE
                    # ---------------------------------------------------------

                    coverage[ref_pos] += 1

                    mapq_sum[ref_pos] += mapq

                    mapq_count[ref_pos] += 1

                    # ---------------------------------------------------------
                    # STRAND
                    # ---------------------------------------------------------

                    if read.is_reverse:

                        reverse_count[ref_pos] += 1

                    else:

                        forward_count[ref_pos] += 1

                    # ---------------------------------------------------------
                    # QUERY BASE
                    # ---------------------------------------------------------

                    query_base = read.query_sequence[
                        query_pos
                    ].upper()

                    # ---------------------------------------------------------
                    # BASE QUALITY
                    # ---------------------------------------------------------

                    base_quality = read.query_qualities[
                        query_pos
                    ]

                    # ---------------------------------------------------------
                    # BASE COUNTS
                    # ---------------------------------------------------------

                    if query_base in base_counts:

                        base_counts[
                            query_base
                        ][ref_pos] += 1

                    # ---------------------------------------------------------
                    # ASSEMBLY / REFERENCE BASE
                    # ---------------------------------------------------------

                    # ---------------------------------------------------------
                    # MISMATCH
                    # ---------------------------------------------------------

                    if (
                        reference_base is not None
                        and query_base != reference_base.upper()
                    ):

                        mismatch_count[ref_pos] += 1

                        alt_bq_sum[ref_pos] += base_quality

                        alt_bq_count[ref_pos] += 1

            # =================================================================
            # WRITE PER-BASE FEATURES
            # =================================================================

            print(
                f"Writing features for {contig}..."
            )

            for pos in range(length):

                position = pos + 1

                cov = coverage[pos]

                # =============================================================
                # POSITIONS WITH COVERAGE
                # =============================================================

                if cov > 0:

                    # ---------------------------------------------------------
                    # MISMATCH RATE
                    # ---------------------------------------------------------

                    mismatch_rate = (
                        mismatch_count[pos] / cov
                    )

                    # ---------------------------------------------------------
                    # INSERTION RATE
                    # ---------------------------------------------------------

                    insertion_rate = (
                        insertion_count[pos] / cov
                    )

                    # ---------------------------------------------------------
                    # DELETION RATE
                    # ---------------------------------------------------------

                    deletion_rate = (
                        deletion_count[pos] / cov
                    )

                    # ---------------------------------------------------------
                    # MEAN MAPQ
                    # ---------------------------------------------------------

                    mean_mapq = (
                        mapq_sum[pos]
                        / mapq_count[pos]
                        if mapq_count[pos] > 0
                        else 0.0
                    )

                    # ---------------------------------------------------------
                    # STRAND BIAS
                    # ---------------------------------------------------------

                    forward_fraction = (
                        forward_count[pos] / cov
                    )

                    reverse_fraction = (
                        reverse_count[pos] / cov
                    )

                    strand_bias = abs(
                        forward_fraction
                        - reverse_fraction
                    )

                    # ---------------------------------------------------------
                    # SOFT-CLIPPING RATE
                    #
                    # IMPORTANT:
                    #
                    # softclip_read_count contains:
                    #
                    #     LEFT clips associated with first_ref
                    #     RIGHT clips associated with last_ref
                    #
                    # Therefore:
                    #
                    #     softclip_rate =
                    #         softclip_count / coverage
                    # ---------------------------------------------------------

                    softclip_rate = (
                        softclip_read_count[pos] / cov
                    )

                    # ---------------------------------------------------------
                    # BASE FRACTIONS
                    # ---------------------------------------------------------

                    A_fraction = (
                        base_counts["A"][pos] / cov
                    )

                    C_fraction = (
                        base_counts["C"][pos] / cov
                    )

                    G_fraction = (
                        base_counts["G"][pos] / cov
                    )

                    T_fraction = (
                        base_counts["T"][pos] / cov
                    )

                    # ---------------------------------------------------------
                    # ALTERNATE BASE MEAN QUALITY
                    # ---------------------------------------------------------

                    alt_mean_bq = (
                        alt_bq_sum[pos]
                        / alt_bq_count[pos]
                        if alt_bq_count[pos] > 0
                        else 0.0
                    )

                # =============================================================
                # POSITIONS WITHOUT COVERAGE
                # =============================================================

                else:

                    mismatch_rate = 0.0

                    insertion_rate = 0.0

                    deletion_rate = 0.0

                    mean_mapq = 0.0

                    strand_bias = 0.0

                    softclip_rate = 0.0

                    A_fraction = 0.0

                    C_fraction = 0.0

                    G_fraction = 0.0

                    T_fraction = 0.0

                    alt_mean_bq = 0.0

                # =============================================================
                # WRITE ROW
                # =============================================================

                writer.writerow([
                    contig,
                    position,
                    cov,
                    round(mismatch_rate, 6),
                    round(insertion_rate, 6),
                    round(deletion_rate, 6),
                    round(mean_mapq, 6),
                    round(strand_bias, 6),
                    round(softclip_rate, 6),
                    round(A_fraction, 6),
                    round(C_fraction, 6),
                    round(G_fraction, 6),
                    round(T_fraction, 6),
                    round(alt_mean_bq, 6)
                ])

    # ========================================================================
    # CLOSE FILES
    # ========================================================================

    bam.close()

    print("\n" + "=" * 70)
    print("ILLUMINA FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print(f"\nOutput saved to:")
    print(output_csv)


