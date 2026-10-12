#!/usr/bin/env python3

import csv
import pysam


def extract_illumina_features(bam_path, output_csv):

    print("\nIllumina per-base feature extraction...")

    print("\nOpening BAM...")
    bam = pysam.AlignmentFile(bam_path, "rb")

    with open(output_csv, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(
            [
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
                "illumina_alt_mean_bq",
            ]
        )

        for contig in bam.references:
            length = bam.get_reference_length(contig)
            print(f"\nProcessing {contig} ({length:,} bp)...")

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
                "T": [0] * length,
            }

            alt_bq_sum = [0.0] * length
            alt_bq_count = [0] * length

            for read in bam.fetch(contig):
                if read.is_unmapped:
                    continue

                if read.is_secondary:
                    continue

                if read.is_supplementary:
                    continue

                mapq = read.mapping_quality

                aligned_pairs = read.get_aligned_pairs(
                    matches_only=False, with_seq=True
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

                if read.cigartuples:
                    first_cigar_op, first_cigar_len = read.cigartuples[0]

                    last_cigar_op, last_cigar_len = read.cigartuples[-1]

                    if first_cigar_op == 4 and first_cigar_len > 0:
                        if 0 <= first_ref < length:
                            softclip_read_count[first_ref] += 1

                    if last_cigar_op == 4 and last_cigar_len > 0:
                        if last_ref != first_ref and 0 <= last_ref < length:
                            softclip_read_count[last_ref] += 1

                previous_reference_position = None

                for query_pos, ref_pos, reference_base in aligned_pairs:
                    if ref_pos is None:
                        if (
                            query_pos is not None
                            and previous_reference_position is not None
                        ):
                            if 0 <= previous_reference_position < length:
                                insertion_count[previous_reference_position] += 1

                        continue

                    if ref_pos < 0 or ref_pos >= length:
                        continue

                    # Update previous reference position
                    previous_reference_position = ref_pos

                    if query_pos is None:
                        deletion_count[ref_pos] += 1

                        continue

                    coverage[ref_pos] += 1

                    mapq_sum[ref_pos] += mapq

                    mapq_count[ref_pos] += 1

                    if read.is_reverse:
                        reverse_count[ref_pos] += 1

                    else:
                        forward_count[ref_pos] += 1

                    query_base = read.query_sequence[query_pos].upper()

                    base_quality = read.query_qualities[query_pos]

                    if query_base in base_counts:
                        base_counts[query_base][ref_pos] += 1

                    if (
                        reference_base is not None
                        and query_base != reference_base.upper()
                    ):
                        mismatch_count[ref_pos] += 1

                        alt_bq_sum[ref_pos] += base_quality

                        alt_bq_count[ref_pos] += 1

            print(f"Writing features for {contig}...")

            for pos in range(length):
                position = pos + 1
                cov = coverage[pos]
                if cov > 0:
                    mismatch_rate = mismatch_count[pos] / cov
                    insertion_rate = insertion_count[pos] / cov
                    deletion_rate = deletion_count[pos] / cov
                    mean_mapq = (
                        mapq_sum[pos] / mapq_count[pos] if mapq_count[pos] > 0 else 0.0
                    )

                    forward_fraction = forward_count[pos] / cov
                    reverse_fraction = reverse_count[pos] / cov
                    strand_bias = abs(forward_fraction - reverse_fraction)
                    softclip_rate = softclip_read_count[pos] / cov

                    A_fraction = base_counts["A"][pos] / cov
                    C_fraction = base_counts["C"][pos] / cov
                    G_fraction = base_counts["G"][pos] / cov
                    T_fraction = base_counts["T"][pos] / cov

                    alt_mean_bq = (
                        alt_bq_sum[pos] / alt_bq_count[pos]
                        if alt_bq_count[pos] > 0
                        else 0.0
                    )

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

                writer.writerow(
                    [
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
                        round(alt_mean_bq, 6),
                    ]
                )

    bam.close()

    print("\nIllumina feature extraction complete.")
    print("Output saved to:")
    print(output_csv)
