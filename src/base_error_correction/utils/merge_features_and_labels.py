#!/usr/bin/env python3

import os
import pandas as pd


# ============================================================================
# V3.2
# ============================================================================
#
# Purpose:
#   Create a one-to-one final dataset with exactly one row per assembly
#   position in the selected main contig.
#
# V3.2 additions:
#
#   1. Keeps the V3.1 one-to-one design.
#   2. Deletion events are mapped to an existing assembly position.
#   3. Multiple deletion events mapped to the same assembly position are
#      collapsed into ONE row.
#   4. NEW: deletion_count records HOW MANY reference bases are deleted
#      at that assembly position.
#
# Example:
#
#   Reference:
#       A T G C C C C A
#
#   Assembly:
#       A T G       A
#
#   Four reference bases are deleted after assembly position 3.
#
#   Final:
#
#       assembly_position = 3
#       deletion          = 1
#       deletion_count    = 4
#
# This means:
#
#       deletion = 1
#       deletion_count = number of reference bases deleted at this position
#
# Insertions and mismatches do not receive a count column because they
# already correspond to an assembly position.
#
# ============================================================================


# ============================================================================
# MAIN
# ============================================================================

def merge_features_and_labels(features_path, labels_path, output_path):

    print("=" * 70)
    print("CREATING FINAL DATASET V3.2")
    print("=" * 70)

    # ========================================================================
    # 1. READ FEATURES
    # ========================================================================

    print("\nReading features...")

    features = pd.read_csv(features_path)

    print("Feature rows:", len(features))
    print("Feature columns:", len(features.columns))

    # ========================================================================
    # 2. READ LABELS
    # ========================================================================

    print("\nReading labels...")

    labels = pd.read_csv(labels_path)

    print("Label rows:", len(labels))
    print("Label columns:", len(labels.columns))

    # ========================================================================
    # 3. CHECK FEATURE COLUMNS
    # ========================================================================

    if "contig" not in features.columns:

        raise ValueError(
            "Feature CSV does not contain 'contig'."
        )

    if "position" in features.columns:

        features["assembly_position"] = pd.to_numeric(
            features["position"],
            errors="raise"
        ).astype("int64")

    elif "assembly_position" in features.columns:

        features["assembly_position"] = pd.to_numeric(
            features["assembly_position"],
            errors="raise"
        ).astype("int64")

    else:

        raise ValueError(
            "Feature CSV does not contain "
            "'position' or 'assembly_position'."
        )

    # ========================================================================
    # 4. SELECT MAIN CONTIG
    # ========================================================================

    print("\nFeature contigs:")

    contig_counts = features["contig"].value_counts()

    print(contig_counts)

    main_contig = contig_counts.idxmax()

    print("\nSelected main contig:")
    print(main_contig)

    features = features[
        features["contig"] == main_contig
    ].copy()

    print(
        "Main contig feature rows:",
        len(features)
    )

    # ========================================================================
    # 5. FEATURE POSITION VALIDATION
    # ========================================================================

    feature_positions = features["assembly_position"]

    print("\nFeature position validation:")

    print("Rows:", len(feature_positions))
    print("Unique:", feature_positions.nunique())
    print("Minimum:", feature_positions.min())
    print("Maximum:", feature_positions.max())

    if feature_positions.isna().any():

        raise ValueError(
            "Feature positions contain missing values."
        )

    if (
        len(feature_positions)
        != feature_positions.nunique()
    ):

        raise ValueError(
            "Feature positions are duplicated."
        )

    expected_positions = pd.RangeIndex(
        start=1,
        stop=len(features) + 1
    )

    actual_positions = pd.Index(
        feature_positions
    )

    positions_are_continuous = (
        actual_positions.equals(
            expected_positions
        )
    )

    if positions_are_continuous:

        print("Continuous positions: PASS")

    else:

        print("Continuous positions: WARNING")

    # ========================================================================
    # 6. CHECK LABEL COLUMNS
    # ========================================================================

    required_label_columns = [
        "assembly_position",
        "reference_position",
        "insertion",
        "deletion",
        "mismatch",
        "error_label",
        "error_type"
    ]

    missing_columns = [
        column
        for column in required_label_columns
        if column not in labels.columns
    ]

    if missing_columns:

        raise ValueError(
            "Label CSV is missing columns: "
            + ", ".join(missing_columns)
        )

    # ========================================================================
    # 7. NORMALIZE LABEL TYPES
    # ========================================================================

    labels["assembly_position"] = pd.to_numeric(
        labels["assembly_position"],
        errors="coerce"
    )

    labels["reference_position"] = pd.to_numeric(
        labels["reference_position"],
        errors="coerce"
    )

    for column in [
        "insertion",
        "deletion",
        "mismatch"
    ]:

        labels[column] = pd.to_numeric(
            labels[column],
            errors="coerce"
        ).fillna(0).astype("int8")

    # ========================================================================
    # 8. ORIGINAL LABEL COUNTS
    # ========================================================================

    raw_deletions = int(
        labels["deletion"].sum()
    )

    raw_insertions = int(
        labels["insertion"].sum()
    )

    raw_mismatches = int(
        labels["mismatch"].sum()
    )

    print("\nOriginal label counts:")

    print(
        "Deletions :",
        raw_deletions
    )

    print(
        "Insertions:",
        raw_insertions
    )

    print(
        "Mismatches:",
        raw_mismatches
    )

    print(
        "Total errors:",
        raw_deletions
        + raw_insertions
        + raw_mismatches
    )

    # ========================================================================
    # 9. SEPARATE NORMAL AND DELETION LABELS
    # ========================================================================

    normal_labels = labels[
        labels["assembly_position"].notna()
    ].copy()

    deletion_labels = labels[
        labels["assembly_position"].isna()
        & (labels["deletion"] == 1)
    ].copy()

    print("\nLabel categories:")

    print(
        "Normal assembly-position labels:",
        len(normal_labels)
    )

    print(
        "Deletion labels without assembly position:",
        len(deletion_labels)
    )

    # ========================================================================
    # 10. NORMAL LABEL POSITIONS
    # ========================================================================

    normal_labels["mapped_assembly_position"] = (
        normal_labels["assembly_position"]
        .astype("int64")
    )

    # ========================================================================
    # 11. NORMAL POSITION VALIDATION
    # ========================================================================

    normal_rows = len(normal_labels)

    normal_unique_positions = (
        normal_labels[
            "mapped_assembly_position"
        ].nunique()
    )

    print("\nNormal label position validation:")

    print(
        "Normal label rows:",
        normal_rows
    )

    print(
        "Unique normal positions:",
        normal_unique_positions
    )

    print(
        "Duplicate normal positions:",
        normal_rows - normal_unique_positions
    )

    # ========================================================================
    # 12. MAP DELETIONS
    # ========================================================================

    print("\n" + "-" * 70)
    print("DELETION MAPPING")
    print("-" * 70)

    if len(deletion_labels) > 0:

        # Only retain columns needed for deletion processing.
        labels_with_context = labels[
            [
                "assembly_position",
                "reference_position",
                "insertion",
                "deletion",
                "mismatch"
            ]
        ].copy()

        # Previous real assembly position.
        labels_with_context[
            "previous_assembly_position"
        ] = (
            labels_with_context[
                "assembly_position"
            ].ffill()
        )

        # Next real assembly position.
        labels_with_context[
            "next_assembly_position"
        ] = (
            labels_with_context[
                "assembly_position"
            ].bfill()
        )

        deletion_labels = labels_with_context[
            labels_with_context["assembly_position"].isna()
            & (labels_with_context["deletion"] == 1)
        ].copy()

        # ------------------------------------------------------------
        # Map deletion to previous assembly position.
        # ------------------------------------------------------------

        deletion_labels[
            "mapped_assembly_position"
        ] = (
            deletion_labels[
                "previous_assembly_position"
            ]
        )

        # Beginning-of-assembly deletion.
        deletion_labels[
            "mapped_assembly_position"
        ] = (
            deletion_labels[
                "mapped_assembly_position"
            ].fillna(
                deletion_labels[
                    "next_assembly_position"
                ]
            )
        )

        if deletion_labels[
            "mapped_assembly_position"
        ].isna().any():

            raise ValueError(
                "Some deletion labels could not be mapped."
            )

        deletion_labels[
            "mapped_assembly_position"
        ] = (
            deletion_labels[
                "mapped_assembly_position"
            ].astype("int64")
        )

        # ====================================================================
        # NEW V3.2
        # Each deletion event represents ONE deleted reference base.
        #
        # Therefore, when multiple deletion rows map to the same assembly
        # position, the number of rows is the number of deleted bases.
        # ====================================================================

        deletion_labels["deletion_count"] = 1

        print(
            "Raw deletion events:",
            len(deletion_labels)
        )

        print(
            "Unique mapped deletion positions:",
            deletion_labels[
                "mapped_assembly_position"
            ].nunique()
        )

        print(
            "Deletion events collapsed:",
            len(deletion_labels)
            - deletion_labels[
                "mapped_assembly_position"
            ].nunique()
        )

        # ====================================================================
        # Validate deletion mapping
        # ====================================================================

        feature_min = int(
            feature_positions.min()
        )

        feature_max = int(
            feature_positions.max()
        )

        mapped_positions = deletion_labels[
            "mapped_assembly_position"
        ]

        invalid_mask = (
            (mapped_positions < feature_min)
            | (mapped_positions > feature_max)
        )

        invalid_count = int(
            invalid_mask.sum()
        )

        print(
            "Mapped positions outside feature range:",
            invalid_count
        )

        if invalid_count > 0:

            print(
                "First invalid positions:",
                mapped_positions[
                    invalid_mask
                ].head(20).tolist()
            )

            raise ValueError(
                "Deletion mapping contains invalid "
                "assembly positions."
            )

        # ====================================================================
        # Show deletion mappings
        # ====================================================================

        print("\nFirst 20 deletion mappings:")

        print(
            deletion_labels[
                [
                    "reference_position",
                    "previous_assembly_position",
                    "mapped_assembly_position"
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

        # ====================================================================
        # Show deletion counts by assembly position
        # ====================================================================

        deletion_summary = (
            deletion_labels
            .groupby(
                "mapped_assembly_position",
                sort=False
            )
            .size()
            .reset_index(
                name="deletion_count"
            )
        )

        print("\nDeletion count summary:")

        print(
            deletion_summary
            .head(20)
            .to_string(index=False)
        )

        print(
            "\nMaximum deleted bases at one assembly position:",
            int(
                deletion_summary[
                    "deletion_count"
                ].max()
            )
        )

    else:

        print(
            "No deletion labels require mapping."
        )

        deletion_summary = pd.DataFrame(
            columns=[
                "mapped_assembly_position",
                "deletion_count"
            ]
        )

    # ========================================================================
    # 13. CREATE NORMAL LABEL TABLE
    # ========================================================================

    normal_for_merge = normal_labels[
        [
            "mapped_assembly_position",
            "insertion",
            "deletion",
            "mismatch"
        ]
    ].copy()

    # Normal assembly positions have zero deletion count initially.
    normal_for_merge["deletion_count"] = 0

    # ========================================================================
    # 14. CREATE DELETION LABEL TABLE
    # ========================================================================

    if len(deletion_labels) > 0:

        deletion_for_merge = (
            deletion_labels[
                [
                    "mapped_assembly_position",
                    "insertion",
                    "deletion",
                    "mismatch",
                    "deletion_count"
                ]
            ]
            .copy()
        )

        all_label_rows = pd.concat(
            [
                normal_for_merge,
                deletion_for_merge
            ],
            ignore_index=True
        )

    else:

        all_label_rows = normal_for_merge

    # ========================================================================
    # 15. UNIFIED LABEL TABLE
    # ========================================================================
    #
    # IMPORTANT:
    #
    #   insertion  -> MAX
    #   deletion   -> MAX
    #   mismatch   -> MAX
    #   deletion_count -> SUM
    #
    # SUM is essential for deletion_count.
    #
    # Example:
    #
    #   position 100:
    #
    #       deletion_count = 1
    #       deletion_count = 1
    #       deletion_count = 1
    #       deletion_count = 1
    #       deletion_count = 1
    #
    #   becomes:
    #
    #       position 100
    #       deletion = 1
    #       deletion_count = 5
    #
    # ========================================================================

    label_table = (
        all_label_rows
        .groupby(
            "mapped_assembly_position",
            sort=False,
            as_index=False
        )
        .agg(
            insertion=("insertion", "max"),
            deletion=("deletion", "max"),
            mismatch=("mismatch", "max"),
            deletion_count=("deletion_count", "sum")
        )
    )

    label_table.rename(
        columns={
            "mapped_assembly_position":
                "assembly_position"
        },
        inplace=True
    )

    label_table["deletion_count"] = (
        label_table[
            "deletion_count"
        ]
        .astype("int32")
    )

    # ========================================================================
    # 16. UNIFIED LABEL TABLE VALIDATION
    # ========================================================================

    print("\n" + "=" * 70)
    print("UNIFIED LABEL TABLE")
    print("=" * 70)

    print(
        "Rows:",
        len(label_table)
    )

    print(
        "Unique positions:",
        label_table[
            "assembly_position"
        ].nunique()
    )

    if (
        len(label_table)
        != label_table[
            "assembly_position"
        ].nunique()
    ):

        raise ValueError(
            "Unified label table contains duplicate positions."
        )

    # ========================================================================
    # 17. LABEL COVERAGE
    # ========================================================================

    print("\n" + "-" * 70)
    print("LABEL COVERAGE")
    print("-" * 70)

    feature_position_index = pd.Index(
        features["assembly_position"]
    )

    label_position_index = pd.Index(
        label_table["assembly_position"]
    )

    missing_label_positions = (
        feature_position_index
        .difference(
            label_position_index
        )
    )

    extra_label_positions = (
        label_position_index
        .difference(
            feature_position_index
        )
    )

    print(
        "Feature positions:",
        len(feature_position_index)
    )

    print(
        "Label positions:",
        len(label_position_index)
    )

    print(
        "Feature positions without labels:",
        len(missing_label_positions)
    )

    print(
        "Label positions outside features:",
        len(extra_label_positions)
    )

    if len(extra_label_positions) > 0:

        print(
            "First extra label positions:",
            extra_label_positions[
                :20
            ].tolist()
        )

        raise ValueError(
            "Label positions exist outside features."
        )

    # ========================================================================
    # 18. MERGE
    # ========================================================================

    print("\nMerging features and labels...")

    merged = features.merge(
        label_table,
        on="assembly_position",
        how="left",
        sort=False,
        validate="one_to_one"
    )

    # ========================================================================
    # 19. FILL MISSING LABELS
    # ========================================================================

    merged["insertion"] = (
        merged["insertion"]
        .fillna(0)
        .astype("int8")
    )

    merged["deletion"] = (
        merged["deletion"]
        .fillna(0)
        .astype("int8")
    )

    merged["mismatch"] = (
        merged["mismatch"]
        .fillna(0)
        .astype("int8")
    )

    # ========================================================================
    # NEW V3.2:
    # Every non-deletion position receives deletion_count = 0.
    # ========================================================================

    merged["deletion_count"] = (
        merged["deletion_count"]
        .fillna(0)
        .astype("int32")
    )

    # ========================================================================
    # 20. DELETION CONSISTENCY CHECK
    # ========================================================================

    invalid_deletion_count = (
        (merged["deletion"] == 0)
        & (merged["deletion_count"] != 0)
    )

    if invalid_deletion_count.any():

        bad_count = int(
            invalid_deletion_count.sum()
        )

        raise ValueError(
            f"Found {bad_count} positions with "
            "deletion_count > 0 but deletion = 0."
        )

    invalid_deletion_flag = (
        (merged["deletion"] == 1)
        & (merged["deletion_count"] <= 0)
    )

    if invalid_deletion_flag.any():

        bad_count = int(
            invalid_deletion_flag.sum()
        )

        raise ValueError(
            f"Found {bad_count} deletion positions "
            "with deletion_count = 0."
        )

    # ========================================================================
    # 21. ERROR LABEL
    # ========================================================================

    merged["error_label"] = (
        (
            (merged["insertion"] == 1)
            | (merged["deletion"] == 1)
            | (merged["mismatch"] == 1)
        )
        .astype("int8")
    )

    # ========================================================================
    # 22. ERROR TYPE
    # ========================================================================

    merged["error_type"] = "correct"

    deletion_mask = (
        merged["deletion"] == 1
    )

    insertion_mask = (
        (merged["insertion"] == 1)
        & ~deletion_mask
    )

    mismatch_mask = (
        (merged["mismatch"] == 1)
        & ~deletion_mask
        & ~insertion_mask
    )

    merged.loc[
        deletion_mask,
        "error_type"
    ] = "deletion"

    merged.loc[
        insertion_mask,
        "error_type"
    ] = "insertion"

    merged.loc[
        mismatch_mask,
        "error_type"
    ] = "mismatch"

    # ========================================================================
    # 23. FINAL VALIDATION
    # ========================================================================

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    final_rows = len(merged)

    unique_positions = (
        merged[
            "assembly_position"
        ].nunique()
    )

    min_position = (
        merged[
            "assembly_position"
        ].min()
    )

    max_position = (
        merged[
            "assembly_position"
        ].max()
    )

    actual_deletions = int(
        merged["deletion"].sum()
    )

    actual_insertions = int(
        merged["insertion"].sum()
    )

    actual_mismatches = int(
        merged["mismatch"].sum()
    )

    total_deleted_bases = int(
        merged["deletion_count"].sum()
    )

    # ========================================================================
    # Expected unique positions
    # ========================================================================

    expected_deletion_positions = (
        deletion_labels[
            "mapped_assembly_position"
        ].nunique()
        if len(deletion_labels) > 0
        else 0
    )

    expected_insertion_positions = (
        normal_labels.loc[
            normal_labels["insertion"] == 1,
            "mapped_assembly_position"
        ].nunique()
    )

    expected_mismatch_positions = (
        normal_labels.loc[
            normal_labels["mismatch"] == 1,
            "mapped_assembly_position"
        ].nunique()
    )

    print(
        "Rows:",
        final_rows
    )

    print(
        "Unique positions:",
        unique_positions
    )

    print(
        "Position range:",
        min_position,
        "-",
        max_position
    )

    print("\nRaw label events:")

    print(
        "Deletion events :",
        raw_deletions
    )

    print(
        "Insertion events:",
        raw_insertions
    )

    print(
        "Mismatch events :",
        raw_mismatches
    )

    print("\nUnique position labels:")

    print(
        "Deletion positions :",
        actual_deletions
    )

    print(
        "Insertion positions:",
        actual_insertions
    )

    print(
        "Mismatch positions:",
        actual_mismatches
    )

    print(
        "Total deleted bases:",
        total_deleted_bases
    )

    print("\nExpected unique position labels:")

    print(
        "Deletion positions :",
        expected_deletion_positions
    )

    print(
        "Insertion positions:",
        expected_insertion_positions
    )

    print(
        "Mismatch positions:",
        expected_mismatch_positions
    )

    # ========================================================================
    # 24. DELETION COUNT VALIDATION
    # ========================================================================

    deletion_count_pass = (
        total_deleted_bases
        == raw_deletions
    )

    print(
        "\nDeletion count validation:",
        "PASS"
        if deletion_count_pass
        else "FAIL"
    )

    print(
        "Raw deletion events:",
        raw_deletions
    )

    print(
        "Final deletion_count sum:",
        total_deleted_bases
    )

    # ========================================================================
    # 25. VALIDATION FLAGS
    # ========================================================================

    rows_pass = (
        final_rows == len(features)
    )

    positions_pass = (
        unique_positions == final_rows
    )

    deletion_pass = (
        actual_deletions
        == expected_deletion_positions
    )

    insertion_pass = (
        actual_insertions
        == expected_insertion_positions
    )

    mismatch_pass = (
        actual_mismatches
        == expected_mismatch_positions
    )

    extra_label_pass = (
        len(extra_label_positions) == 0
    )

    # ========================================================================
    # 26. PRINT VALIDATION
    # ========================================================================

    print("\nValidation:")

    print(
        "Rows              :",
        "PASS" if rows_pass else "FAIL"
    )

    print(
        "Positions         :",
        "PASS" if positions_pass else "FAIL"
    )

    print(
        "Deletion positions:",
        "PASS" if deletion_pass else "FAIL"
    )

    print(
        "Insertion positions:",
        "PASS" if insertion_pass else "FAIL"
    )

    print(
        "Mismatch positions:",
        "PASS" if mismatch_pass else "FAIL"
    )

    print(
        "Deletion count     :",
        "PASS" if deletion_count_pass else "FAIL"
    )

    print(
        "Label positions   :",
        "PASS" if extra_label_pass else "FAIL"
    )

    # ========================================================================
    # 27. FINAL STATUS
    # ========================================================================

    validation_passed = (
        rows_pass
        and positions_pass
        and deletion_pass
        and insertion_pass
        and mismatch_pass
        and deletion_count_pass
        and extra_label_pass
    )

    if validation_passed:

        print("\nSTATUS: PASS")

        # ------------------------------------------------------------
        # Create output directory
        # ------------------------------------------------------------

        output_directory = os.path.dirname(
            os.path.abspath(output_path)
        )

        os.makedirs(
            output_directory,
            exist_ok=True
        )

        # ------------------------------------------------------------
        # Save
        # ------------------------------------------------------------

        print("\nSaving dataset...")

        merged.to_csv(
            output_path,
            index=False
        )

        print("\nDataset saved:")
        print(output_path)

        print("\nFinal dataset shape:")
        print(
            merged.shape
        )

        # ------------------------------------------------------------
        # Deletion summary
        # ------------------------------------------------------------

        print("\nDeletion summary:")

        deletion_positions = merged.loc[
            merged["deletion"] == 1,
            [
                "assembly_position",
                "deletion_count"
            ]
        ]

        if len(deletion_positions) > 0:

            print(
                deletion_positions
                .head(20)
                .to_string(index=False)
            )

        else:

            print(
                "No deletion positions."
            )

        # ------------------------------------------------------------
        # Error distribution
        # ------------------------------------------------------------

        print("\nFinal error distribution:")

        print(
            merged[
                "error_type"
            ].value_counts()
        )

        print("\n" + "=" * 70)
        print("V3.2 COMPLETE")
        print("=" * 70)

    else:

        print("\nSTATUS: FAIL")

        print(
            "\nDataset was NOT saved because "
            "validation failed."
        )

        print("\nDiagnostic summary:")

        print(
            "Feature rows:",
            len(features)
        )

        print(
            "Final rows:",
            final_rows
        )

        print(
            "Raw deletion events:",
            raw_deletions
        )

        print(
            "Final deletion positions:",
            actual_deletions
        )

        print(
            "Expected deletion positions:",
            expected_deletion_positions
        )

        print(
            "Final deletion_count sum:",
            total_deleted_bases
        )

        print(
            "Feature positions without labels:",
            len(missing_label_positions)
        )

        print(
            "Label positions outside features:",
            len(extra_label_positions)
        )

        raise SystemExit(1)


