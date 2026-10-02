#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path


# ============================================================
# Utility functions
# ============================================================

def print_header(message):
    print("\n" + "=" * 75)
    print(message)
    print("=" * 75)


def run_command(cmd, description, stdout_file=None):
    print_header(description)

    print("Command:")
    print(" ".join(str(x) for x in cmd))

    try:
        if stdout_file is not None:
            with open(stdout_file, "w") as f:
                result = subprocess.run(
                    cmd,
                    stdout=f
                )
        else:
            result = subprocess.run(cmd)

    except FileNotFoundError:
        print(f"\nERROR: Program not found: {cmd[0]}")
        sys.exit(1)

    if result.returncode != 0:
        print(f"\nERROR: {description} failed.")
        sys.exit(result.returncode)

    print(f"✓ {description} completed")


def check_program(program):
    result = subprocess.run(
        ["which", program],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print(f"ERROR: '{program}' was not found in PATH.")
        sys.exit(1)

    print(f"✓ {program}: {result.stdout.strip()}")


def check_file(path, description):
    path = Path(path)

    if not path.exists():
        print(f"\nERROR: {description} does not exist:")
        print(f"  {path}")
        sys.exit(1)

    print(f"✓ {description}: {path}")


def run_python_script(script, arguments, description):
    command = [
        sys.executable,
        str(script)
    ] + [str(x) for x in arguments]

    run_command(command, description)


# ============================================================
# Main
# ============================================================

def run_feature_pipeline(
    assembly_path,
    illumina_r1_path,
    illumina_r2_path,
    ont_path,
    output_path,
    threads=4,
    window_size=5,
    keep_sam=False,
):

    # ========================================================
    # Resolve paths
    # ========================================================

    assembly = Path(assembly_path).resolve()
    illumina_r1 = Path(illumina_r1_path).resolve()
    illumina_r2 = Path(illumina_r2_path).resolve()
    ont = Path(ont_path).resolve()
    final_output = Path(output_path).resolve()

    output_dir = final_output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # Existing scripts
    # ========================================================

    scripts_dir = Path.home() / "Raleesa" / "scripts"
    utility_dir = Path(__file__).resolve().parent / "utility"

    illumina_script = utility_dir / "extract_illumina_features_with_softclip.py"
    ont_script = utility_dir / "extract_ont_features_with_softclip.py"
    sequence_script = scripts_dir / "extract_sequence_features_v2.py"
    assembly_script = scripts_dir / "extract_assembly_features.py"
    coverage_script = scripts_dir / "extract_coverage.py"
    combine_script = scripts_dir / "combine_per_base_features_v2.py"

    # ========================================================
    # Print configuration
    # ========================================================

    print_header("FEATURE EXTRACTION PIPELINE")

    print(f"Assembly:       {assembly}")
    # print(f"Reference:      {reference}")
    print(f"Illumina R1:    {illumina_r1}")
    print(f"Illumina R2:    {illumina_r2}")
    print(f"ONT:            {ont}")
    print(f"Output:         {final_output}")
    print(f"Threads:        {threads}")
    print(f"Window size:    {window_size}")

    # ========================================================
    # Check input files
    # ========================================================

    print_header("CHECKING INPUT FILES")

    check_file(assembly, "Rotated assembly")
    # check_file(reference, "Reference FASTA")
    check_file(illumina_r1, "Illumina R1")
    check_file(illumina_r2, "Illumina R2")
    check_file(ont, "ONT reads")

    # ========================================================
    # Check scripts
    # ========================================================

    print_header("CHECKING FEATURE SCRIPTS")

    check_file(
        illumina_script,
        "Illumina feature script"
    )

    check_file(
        ont_script,
        "ONT feature script"
    )

    check_file(
        sequence_script,
        "Sequence feature script"
    )

    check_file(
        assembly_script,
        "Assembly feature script"
    )

    check_file(
        coverage_script,
        "Coverage script"
    )

    check_file(
        combine_script,
        "Combine feature script"
    )

    # ========================================================
    # Check external programs
    # ========================================================

    print_header("CHECKING REQUIRED PROGRAMS")

    check_program("bwa")
    check_program("minimap2")
    check_program("samtools")

    # ========================================================
    # Define intermediate files
    # ========================================================

    illumina_sam = output_dir / "illumina.sam"
    illumina_bam = output_dir / "illumina.bam"
    illumina_bai = output_dir / "illumina.bam.bai"

    ont_sam = output_dir / "ont.sam"
    ont_bam = output_dir / "ont.bam"
    ont_bai = output_dir / "ont.bam.bai"

    illumina_features = output_dir / "illumina_features.csv"
    ont_features = output_dir / "ont_features.csv"
    sequence_features = output_dir / "sequence_features.csv"
    assembly_features = output_dir / "assembly_features.csv"
    coverage_features = output_dir / "coverage.csv"

    # ========================================================
    # STEP 1
    # BWA index
    # ========================================================

    bwa_index = Path(str(assembly) + ".bwt")

    if bwa_index.exists():
        print("\n✓ BWA index already exists.")
    else:
        run_command(
            [
                "bwa",
                "index",
                str(assembly)
            ],
            "Creating BWA index"
        )

    # ========================================================
    # STEP 2
    # Illumina alignment
    # ========================================================

    if illumina_bam.exists() and illumina_bai.exists():

        print("\n✓ Illumina BAM already exists.")
        print(f"  {illumina_bam}")

    else:

        # BWA MEM → SAM
        run_command(
            [
                "bwa",
                "mem",
                "-t",
                str(threads),
                str(assembly),
                str(illumina_r1),
                str(illumina_r2)
            ],
            "Aligning Illumina reads with BWA-MEM",
            stdout_file=illumina_sam
        )

        # SAM → sorted BAM
        run_command(
            [
                "samtools",
                "sort",
                "-@",
                str(threads),
                "-o",
                str(illumina_bam),
                str(illumina_sam)
            ],
            "Sorting Illumina BAM"
        )

        # BAM index
        run_command(
            [
                "samtools",
                "index",
                str(illumina_bam)
            ],
            "Indexing Illumina BAM"
        )

    # ========================================================
    # STEP 3
    # Illumina alignment statistics
    # ========================================================

    run_command(
        [
            "samtools",
            "flagstat",
            str(illumina_bam)
        ],
        "Checking Illumina alignment statistics"
    )

    # ========================================================
    # STEP 4
    # ONT alignment
    # ========================================================

    if ont_bam.exists() and ont_bai.exists():

        print("\n✓ ONT BAM already exists.")
        print(f"  {ont_bam}")

    else:

        # minimap2 → SAM
        run_command(
            [
                "minimap2",
                "-ax",
                "map-ont",
                "--MD",
                "-t",
                str(threads),
                str(assembly),
                str(ont)
            ],
            "Aligning ONT reads with minimap2",
            stdout_file=ont_sam
        )

        # SAM → sorted BAM
        run_command(
            [
                "samtools",
                "sort",
                "-@",
                str(threads),
                "-o",
                str(ont_bam),
                str(ont_sam)
            ],
            "Sorting ONT BAM"
        )

        # BAM index
        run_command(
            [
                "samtools",
                "index",
                str(ont_bam)
            ],
            "Indexing ONT BAM"
        )

    # ========================================================
    # STEP 5
    # ONT alignment statistics
    # ========================================================

    run_command(
        [
            "samtools",
            "flagstat",
            str(ont_bam)
        ],
        "Checking ONT alignment statistics"
    )

    # ========================================================
    # STEP 6
    # Illumina features
    # ========================================================

    if illumina_features.exists():

        print("\n✓ Illumina feature CSV already exists.")

    else:

        run_python_script(
            illumina_script,
            [
                "--bam",
                illumina_bam,
                "--output",
                illumina_features
            ],
            "Extracting Illumina features"
        )

    # ========================================================
    # STEP 7
    # ONT features
    # ========================================================

    if ont_features.exists():

        print("\n✓ ONT feature CSV already exists.")

    else:

        run_python_script(
            ont_script,
            [
                "--bam",
                ont_bam,
                "--output",
                ont_features
            ],
            "Extracting ONT features"
        )

    # ========================================================
    # STEP 8
    # Sequence features
    # ========================================================

    if sequence_features.exists():

        print("\n✓ Sequence feature CSV already exists.")

    else:

        run_python_script(
            sequence_script,
            [
                "--fasta",
                assembly,
                "--output",
                sequence_features,
                "--window-size",
                window_size
            ],
            "Extracting sequence features"
        )

    # ========================================================
    # STEP 9
    # Assembly features
    # ========================================================

    if assembly_features.exists():

        print("\n✓ Assembly feature CSV already exists.")

    else:

        run_python_script(
            assembly_script,
            [
                "--fasta",
                assembly,
                "--output",
                assembly_features
            ],
            "Extracting assembly features"
        )

    # ========================================================
    # STEP 10
    # Coverage
    # ========================================================

    if coverage_features.exists():

        print("\n✓ Coverage CSV already exists.")

    else:

        run_python_script(
            coverage_script,
            [
                "--bam",
                illumina_bam,
                "--output",
                coverage_features
            ],
            "Extracting coverage"
        )

    # ========================================================
    # STEP 11
    # Combine features
    # ========================================================

    run_python_script(
        combine_script,
        [
            "--assembly",
            assembly_features,
            "--illumina",
            illumina_features,
            "--ont",
            ont_features,
            "--sequence",
            sequence_features,
            "--output",
            final_output
        ],
        "Combining all per-base features"
    )

    # ========================================================
    # STEP 12
    # Verify output
    # ========================================================

    print_header("PIPELINE COMPLETED")

    if not final_output.exists():

        print("ERROR: Final feature CSV was not created.")
        sys.exit(1)

    print(f"✓ FINAL FEATURE CSV:")
    print(f"  {final_output}")

    # --------------------------------------------------------
    # Show line count
    # --------------------------------------------------------

    try:

        with open(final_output, "r") as f:
            line_count = sum(1 for _ in f)

        print(f"\nRows including header: {line_count}")
        print(f"Data rows:             {line_count - 1}")

    except Exception as e:

        print(f"Could not count rows: {e}")

    # ========================================================
    # Optional SAM cleanup
    # ========================================================

    if not keep_sam:

        for sam_file in [illumina_sam, ont_sam]:

            if sam_file.exists():

                print(f"\nRemoving intermediate SAM:")
                print(f"  {sam_file}")

                sam_file.unlink()

    # ========================================================
    # Summary
    # ========================================================

    print_header("OUTPUT FILES")

    files = [
        illumina_bam,
        illumina_bai,
        ont_bam,
        ont_bai,
        illumina_features,
        ont_features,
        sequence_features,
        assembly_features,
        coverage_features,
        final_output
    ]

    for file in files:

        if file.exists():

            print(f"✓ {file}")

    print("\n✓ COMPLETE FEATURE PIPELINE FINISHED.")
    print(f"\nFinal file:\n{final_output}")


if __name__ == "__main__":
    main()
