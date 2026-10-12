from pathlib import Path
import subprocess


class Aligner:
    """Align Illumina and ONT reads to one assembly."""

    def __init__(self, threads=4):
        self.threads = threads

    def ensure_bwa_index(self, assembly_fasta):
        """Create the BWA index when any required sidecar is missing."""
        assembly_fasta = Path(assembly_fasta)
        if not assembly_fasta.is_file():
            raise FileNotFoundError(f"Assembly FASTA not found: {assembly_fasta}")

        index_suffixes = ("amb", "ann", "bwt", "pac", "sa")
        if all(Path(f"{assembly_fasta}.{suffix}").is_file() for suffix in index_suffixes):
            return

        subprocess.run(["bwa", "index", str(assembly_fasta)], check=True)

    def align_fastq(
        self,
        assembly_fasta,
        illumina_r1,
        illumina_r2,
        ont_fastq,
        output_directory,
    ):
        """Align FASTQ reads and return sorted, indexed BAM paths."""
        assembly_fasta = Path(assembly_fasta)
        illumina_r1 = Path(illumina_r1)
        illumina_r2 = Path(illumina_r2)
        ont_fastq = Path(ont_fastq)
        for path, label in (
            (assembly_fasta, "Assembly FASTA"),
            (illumina_r1, "Illumina R1 FASTQ"),
            (illumina_r2, "Illumina R2 FASTQ"),
            (ont_fastq, "ONT FASTQ"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")

        self.ensure_bwa_index(assembly_fasta)

        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        illumina_sam = output_directory / "illumina.sam"
        illumina_bam = output_directory / "illumina.bam"
        ont_sam = output_directory / "ont.sam"
        ont_bam = output_directory / "ont.bam"
        for bam_path in (illumina_bam, ont_bam):
            if bam_path.is_symlink():
                bam_path.unlink()
            for index_path in (Path(f"{bam_path}.bai"), bam_path.with_suffix(".bai")):
                if index_path.is_symlink():
                    index_path.unlink()

        with illumina_sam.open("w") as sam_file:
            subprocess.run(
                [
                    "bwa",
                    "mem",
                    "-t",
                    str(self.threads),
                    str(assembly_fasta),
                    str(illumina_r1),
                    str(illumina_r2),
                ],
                check=True,
                stdout=sam_file,
            )
        subprocess.run(
            [
                "samtools",
                "sort",
                "-@",
                str(self.threads),
                "-o",
                str(illumina_bam),
                str(illumina_sam),
            ],
            check=True,
        )
        subprocess.run(["samtools", "index", str(illumina_bam)], check=True)

        with ont_sam.open("w") as sam_file:
            subprocess.run(
                [
                    "minimap2",
                    "-ax",
                    "map-ont",
                    "--MD",
                    "-t",
                    str(self.threads),
                    str(assembly_fasta),
                    str(ont_fastq),
                ],
                check=True,
                stdout=sam_file,
            )
        subprocess.run(
            [
                "samtools",
                "sort",
                "-@",
                str(self.threads),
                "-o",
                str(ont_bam),
                str(ont_sam),
            ],
            check=True,
        )
        subprocess.run(["samtools", "index", str(ont_bam)], check=True)
        return illumina_bam, ont_bam


def get_alignments(
    assembly_fasta, illumina_r1, illumina_r2, ont_fastq, output_directory, threads=4
):
    aligner = Aligner(threads=threads)
    illumina_bam, ont_bam = aligner.align_fastq(
        assembly_fasta, illumina_r1, illumina_r2, ont_fastq, output_directory
    )
    return illumina_bam, ont_bam
