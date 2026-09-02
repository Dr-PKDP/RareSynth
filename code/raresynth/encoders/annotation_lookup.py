"""
Query layer joining AlphaMissense/CADD/phyloP/GERP against our own observed
variants (from GDC MAF Chromosome/Start_Position/Reference_Allele/
Tumor_Seq_Allele2 columns), rather than processing whole-genome files.

Two access patterns, by necessity of how each source is distributed:

  - AlphaMissense: allele-specific (a missense prediction is a property of
    the specific substitution, not just the position), queried via a local
    tabix index built by annotation_sources.build_alphamissense_tabix_index.
  - CADD (bigwig track) / phyloP / GERP: position-only bigWig tracks,
    queried via pyBigWig. CADD's bigWig specifically is the MAXIMUM score
    across the 3 possible alternate alleles at a position, not the score for
    our specific observed allele -- an approximation, not exact, but a
    reasonable one given these features are aggregated to a per-gene MAXIMUM
    downstream regardless (see genomic.py's build_gene_annotation_matrix).

Everything in this file that touches pyBigWig or a live tabix index needs a
real server run to verify (no network, no pyBigWig, no real data files
available in the environment this was written in) -- only the pure
string/position-matching logic (AlphaMissense line parsing) is verified
here via synthetic data built from the documented file format.
"""

from __future__ import annotations

import subprocess

# AlphaMissense's exact header is not fully confirmed against a live download
# in this session -- verify the real first line with
# `zcat AlphaMissense_hg38.tsv.gz | head -5` before trusting this column
# order, per the standard column layout documented across VEP plugin usage
# and the Cheng et al. 2023 Science supplement.
ALPHAMISSENSE_COLUMNS = [
    "CHROM", "POS", "REF", "ALT", "genome", "uniprot_id",
    "transcript_id", "protein_variant", "am_pathogenicity", "am_class",
]


def parse_alphamissense_line(line):
    """One tabix-returned line -> dict, or None if it's a comment/header."""
    if not line or line.startswith("#"):
        return None
    parts = line.rstrip("\n").split("\t")
    if len(parts) != len(ALPHAMISSENSE_COLUMNS):
        return None  # malformed row -- caller should count/report these, not crash
    return dict(zip(ALPHAMISSENSE_COLUMNS, parts))


def query_alphamissense(am_path, chrom, pos, ref, alt, tabix_bin="tabix"):
    """Return am_pathogenicity (float) for one exact variant, or None if
    absent (e.g. not a missense variant -- AlphaMissense only covers
    missense substitutions, so a nonsense or synonymous variant correctly
    returns None here, not an error).

    chrom should match the file's own chromosome naming (confirm 'chr1' vs
    '1' against a live download before trusting this -- MAF files use
    'chr1'-style names per the real MAF snippet seen in this project,
    AlphaMissense's convention needs the same live check).

    Prefer batch_query_alphamissense for more than a handful of variants --
    this single-variant version spawns one tabix subprocess per call, which
    does not scale to a cohort's worth of variants.
    """
    region = f"{chrom}:{pos}-{pos}"
    result = subprocess.run([tabix_bin, str(am_path), region],
                            capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        row = parse_alphamissense_line(line)
        if row is None:
            continue
        if row["REF"] == ref and row["ALT"] == alt:
            return float(row["am_pathogenicity"])
    return None


def batch_query_alphamissense(am_path, variants, tabix_bin="tabix"):
    """Look up AlphaMissense scores for many variants in ONE tabix call
    instead of one call per variant.

    Calling tabix once per variant does not scale: a single sample's MAF
    can have dozens to low hundreds of variants, and a whole cohort run
    would spawn one subprocess per variant across thousands of samples.
    This builds one BED-style region file covering every UNIQUE (chrom,
    pos) in the input, queries once with `tabix -R`, then matches each
    requested (chrom, pos, ref, alt) against the returned candidate rows
    for its position (a position can have more than one alternate allele,
    so allele matching still happens after the batched fetch, not instead
    of it).

    variants: iterable of (chrom, pos, ref, alt) tuples
    Returns: dict {(chrom, pos, ref, alt): am_pathogenicity or None}
    """
    import tempfile

    variants = list(variants)
    if not variants:
        return {}

    unique_positions = sorted({(c, p) for c, p, r, a in variants})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as bed:
        for chrom, pos in unique_positions:
            bed.write(f"{chrom}\t{pos - 1}\t{pos}\n")  # BED is 0-based half-open
        bed_path = bed.name

    try:
        result = subprocess.run(
            [tabix_bin, "-R", bed_path, str(am_path)],
            capture_output=True, text=True, check=True,
        )
    finally:
        import os
        os.unlink(bed_path)

    # index all returned rows by position for fast per-variant allele matching
    by_position = {}
    for line in result.stdout.splitlines():
        row = parse_alphamissense_line(line)
        if row is None:
            continue
        key = (row["CHROM"], int(row["POS"]))
        by_position.setdefault(key, []).append(row)

    out = {}
    for chrom, pos, ref, alt in variants:
        candidates = by_position.get((chrom, pos), [])
        match = next((c for c in candidates
                     if c["REF"] == ref and c["ALT"] == alt), None)
        out[(chrom, pos, ref, alt)] = (
            float(match["am_pathogenicity"]) if match else None
        )
    return out


class BigWigLookup:
    """Thin wrapper over pyBigWig for position-only score queries (CADD-max,
    phyloP, GERP).

    Chromosome naming is NOT consistent across these three real files --
    confirmed live: CADD and phyloP both use "chr1"-style names (matching
    our MAF files), but GERP (sourced from Ensembl Compara, not UCSC) uses
    bare "1"-style names. Querying GERP with "chr1" silently returned None
    for a position where real data existed, with no error raised at all --
    exactly the kind of failure that would have made the whole gerp_max
    feature meaningless with no visible sign anything was wrong. This class
    now detects which convention a given file actually uses (checked once,
    from the file's own chroms() listing, not assumed) and normalizes the
    query accordingly, so callers never need to special-case any one source.
    """

    def __init__(self, bigwig_path):
        import pyBigWig  # deferred import: only needed where this runs live
        self.bw = pyBigWig.open(str(bigwig_path))
        chroms = set(self.bw.chroms().keys())
        # detect convention from the file itself rather than assuming --
        # check a few common chromosome names in both forms
        self._uses_chr_prefix = any(f"chr{n}" in chroms for n in ("1", "2", "X"))
        self._uses_bare = any(n in chroms for n in ("1", "2", "X"))
        if not self._uses_chr_prefix and not self._uses_bare:
            raise ValueError(
                f"{bigwig_path}: could not determine chromosome naming "
                f"convention from this file's chroms() listing "
                f"(sample: {list(chroms)[:5]}) -- inspect manually before "
                f"trusting any query against it"
            )

    def _normalize_chrom(self, chrom):
        has_prefix = chrom.startswith("chr")
        if has_prefix and not self._uses_chr_prefix:
            return chrom[3:]
        if not has_prefix and self._uses_chr_prefix:
            return f"chr{chrom}"
        return chrom

    def query(self, chrom, pos):
        """1-based position in, matching MAF convention; pyBigWig itself is
        0-based half-open, so this converts. Returns None if the position is
        outside any covered region (common at chromosome ends / gaps) rather
        than raising.
        """
        chrom = self._normalize_chrom(chrom)
        try:
            vals = self.bw.values(chrom, pos - 1, pos)
            v = vals[0] if vals else None
            return float(v) if v is not None and v == v else None  # v==v filters NaN
        except RuntimeError:
            # pyBigWig raises RuntimeError for an out-of-bounds/unknown chrom
            # rather than returning None -- caught and normalized here so
            # callers don't need pyBigWig-specific exception handling
            return None

    def close(self):
        self.bw.close()
