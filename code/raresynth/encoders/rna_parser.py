"""
Parse GDC's STAR-counts RNA-seq TSV format into a clean gene -> TPM dict.

File shape (confirmed against a real downloaded file, not assumed):

    # gene-model: GENCODE v36
    gene_id\tgene_name\tgene_type\tunstranded\tstranded_first\tstranded_second\ttpm_unstranded\tfpkm_unstranded\tfpkm_uq_unstranded
    N_unmapped\t1119332\t1119332\t1119332
    N_multimapping\t4161493\t4161493\t4161493
    N_noFeature\t3363885\t28590771\t28547217
    N_ambiguous\t5128483\t1241035\t1244999
    ENSG00000000003.15\tTSPAN6\tprotein_coding\t3473\t1768\t1705\t54.5637\t16.3336\t16.0655
    ...

Three things a naive parser gets wrong here:
  1. Line 1 is a comment, not data.
  2. The four "N_*" summary rows have only 4 columns (id + 3 counts), not the
     9 columns real gene rows have -- a fixed-width parse breaks on these.
  3. gene_id carries a version suffix (".15") that will not match a gene
     index, gene embedding table, or Geneformer token map built from
     unversioned Ensembl IDs unless it is stripped first.

GDC's own guidance is to use tpm_unstranded for cross-project comparability
(the workflow is unstranded by default; stranded columns exist but are not
the primary output). We follow that here.
"""

from __future__ import annotations

import re

_ENSEMBL_VERSION = re.compile(r"\.\d+$")


def strip_ensembl_version(gene_id: str) -> str:
    return _ENSEMBL_VERSION.sub("", gene_id)


def parse_star_counts_tsv(path, value_col="tpm_unstranded"):
    """Return {unversioned_ensembl_id: value} for one GDC STAR-counts file.

    Rows starting with 'N_' (summary/QC rows, not genes) are dropped. Gene
    IDs are de-versioned. Duplicate unversioned IDs (rare, but PAR-linked
    genes on X/Y can collide after stripping the version) keep the first
    occurrence and are reported, not silently overwritten.
    """
    with open(path) as fh:
        lines = fh.readlines()

    # first non-comment line is the header
    start = 0
    while start < len(lines) and lines[start].startswith("#"):
        start += 1
    if start >= len(lines):
        raise ValueError(f"{path}: no header found (file is all comments/empty)")

    header = lines[start].rstrip("\n").split("\t")
    if value_col not in header:
        raise ValueError(
            f"{path}: column '{value_col}' not in header {header}. "
            f"GDC may have changed the STAR-counts schema -- verify by hand."
        )
    val_idx = header.index(value_col)
    gene_id_idx = header.index("gene_id")

    result = {}
    dupes = []
    for line in lines[start + 1:]:
        parts = line.rstrip("\n").split("\t")
        if not parts or not parts[0] or parts[0].startswith("N_"):
            continue
        if len(parts) <= val_idx:
            continue  # malformed row, shorter than expected -- skip, do not crash
        gid = strip_ensembl_version(parts[gene_id_idx])
        try:
            val = float(parts[val_idx])
        except ValueError:
            continue
        if gid in result:
            dupes.append(gid)
            continue
        result[gid] = val

    if dupes:
        import sys
        print(f"  [{path}] {len(dupes)} duplicate gene id(s) after "
             f"de-versioning, first occurrence kept: {dupes[:5]}"
             f"{' ...' if len(dupes) > 5 else ''}", file=sys.stderr)

    return result
