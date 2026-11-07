"""Parser for GDC STAR gene-count files."""

from __future__ import annotations

import re

_ENSEMBL_VERSION = re.compile(r"\.\d+$")


def strip_ensembl_version(gene_id: str) -> str:
    return _ENSEMBL_VERSION.sub("", gene_id)


def parse_star_counts_tsv(path, value_col="tpm_unstranded"):
    with open(path) as fh:
        lines = fh.readlines()

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
            continue
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
