"""
Parse gnomAD v4.1 constraint metrics (gnomad.v4.1.constraint_metrics.tsv)
into a per-gene {pLI, LOEUF} dict.

Confirmed real structure (not assumed) from the actual downloaded file:

  - TRANSCRIPT-level, not gene-level: a real gene like A1BG has THREE rows
    -- one using a RefSeq-style id ("NM_130786.4") with gene_id literally
    the string "1" (not a real Ensembl gene id), one for the canonical/
    MANE-Select Ensembl transcript, one for a non-canonical transcript.
    The first row's gene_id does NOT start with "ENSG" -- filtered out on
    that basis rather than by the canonical/mane_select flags alone, since
    those flags do not reliably distinguish it from the real Ensembl row
    (both were "true" for A1BG's RefSeq-style row in the real file).
  - pLI is column 11, "lof_hc_lc.pLI" (loss-of-function high+low confidence
    combined) -- NOT column 19 "lof.pLI", which uses a stricter LoF set.
    "lof_hc_lc" is the modern standard pLI gnomAD itself reports.
  - LOEUF has NO column literally named "loeuf" -- it is column 23,
    "lof.oe_ci.upper" (the upper bound of the LoF observed/expected
    confidence interval), which is LOEUF's actual definition. Confirmed
    against the real header by position and by cross-checking the
    definition, not guessed from a plausible-sounding column name.

One row per gene is selected by preferring the MANE Select transcript
(the modern standard for "the" representative transcript per gene),
falling back to the canonical transcript if no MANE Select row exists for
that gene.
"""

from __future__ import annotations

import csv


def parse_gnomad_constraint(path):
    """Return {gene_symbol: {"pLI": float, "LOEUF": float}}, one row per
    gene, selecting MANE Select (falling back to canonical) among that
    gene's transcript-level rows.
    """
    candidates = {}  # gene -> list of (is_mane, is_canonical, pli, loeuf)

    with open(path) as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        required = ["gene", "gene_id", "canonical", "mane_select",
                   "lof_hc_lc.pLI", "lof.oe_ci.upper"]
        missing = [c for c in required if c not in idx]
        if missing:
            raise ValueError(
                f"{path}: expected columns not found: {missing} -- gnomAD "
                f"may have changed its constraint file schema, verify by "
                f"hand before trusting this parser"
            )

        for row in reader:
            if not row or len(row) <= max(idx.values()):
                continue
            gene = row[idx["gene"]]
            gene_id = row[idx["gene_id"]]
            if not gene_id.startswith("ENSG"):
                continue  # excludes the RefSeq-style summary row

            is_mane = row[idx["mane_select"]] == "true"
            is_canonical = row[idx["canonical"]] == "true"
            pli_str = row[idx["lof_hc_lc.pLI"]]
            loeuf_str = row[idx["lof.oe_ci.upper"]]
            if pli_str == "NA" or loeuf_str == "NA":
                continue

            try:
                pli, loeuf = float(pli_str), float(loeuf_str)
            except ValueError:
                continue

            candidates.setdefault(gene, []).append(
                (is_mane, is_canonical, pli, loeuf)
            )

    result = {}
    n_mane, n_canonical_fallback, n_arbitrary_fallback = 0, 0, 0
    for gene, rows in candidates.items():
        mane_rows = [r for r in rows if r[0]]
        canonical_rows = [r for r in rows if r[1]]
        if mane_rows:
            _, _, pli, loeuf = mane_rows[0]
            n_mane += 1
        elif canonical_rows:
            _, _, pli, loeuf = canonical_rows[0]
            n_canonical_fallback += 1
        else:
            _, _, pli, loeuf = rows[0]
            n_arbitrary_fallback += 1
        result[gene] = {"pLI": pli, "LOEUF": loeuf}

    print(f"parsed {len(result)} genes: {n_mane} via MANE Select, "
         f"{n_canonical_fallback} via canonical fallback, "
         f"{n_arbitrary_fallback} via arbitrary fallback (neither flag set)")
    return result
