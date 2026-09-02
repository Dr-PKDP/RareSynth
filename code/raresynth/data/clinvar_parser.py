"""
Parse ClinVar's VCF for the set of genes with at least one Pathogenic or
Likely_pathogenic variant.

Confirmed real structure (not assumed) from the actual downloaded file:

  INFO field, semicolon-separated key=value pairs. Two fields matter:
    CLNSIG=<value>     classification, can be compound: multiple
                       classifications joined by "|" (e.g. a variant with
                       both a clinical significance AND a risk_factor flag:
                       "Likely_pathogenic|risk_factor"), and ClinVar's own
                       "combined classification" values joined by "/"
                       (e.g. "Likely_pathogenic/Likely_pathogenic,_low_penetrance").
    GENEINFO=<value>   gene symbol:EntrezID, can list MULTIPLE overlapping
                       genes joined by "|" (e.g.
                       "SAMD11:148398|LOC107985728:107985728") -- a variant
                       affecting overlapping gene models should count
                       toward every gene listed, not just the first.

Filtering policy, confirmed against the real CLNSIG vocabulary rather than
assumed: a variant counts as P/LP if "pathogenic" appears in CLNSIG
(case-insensitive) AND "conflicting" does not. This simple two-condition
check was verified to correctly handle every real pattern found in the
actual file, including the one that looks like it should slip through:
"Conflicting_classifications_of_pathogenicity" literally contains the
substring "pathogenic" (as the first 10 characters of "pathogenicity"),
so a naive "contains pathogenic" check alone would wrongly include it --
the "conflicting" exclusion catches this specific case correctly. Also
verified correct on compound values like
"Likely_pathogenic/Likely_risk_allele" (included, since ClinVar's own
combined term leads with Likely_pathogenic) and
"Likely_pathogenic,_low_penetrance" (included, still a real P/LP call).
"""

from __future__ import annotations

import gzip


def _open_maybe_gz(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def _is_plp(clnsig):
    lowered = clnsig.lower()
    return "pathogenic" in lowered and "conflicting" not in lowered


def _parse_info_field(info, key):
    for part in info.split(";"):
        if part.startswith(key + "="):
            return part[len(key) + 1:]
    return None


def parse_clinvar_plp_genes(path):
    """Return a set of gene symbols with at least one Pathogenic or
    Likely_pathogenic variant in ClinVar.
    """
    genes = set()
    n_variants_seen = 0
    n_plp_variants = 0

    with _open_maybe_gz(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            n_variants_seen += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            info = fields[7]

            clnsig = _parse_info_field(info, "CLNSIG")
            if clnsig is None or not _is_plp(clnsig):
                continue
            n_plp_variants += 1

            geneinfo = _parse_info_field(info, "GENEINFO")
            if geneinfo is None:
                continue
            for gene_entry in geneinfo.split("|"):
                symbol = gene_entry.split(":")[0]
                if symbol:
                    genes.add(symbol)

    print(f"parsed {n_variants_seen} total variants, {n_plp_variants} "
         f"classified Pathogenic/Likely_pathogenic, "
         f"{len(genes)} unique genes with at least one such variant")
    return genes
