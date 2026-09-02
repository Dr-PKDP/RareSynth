"""
Parse GTEx v8 gene TPM + sample attributes into a {(gene_symbol, tissue):
median_tpm} lookup, for the expressed_in_tissue feature in
encoders/genomic.py::build_gene_annotation_matrix.

Confirmed real structure (not assumed) from the actual downloaded files:

  GTEx_Analysis_..._gene_tpm.gct.gz -- standard GCT format:
    line 1: "#1.2" (version)
    line 2: "<n_genes>\t<n_samples>"
    line 3: header -- "Name\tDescription\t<sample_id_1>\t<sample_id_2>..."
    line 4+: "<versioned_ensembl_id>\t<gene_symbol>\t<tpm_1>\t<tpm_2>..."

  GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt -- tab-separated,
    column 1 = SAMPID (matches the GCT's sample column headers directly,
    e.g. "GTEX-1117F-0226-SM-5GZZ7" -- no id transformation needed),
    column 6 = SMTS (general tissue, e.g. "Blood", "Lung") -- used here
    rather than the much finer-grained column 7 (SMTSD, e.g. "Brain -
    Cortex") since TCGA's own tissue-of-origin labels are coarse and would
    not map cleanly onto SMTSD's finer subdivisions.

build_gene_annotation_matrix keys its lookup by gene SYMBOL (from the
MAF's Hugo_Symbol column), so this uses the GCT's "Description" column
directly as the join key -- the versioned Ensembl "Name" column is not
needed for this purpose at all.
"""

from __future__ import annotations


def load_gtex_tissue_by_sample(sample_attributes_path):
    """Return {SAMPID: SMTS} for every GTEx sample."""
    import csv

    tissue_by_sample = {}
    with open(sample_attributes_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            sampid = row.get("SAMPID")
            tissue = row.get("SMTS")
            if sampid and tissue:
                tissue_by_sample[sampid] = tissue
    return tissue_by_sample


def build_gtex_median_tpm(gct_path, sample_attributes_path):
    """Return {(gene_symbol, tissue): median_tpm} across all GTEx v8
    samples of that tissue.

    Loaded with pandas for the large matrix (56,200 genes x 17,382
    samples); memory use is kept down with an explicit float32 dtype
    (roughly 3.9 GB for the numeric block rather than pandas' default
    float64's ~7.8 GB) -- the server has ample RAM either way, but no
    reason to be wasteful.
    """
    import pandas as pd

    tissue_by_sample = load_gtex_tissue_by_sample(sample_attributes_path)

    df = pd.read_csv(gct_path, sep="\t", skiprows=2, index_col=None)
    sample_cols = [c for c in df.columns if c not in ("Name", "Description")]
    for c in sample_cols:
        df[c] = df[c].astype("float32")

    matched_cols = [c for c in sample_cols if c in tissue_by_sample]
    unmatched = len(sample_cols) - len(matched_cols)
    if unmatched:
        print(f"  {unmatched}/{len(sample_cols)} GCT sample columns had no "
             f"matching entry in the sample attributes file (unexpected "
             f"if >0 -- verify the join is working as intended)")

    tissues = sorted(set(tissue_by_sample[c] for c in matched_cols))
    print(f"  {len(matched_cols)} samples matched across {len(tissues)} "
         f"tissue types")

    # duplicate gene symbols exist in GENCODE for a small number of genes
    # (paralogs/historical naming collisions) -- collapse by taking the
    # row-wise max across duplicates rather than silently keeping only
    # the first occurrence, so a gene's real expression signal is not
    # lost to an arbitrary tie-break
    if df["Description"].duplicated().any():
        n_dup = df["Description"].duplicated().sum()
        print(f"  {n_dup} duplicate gene symbol(s) in GTEx (paralogs/"
             f"naming collisions) -- collapsed via max across duplicates")
        df = df.groupby("Description")[sample_cols].max().reset_index()

    result = {}
    for tissue in tissues:
        cols = [c for c in matched_cols if tissue_by_sample[c] == tissue]
        medians = df[cols].median(axis=1)
        for gene, med in zip(df["Description"], medians):
            result[(gene, tissue)] = float(med)

    print(f"  built {len(result)} (gene, tissue) median TPM entries")
    return result
