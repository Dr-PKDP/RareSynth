"""
Data acquisition for the open-access-only configuration.

Everything here is downloadable without an application, a DUA, or an
institutional affiliation.  The UDN, 100,000 Genomes and UK Biobank resources
named in the original draft are deliberately absent: none of them can supply
five-modality rare-disease cases, and claiming them in a paper we cannot run
is the failure mode this rewrite exists to avoid.

Cohort roles
------------
train / internal-val   TCGA + GTEx
external validation 1  CPTAC          (independent institutions, independent
                                       processing, paired WSI + RNA + clinical)
external validation 2  Kremer et al.  (real Mendelian disease: skin fibroblast
                                       RNA-seq, 119 samples / 105 individuals,
                                       gene-level counts on Zenodo)
mechanism supervision  LINCS L1000, DepMap
priors                 ClinVar, gnomAD, OMIM/Orphanet, HPO, Open Targets

Storage estimate
----------------
    GTEx RNA-seq (v8 gene TPM)                      ~4 GB
    GTEx WSIs (~25k slides, SVS)                    ~6 TB  full
                                                    ~400 GB for a 3k-slide subset
    TCGA RNA-seq + MAF + clinical (GDC)             ~30 GB
    TCGA diagnostic WSIs (~11k)                     ~12 TB full, ~500 GB subset
    TCIA radiology (TCGA-matched collections)       ~250 GB
    CPTAC RNA + WSI                                 ~600 GB
    LINCS L1000 level 5                             ~50 GB
    Kremer counts (Zenodo 3887451)                  ~200 MB

Tile-embedding extraction is the disk bottleneck, not the model.  The
recommended pattern is stream-and-discard: download a slide, tile it, run UNI,
write the 1024-d slide vector, delete the SVS.  Peak disk then stays under
1 TB for the whole study.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import requests

GDC_API = "https://api.gdc.cancer.gov"
GTEX_PORTAL = "https://gtexportal.org/api/v2"
ZENODO_KREMER = "https://zenodo.org/records/3887451"


@dataclass
class Source:
    name: str
    role: str            # train | external_1 | external_2 | mechanism | prior
    modalities: tuple
    access: str          # open | open-registration
    note: str


REGISTRY = [
    Source("TCGA", "train", ("geno", "rna", "path", "rad", "ehr"), "open",
           "GDC open tier: RNA-seq STAR counts, somatic MAF, clinical XML, "
           "diagnostic slides. Germline VCFs are controlled and are NOT used; "
           "the genomic modality is built from the open somatic MAF plus "
           "population-level annotation."),
    Source("GTEx v8", "train", ("geno", "rna", "path"), "open",
           "Gene TPM + histology WSIs at 20x. 838 donors have both RNA-seq and "
           "matched histology, giving the paired normal baseline."),
    Source("TCIA", "train", ("rad",), "open",
           "TCGA-matched collections (GBM, LGG, BRCA, LUAD, KIRC, ...). "
           "Overlap with cases that also have RNA+WSI is ~1.5-2.5k, not 5k."),
    Source("CPTAC", "external_1", ("rna", "path", "ehr"), "open",
           "Independent multimodal validation cohort."),
    Source("Kremer 2017 (Zenodo 3887451)", "external_2", ("rna",), "open",
           "Mendelian/mitochondrial disease fibroblast RNA-seq, gene-level "
           "counts, GENCODE v34, with sample annotation. Tests the "
           "cancer-to-monogenic transfer claim on real rare disease."),
    Source("LINCS L1000", "mechanism", ("rna",), "open-registration",
           "GSE92742 / GSE70138 level-5 consensus signatures for PPN training."),
    Source("DepMap", "mechanism", ("rna",), "open",
           "CRISPR knockout + expression across ~1,100 cell lines."),
    Source("ClinVar / gnomAD v4 / OMIM / Orphanet / HPO", "prior", (), "open",
           "Gene-level pathogenicity, constraint, disease-gene and phenotype "
           "ontology inputs."),
    Source("AlphaMissense hg38", "prior", ("geno",), "open",
           "Genome Deepmind missense pathogenicity predictions, ~71M variants, "
           "GRCh38 coordinates matching TCGA's MAF build. Confirmed direct "
           "download, bgzip-compressed, tabix-indexable. "
           "https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz"),
    Source("CADD v1.7 GRCh38", "prior", ("geno",), "open",
           "Combined Annotation Dependent Depletion scores. Genome-wide "
           "score file is 81GB (never fully downloaded); a single bigWig "
           "track (max score per position across alleles) is used instead "
           "for a manageable, position-queryable local file. "
           "https://krishna.gs.washington.edu/download/CADD/bigWig/CADD_GRCh38-v1.7.bw"),
    Source("phyloP100way hg38", "prior", ("geno",), "open",
           "UCSC 100-vertebrate phyloP conservation scores, single bigWig. "
           "https://hgdownload.cse.ucsc.edu/goldenpath/hg38/phyloP100way/"
           "hg38.100way.phyloP100way.bw"),
    Source("GERP (Ensembl Compara) hg38", "prior", ("geno",), "open",
           "UCSC does not host GERP for hg38 (confirmed) -- Ensembl Compara's "
           "bigWig is the current real source. The 'current_compara' path is "
           "NOT a stable permalink (confirmed via a live GitHub issue where "
           "it silently changed release numbers) -- pin to an explicit "
           "release number and re-verify if this URL 404s. "
           "https://ftp.ensembl.org/pub/release-115/compara/conservation_scores/"
           "92_mammals.gerp_conservation_score/gerp_conservation_scores.homo_sapiens.GRCh38.bw"),
]


def print_registry():
    w = max(len(s.name) for s in REGISTRY)
    for s in REGISTRY:
        print(f"{s.name:<{w}}  {s.role:<11} {s.access:<18} {','.join(s.modalities)}")
        print(f"{'':<{w}}  {s.note}")


# --------------------------------------------------------------------------
# GDC
# --------------------------------------------------------------------------

def gdc_query_files(data_type: str, projects=None, extra_filters=None, size=20000):
    """Query the GDC file index for open-access files of a given type."""
    content = [
        {"op": "in", "content": {"field": "files.data_type", "value": [data_type]}},
        {"op": "in", "content": {"field": "files.access", "value": ["open"]}},
    ]
    if projects:
        content.append(
            {"op": "in", "content": {"field": "cases.project.project_id",
                                     "value": list(projects)}}
        )
    if extra_filters:
        content.extend(extra_filters)
    params = {
        "filters": json.dumps({"op": "and", "content": content}),
        "fields": ",".join([
            "file_id", "file_name", "file_size", "md5sum", "data_type",
            "data_format", "cases.submitter_id", "cases.case_id",
            "cases.project.project_id", "cases.samples.sample_type",
        ]),
        "format": "JSON",
        "size": str(size),
    }
    r = requests.get(f"{GDC_API}/files", params=params, timeout=180)
    r.raise_for_status()
    return r.json()["data"]["hits"]


def gdc_download(file_ids, out_dir, chunk=64):
    """Download open GDC files in batches via the bulk data endpoint."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = list(file_ids)
    for i in range(0, len(ids), chunk):
        batch = ids[i : i + chunk]
        r = requests.post(
            f"{GDC_API}/data",
            data=json.dumps({"ids": batch}),
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=3600,
        )
        r.raise_for_status()
        tgz = out / f"batch_{i//chunk:04d}.tar.gz"
        with open(tgz, "wb") as fh:
            for c in r.iter_content(1 << 20):
                fh.write(c)
        subprocess.run(["tar", "-xzf", str(tgz), "-C", str(out)], check=True)
        tgz.unlink()
        print(f"  batch {i//chunk} ({len(batch)} files) done", flush=True)


def build_tcga_manifest(projects=None, out_json="tcga_manifest.json"):
    """Find TCGA cases that have RNA-seq, a diagnostic slide, and clinical data.

    Returns a per-case record listing which modalities are available, which is
    what the availability mask in training consumes. Cases missing radiology
    are kept -- modality dropout is designed for exactly this.
    """
    rna = gdc_query_files("Gene Expression Quantification", projects)
    wsi = gdc_query_files("Slide Image", projects)
    maf = gdc_query_files("Masked Somatic Mutation", projects)

    cases = {}

    def add(hits, key):
        for h in hits:
            for c in h.get("cases", []):
                sid = c.get("submitter_id")
                if not sid:
                    continue
                rec = cases.setdefault(
                    sid,
                    {"submitter_id": sid,
                     "project": c.get("project", {}).get("project_id"),
                     "files": {}},
                )
                rec["files"].setdefault(key, []).append(
                    {"file_id": h["file_id"], "file_name": h["file_name"],
                     "size": h.get("file_size")}
                )

    add(rna, "rna")
    add(wsi, "path")
    add(maf, "geno")

    Path(out_json).write_text(json.dumps(cases, indent=2))
    n_full = sum(1 for v in cases.values() if {"rna", "path"} <= set(v["files"]))
    print(f"{len(cases)} cases indexed; {n_full} with both RNA-seq and a slide")
    return cases


# --------------------------------------------------------------------------
# GTEx
# --------------------------------------------------------------------------

GTEX_FILES = {
    "gene_tpm": ("https://storage.googleapis.com/adult-gtex/bulk-gex/v8/"
                 "rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_"
                 "gene_tpm.gct.gz"),
    "sample_attrs": ("https://storage.googleapis.com/adult-gtex/annotations/v8/"
                     "metadata-files/GTEx_Analysis_v8_Annotations_"
                     "SampleAttributesDS.txt"),
    "subject_phenotypes": ("https://storage.googleapis.com/adult-gtex/"
                           "annotations/v8/metadata-files/"
                           "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"),
}


def gtex_histology_url(tissue_sample_id: str) -> str:
    """Direct SVS endpoint for a GTEx histology slide.

    Slide IDs follow the GTEx tissue sample ID (e.g. GTEX-1117F-0526).
    Verify one URL by hand before launching a bulk pull; the portal has
    changed its image host in the past.
    """
    return f"https://brd.nci.nih.gov/brd/imagedownload/{tissue_sample_id}"


def download(url: str, dest: Path, expect_md5: str | None = None):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    with requests.get(url, stream=True, timeout=3600) as r:
        r.raise_for_status()
        h = hashlib.md5()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for c in r.iter_content(1 << 20):
                fh.write(c)
                h.update(c)
        if expect_md5 and h.hexdigest() != expect_md5:
            tmp.unlink()
            raise RuntimeError(f"md5 mismatch for {url}")
        tmp.rename(dest)
    return dest


if __name__ == "__main__":
    print_registry()
