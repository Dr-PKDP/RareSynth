"""
LINCS L1000 and DepMap acquisition -- mechanism-supervision data for PPN
training (see raresynth/model/ppn.py).

Both feed the same purpose: paired (gene perturbed, expression shift)
observations that teach the Perturbation Prediction Network what a given
gene's knockout/knockdown does to the transcriptome. LINCS gives this across
many cell lines and small-molecule/genetic perturbations; DepMap gives it
specifically for CRISPR knockouts paired with baseline expression across
~1,100 cancer cell lines.

LINCS L1000
-----------
Deposited in NCBI GEO, fully open, no account needed -- confirmed by
requesting a single file by hand and by cross-checking multiple independent
published pipelines that all wget the same paths directly. We use Level 5
(MODZ consensus signatures) only, not the much larger Level 3/4 raw data,
since Level 5 is what every downstream use case (including this one) is
built on and is roughly 20x smaller.

  Phase 1: GSE92742 (2017, will not be updated further)
  Phase 2: GSE70138 (updated through 2020, now static)

DepMap
------
Public releases are versioned quarterly (e.g. 24Q4, 25Q2) and served via
Figshare under the Broad Institute's "Broad_DepMap" author account. Rather
than hardcoding a release string that goes stale every three months, this
queries the Figshare API for that author's articles and picks the most
recent one whose title matches the public-release naming pattern.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import requests

from .sources import download  # reuse the resumable, checksummed downloader

USER_AGENT = "raresynth-research-script/1.0 (contact: pduttapramanik@uth.tmh.edu)"

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"

LINCS_FILES = {
    # Phase 1 -- GSE92742
    "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz":
        f"{GEO_BASE}/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz",
    "GSE92742_Broad_LINCS_sig_info.txt.gz":
        f"{GEO_BASE}/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_sig_info.txt.gz",
    "GSE92742_Broad_LINCS_gene_info.txt.gz":
        f"{GEO_BASE}/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_gene_info.txt.gz",
    "GSE92742_Broad_LINCS_cell_info.txt.gz":
        f"{GEO_BASE}/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_cell_info.txt.gz",
    "GSE92742_Broad_LINCS_pert_info.txt.gz":
        f"{GEO_BASE}/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_pert_info.txt.gz",
    # Phase 2 -- GSE70138 (filenames carry the 2017-03-06 deposition date;
    # this series stopped receiving updates in 2020, so this is the final
    # version despite the date looking old)
    "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx.gz":
        f"{GEO_BASE}/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx.gz",
    "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz":
        f"{GEO_BASE}/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz",
    "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz":
        f"{GEO_BASE}/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz",
    "GSE70138_Broad_LINCS_cell_info_2017-04-28.txt.gz":
        f"{GEO_BASE}/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_cell_info_2017-04-28.txt.gz",
}


def download_lincs(out_dir):
    """Download LINCS L1000 Level 5 data (~4-5 GB total, not the ~50 GB
    originally estimated -- that estimate assumed the raw Level 3/4 data,
    which we do not need)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, url in LINCS_FILES.items():
        dest = out / name
        if dest.exists():
            print(f"  {name}: already present, skipping")
            continue
        print(f"  downloading {name} ...", flush=True)
        try:
            download(url, dest)
            print(f"    -> {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            print(f"    (verify the URL still resolves: {url})")


# --------------------------------------------------------------------------
# DepMap via the Figshare API
# --------------------------------------------------------------------------

FIGSHARE_API = "https://api.figshare.com/v2"
BROAD_DEPMAP_AUTHOR_ID = 5514062

# Files needed for PPN training: CRISPR knockout effect (the "perturbation")
# paired with baseline expression (the "control"), plus cell-line metadata
# to match samples across the two files by DepMap ID.
DEPMAP_WANTED_FILES = [
    "CRISPRGeneEffect.csv",
    "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    "Model.csv",
]


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def find_latest_depmap_release(session=None, search_terms="DepMap Public"):
    """Find the most recent public DepMap quarterly release via Figshare.

    There is no public "list articles by author" endpoint on Figshare v2 --
    an earlier version of this function assumed /v2/authors/{id}/articles
    existed and it does not (confirmed by a live 404). The documented,
    no-auth-required way to find articles is POST /v2/articles/search.

    A text search alone risks a false positive (some unrelated article whose
    text happens to mention "DepMap"), so candidates are cross-checked
    against the full article record's author list for the Broad DepMap
    author id before being accepted.
    """
    s = session or _session()
    r = s.post(f"{FIGSHARE_API}/articles/search",
              json={"search_for": search_terms, "page_size": 50},
              timeout=60)
    r.raise_for_status()
    hits = r.json()

    quarter_pat = re.compile(r"\b\d{2}Q[1-4]\b")
    title_candidates = [
        h for h in hits
        if "public" in h.get("title", "").lower()
        and quarter_pat.search(h.get("title", ""))
    ]
    if not title_candidates:
        raise RuntimeError(
            "No article title matching the public quarterly DepMap release "
            "pattern (e.g. \"DepMap 25Q2 Public\") found via Figshare "
            "search. The naming convention may have changed -- check "
            "https://depmap.org/portal/download/all/ manually."
        )

    verified = []
    for h in title_candidates:
        detail_r = s.get(f"{FIGSHARE_API}/articles/{h['id']}", timeout=60)
        if detail_r.status_code != 200:
            continue
        detail = detail_r.json()
        authors = detail.get("authors", [])
        is_broad = any(
            a.get("id") == BROAD_DEPMAP_AUTHOR_ID
            or "broad" in (a.get("full_name") or "").lower()
            for a in authors
        )
        if is_broad:
            verified.append({
                "id": h["id"],
                "title": h["title"],
                "published_date": detail.get("published_date"),
            })

    if not verified:
        raise RuntimeError(
            f"Found {len(title_candidates)} title match(es) for the public "
            "release pattern but none were authored by the Broad DepMap "
            "account -- likely a false positive on title text alone. "
            "Verify manually at https://depmap.org/portal/download/all/ "
            f"and check candidate titles: {[c['title'] for c in title_candidates]}"
        )

    verified.sort(key=lambda a: a["published_date"] or "", reverse=True)
    latest = verified[0]
    return latest["id"], latest["title"], latest["published_date"]


def list_depmap_files(article_id, session=None):
    s = session or _session()
    r = s.get(f"{FIGSHARE_API}/articles/{article_id}", timeout=60)
    r.raise_for_status()
    files = r.json().get("files", [])
    return [{"name": f["name"], "size": f["size"], "url": f["download_url"]}
            for f in files]


def download_depmap(out_dir, wanted=None, session=None):
    """Discover the latest public release, then download only the files
    in ``wanted`` (default: DEPMAP_WANTED_FILES). Full releases include many
    files (mutation calls, copy number, drug screens, ...) most of which we
    do not need for PPN training.
    """
    s = session or _session()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wanted = wanted or DEPMAP_WANTED_FILES

    article_id, title, published = find_latest_depmap_release(s)
    print(f"Latest public DepMap release: \"{title}\" "
         f"(article {article_id}, published {published})")

    files = list_depmap_files(article_id, s)
    by_name = {f["name"]: f for f in files}
    print(f"{len(files)} files in this release; downloading {len(wanted)} "
         f"of them:")

    for name in wanted:
        if name not in by_name:
            print(f"  {name}: NOT FOUND in this release's file list. "
                 f"Available files: {sorted(by_name.keys())[:10]}"
                 f"{' ...' if len(by_name) > 10 else ''}")
            continue
        f = by_name[name]
        dest = out / name
        if dest.exists() and dest.stat().st_size == f["size"]:
            print(f"  {name}: already present ({f['size']/1e6:.1f} MB), skipping")
            continue
        print(f"  downloading {name} ({f['size']/1e6:.1f} MB) ...", flush=True)
        try:
            download(f["url"], dest)
            print(f"    -> {dest}")
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")

    return article_id, title


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("discover")  # just print the latest DepMap release, no download

    d = sub.add_parser("download")
    d.add_argument("--out", required=True)
    d.add_argument("--skip-lincs", action="store_true")
    d.add_argument("--skip-depmap", action="store_true")

    args = ap.parse_args()
    if args.cmd == "discover":
        aid, title, pub = find_latest_depmap_release()
        print(f"Latest public DepMap release: \"{title}\" "
             f"(article {aid}, published {pub})")
        for f in list_depmap_files(aid):
            print(f"  {f['name']:50s} {f['size']/1e6:8.1f} MB")
    elif args.cmd == "download":
        out = Path(args.out)
        if not args.skip_lincs:
            print("=== LINCS L1000 ===")
            download_lincs(out / "lincs")
        if not args.skip_depmap:
            print("\n=== DepMap ===")
            download_depmap(out / "depmap")


if __name__ == "__main__":
    main()
