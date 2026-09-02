"""
Download the four variant-annotation sources needed to fill the gene
annotation feature vector's full 12-dimension design (see PROGRESS.md for
why these were originally missing from the GDC MAF and how this closes
the gap rather than substituting).

Every URL here was confirmed live (fetched or found in current, dated
search results) rather than assumed -- see chat history for the
verification trail. The one URL flagged as unstable (GERP's
"current_compara" symlink) is pinned to an explicit release number instead.

    AlphaMissense_hg38.tsv.gz   ~1-3 GB   allele-specific, needs local tabix index
    CADD_GRCh38-v1.7.bw         a few GB  position-max bigWig (not allele-specific)
    hg38.phyloP100way.bw        ~9 GB     position bigWig (not allele-specific)
    gerp_conservation_scores...bw a few GB position bigWig (not allele-specific)

None of these are downloaded via the CADD/UCSC/Ensembl GENOME-WIDE SCORE
FILES (81GB+, not needed) -- only the compact bigWig tracks or, for
AlphaMissense, the one file that format is distributed as (still only
~1-3GB, not the 81GB scale of CADD's raw score file).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .sources import download

ALPHAMISSENSE_URL = "https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz"
CADD_BIGWIG_URL = "https://krishna.gs.washington.edu/download/CADD/bigWig/CADD_GRCh38-v1.7.bw"
PHYLOP_BIGWIG_URL = ("https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/"
                    "hg38.phyloP100way.bw")
# NOTE: two things confirmed wrong before landing on this URL, both by live
# 404s corrected against the real directory listing (not the README, which
# turned out to describe a filename that does not actually exist):
#   1. path is case-sensitive: /goldenPath/ (capital P), not /goldenpath/
#   2. actual filename is "hg38.phyloP100way.bw" -- NOT
#      "hg38.100way.phyloP100way.bw" as UCSC's own README.txt claims. The
#      real directory listing (with live clickable links, checked directly
#      rather than trusting the descriptive text) shows only
#      hg38.phyloP100way.bw (9.2G) actually exists at this path.
GERP_BIGWIG_URL = ("https://ftp.ensembl.org/pub/release-115/compara/conservation_scores/"
                   "92_mammals.gerp_conservation_score/"
                   "gerp_conservation_scores.homo_sapiens.GRCh38.bw")


def verify_bigwig_integrity(path, n_test_chroms=5):
    """Actually query a bigWig file after downloading it, not just check
    that it exists at a plausible size.

    This exists because of a real, serious failure found on the live
    server: a downloaded GERP bigWig had the correct file size, opened
    successfully with pyBigWig, and even reported a plausible-looking
    global coverage statistic (87.3% of the genome) -- but 260 of 275 real
    interval queries against actual exonic positions raised a hard
    bwGetOverlappingIntervalsCore RuntimeError, meaning the file's internal
    index was corrupted despite every size/existence-based check passing.
    A byte-count or even an MD5 check would NOT reliably have caught this
    specific failure mode (the file's total size was correct; the
    corruption was internal to the index structure, not a truncation).

    Queries several different chromosomes (not just one position) since a
    single successful query is not enough evidence -- the original
    corrupted file DID return a valid answer for the one position spot-
    checked by hand before the real scope of the problem was found.

    Returns True if all test queries succeed, False otherwise (with a
    printed reason -- never silently returns False with no explanation).
    """
    import pyBigWig

    try:
        bw = pyBigWig.open(str(path))
    except Exception as e:
        print(f"  integrity check FAILED: could not open {path}: {e}")
        return False

    chroms = list(bw.chroms().items())
    if not chroms:
        print(f"  integrity check FAILED: {path} opened but has no chromosomes")
        bw.close()
        return False

    test_chroms = chroms[:: max(1, len(chroms) // n_test_chroms)][:n_test_chroms]
    failures = []
    for chrom, length in test_chroms:
        pos = length // 2  # middle of the chromosome, not position 0 --
                            # some corruption patterns spare the file start
        try:
            bw.values(chrom, pos, pos + 1)
        except RuntimeError as e:
            failures.append((chrom, pos, str(e)))

    bw.close()
    if failures:
        print(f"  integrity check FAILED: {len(failures)}/{len(test_chroms)} "
             f"test queries raised an error, e.g. {failures[0]}")
        return False
    print(f"  integrity check passed ({len(test_chroms)} chromosomes queried)")
    return True


def download_bigwig_with_integrity_check(url, dest, max_attempts=2):
    """Download a bigWig file and verify it actually works before accepting
    it -- see verify_bigwig_integrity for why size/existence alone is not
    sufficient evidence a bigWig downloaded correctly.
    """
    dest = Path(dest)
    for attempt in range(1, max_attempts + 1):
        if not dest.exists():
            download(url, dest)
        if verify_bigwig_integrity(dest):
            return dest
        print(f"  attempt {attempt}/{max_attempts} failed integrity check")
        dest.unlink()
        if attempt == max_attempts:
            raise RuntimeError(
                f"{dest} failed the bigWig integrity check {max_attempts} "
                f"times in a row -- not transient, investigate the source "
                f"before retrying again"
            )
        print("  retrying with a fresh download...")
    return dest


def download_all(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}

    bigwig_sources = [
        ("cadd", "CADD_GRCh38-v1.7.bw", CADD_BIGWIG_URL),
        ("phylop", "hg38.phyloP100way.bw", PHYLOP_BIGWIG_URL),
        ("gerp", "gerp_conservation_scores.homo_sapiens.GRCh38.bw", GERP_BIGWIG_URL),
    ]

    print(f"downloading AlphaMissense_hg38.tsv.gz ...")
    am_dest = out / "AlphaMissense_hg38.tsv.gz"
    try:
        download(ALPHAMISSENSE_URL, am_dest)
        print(f"  -> {am_dest} ({am_dest.stat().st_size/1e6:.1f} MB)")
        results["alphamissense"] = am_dest
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        results["alphamissense"] = None

    for key, filename, url in bigwig_sources:
        print(f"\ndownloading {filename} ...")
        dest = out / filename
        try:
            download_bigwig_with_integrity_check(url, dest)
            print(f"  -> {dest} ({dest.stat().st_size/1e6:.1f} MB)")
            results[key] = dest
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            print(f"  This source failed but the others will still be "
                 f"attempted. Rerun this same command afterward -- already "
                 f"-downloaded (and integrity-verified) files are skipped "
                 f"automatically.")
            results[key] = None

    return results


def build_alphamissense_tabix_index(am_path, tabix_bin="tabix"):
    """AlphaMissense's file is bgzip-compressed but not pre-indexed --
    confirmed from a live report of the exact same file/index situation
    (biostars thread). Columns 1-2 are chrom/position (single-base, so
    -b 2 -e 2), matching the documented tabix command used by the
    Ensembl VEP AlphaMissense plugin against this same file.
    """
    am_path = Path(am_path)
    tbi_path = am_path.with_suffix(am_path.suffix + ".tbi")
    if tbi_path.exists():
        print(f"index already exists: {tbi_path}")
        return tbi_path
    cmd = [tabix_bin, "-s", "1", "-b", "2", "-e", "2", "-f", "-S", "1", str(am_path)]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"index built: {tbi_path}")
    return tbi_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-index", action="store_true")
    args = ap.parse_args()
    paths = download_all(args.out)
    if args.build_index:
        if paths.get("alphamissense") is None:
            print("\nskipping tabix index: AlphaMissense download did not "
                 "succeed, see above")
        else:
            build_alphamissense_tabix_index(paths["alphamissense"])
