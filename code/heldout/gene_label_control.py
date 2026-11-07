"""
Gene-label specificity control for the fibroblast held-out test (DESCRIPTIVE; written 2026-10-21, after the primary
result existed, so it cannot be confirmatory).  Builds a direction for every comparison gene exactly the way the UFM1
direction was built, then asks where the case's statistic under UFM1 falls among the case's statistics under those
directions.

Construction copied from build_ufm1_consensus_direction.py and fix_rna_scale_mismatch.py:
  direction_G  = mean over G's trt_sh.cgs consensus signatures (LINCS Level 5 COMPZ.MODZ) -> (12328,)
  aligned_G    = direction_G read in the order of the map's overlap_ensembl_genes (by gene identity)
  guidance_G   = coef @ aligned_G                      (coef from lincs_to_rna_embedding_map.npz; no intercept)
Comparison set (fixed here in advance): every gene other than UFM1 with trt_sh.cgs signatures in >= --min-cell-lines
distinct cell lines (default 9, UFM1's own count).  Nothing is chosen by looking at scores.

Stages:
  count  prints how many genes qualify (reads sig_info only)
  build  extracts signatures from the .gctx, writes guidance_by_gene.npz, and checks that UFM1's rebuilt guidance equals
         the saved ufm1_rna_embedding_guidance.npz (cosine must exceed 0.9999) -- no cohort file is read
  score  requires the primary result file; reads the cohort embeddings (unit-normalized like the primary), uses the CORE
         of the primary script (SHA checked), writes control_result.json
"""
import argparse, gzip, hashlib, json, os, sys, time
import numpy as np

D = "/data/pduttapramanik/raresynth/"
LIN = D + "data/raw/mechanism/lincs/"
PPN = D + "data/raw/mechanism/ppn_extracted/"
GCTX = LIN + "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
SIG_INFO = LIN + "GSE92742_Broad_LINCS_sig_info.txt.gz"
GENE_INFO = LIN + "GSE92742_Broad_LINCS_gene_info.txt.gz"
MAP = PPN + "lincs_to_rna_embedding_map.npz"
SYM2ENS = PPN + "lincs_measured_gene_symbol_to_ensembl.json"
UFM1_SAVED = PPN + "ufm1_rna_embedding_guidance.npz"
OUT_DIR = D + "runs/pfib_test/"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def qualifying_genes(sig_info, min_cell_lines, exclude=("UFM1",)):
    s = sig_info[sig_info["pert_type"] == "trt_sh.cgs"]
    g = s.groupby("pert_iname").agg(n_sig=("sig_id", "size"), n_cell=("cell_id", "nunique"))
    g = g[(g.n_cell >= min_cell_lines) & (~g.index.isin(exclude))]
    return g.sort_index()


def row_index_for_overlap(measured_ids, pr_to_symbol, symbol_to_ensembl, overlap_ensembl):
    ens_of_row = {}
    for i, pr in enumerate(measured_ids):
        sym = pr_to_symbol.get(str(pr))
        ens = symbol_to_ensembl.get(sym) if sym else None
        if ens is not None:
            ens_of_row[ens] = i
    idx = np.array([ens_of_row.get(e, -1) for e in overlap_ensembl])
    if (idx < 0).any():
        raise RuntimeError(f"{int((idx < 0).sum())} overlap genes have no LINCS row")
    return idx


def guidance_from_direction(direction, row_idx, coef):
    return coef @ direction[row_idx].astype(np.float64)


def score_directions(E, G, case_index, loo_cos, perm_p, rank_of):
    """E: unit rows (N,768); G: (nG,768) guidance; returns per-gene case statistic, rank, p."""
    out = []
    for g in G:
        gh = g / (np.linalg.norm(g) + 1e-12)
        s = loo_cos(E, gh)
        out.append((float(s[case_index]), rank_of(s, case_index), perm_p(s, [case_index])))
    return out


def selftest():
    rng = np.random.default_rng(0)
    n_rows, n_over = 80, 50
    coef = rng.normal(size=(768, n_over))
    measured = [str(1000 + i) for i in range(n_rows)]
    pr2sym = {m: f"S{i}" for i, m in enumerate(measured)}
    perm = rng.permutation(n_rows)[:n_over]
    sym2ens = {f"S{i}": f"ENSG{i:05d}" for i in range(n_rows)}
    overlap = [sym2ens[f"S{i}"] for i in perm]
    idx = row_index_for_overlap(measured, pr2sym, sym2ens, overlap)
    assert (idx == perm).all()
    direction = rng.normal(size=n_rows)
    g = guidance_from_direction(direction, idx, coef)
    assert np.allclose(g, coef @ direction[perm])
    try:
        row_index_for_overlap(measured, pr2sym, sym2ens, overlap + ["ENSG99999"]); raise SystemExit("missing gene not refused")
    except RuntimeError:
        pass
    # scoring on the real CORE: a case pushed along gene 3's guidance ranks first for gene 3 and not for most others
    here = os.path.dirname(os.path.abspath(__file__))
    cand = [os.path.join(here, "../../rp/amend/pfib_primary_test_v2.py"), os.path.join(here, "pfib_primary_test_v2.py")]
    path = next((p for p in cand if os.path.exists(p)), None)
    if path is None:
        sys.exit("selftest needs pfib_primary_test_v2.py beside this file")
    core = {"np": np}
    exec(open(path).read().split("# --- BEGIN CORE ---")[1].split("# --- END CORE ---")[0], core)
    G = rng.normal(size=(6, 768)); E = core["unit"](rng.normal(size=(120, 768)))
    E[7] = core["unit"](E[7] + 0.5 * G[3] / np.linalg.norm(G[3]))
    res = score_directions(E, G, 7, core["loo_cos"], core["perm_p"], core["rank_of"])
    assert res[3][1] == 1 and res[3][2] < 0.05, res[3]
    assert sum(r[1] == 1 for r in res) == 1
    print("selftest OK (alignment, refusal)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["count", "build", "score", "selftest"])
    ap.add_argument("--min-cell-lines", type=int, default=9)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--primary-script", default=D + "pfib_primary_test_v2.py")
    ap.add_argument("--primary-sha", default="")
    ap.add_argument("--primary-result", default=OUT_DIR + "pfib_primary_result.json")
    ap.add_argument("--cohort", default=D + "data/embeddings/rna_pfib423_full.npz")
    ap.add_argument("--ufm1-id", default="R20754")
    ap.add_argument("--directions", default=OUT_DIR + "guidance_by_gene.npz")
    ap.add_argument("--out", default=OUT_DIR + "gene_label_control_result.json")
    a = ap.parse_args()
    if a.stage == "selftest":
        return selftest()
    import pandas as pd
    if a.stage in ("count", "build"):
        with gzip.open(SIG_INFO, "rt") as f:
            sig_info = pd.read_csv(f, sep="\t")
        genes = qualifying_genes(sig_info, a.min_cell_lines)
        print(f"comparison genes with trt_sh.cgs in >= {a.min_cell_lines} cell lines (UFM1 excluded): {len(genes)}")
        print(genes.n_cell.describe().round(1).to_string())
        u = sig_info[(sig_info.pert_type == "trt_sh.cgs") & (sig_info.pert_iname == "UFM1")]
        print(f"UFM1: {len(u)} signatures, {u.cell_id.nunique()} cell lines")
        if a.stage == "count":
            return
        from cmapPy.pandasGEXpress.parse import parse
        m = np.load(MAP, allow_pickle=True)
        coef, overlap = m["coef"].astype(np.float64), [str(x) for x in m["overlap_ensembl_genes"]]
        sym2ens = json.load(open(SYM2ENS))
        with gzip.open(GENE_INFO, "rt") as f:
            gi = pd.read_csv(f, sep="\t")
        pr2sym = dict(zip(gi["pr_gene_id"].astype(str), gi["pr_gene_symbol"]))
        by_gene = {gname: sig_info[(sig_info.pert_type == "trt_sh.cgs") & (sig_info.pert_iname == gname)].sig_id.tolist()
                   for gname in list(genes.index) + ["UFM1"]}
        names = list(by_gene)
        row_idx, measured_ref, G, nsig = None, None, [], []
        t0 = time.time()
        for b in range(0, len(names), a.batch):
            chunk = names[b:b + a.batch]
            cids = [s for g in chunk for s in by_gene[g]]
            df = parse(GCTX, cid=cids).data_df
            measured = [str(x) for x in df.index]
            if measured_ref is None:
                measured_ref = measured
                row_idx = row_index_for_overlap(measured, pr2sym, sym2ens, overlap)
            elif measured != measured_ref:
                sys.exit("REFUSING: row order differs between batches.")
            for g in chunk:
                cols = df[by_gene[g]].values.astype(np.float64)       # explicit by sig_id, never positional
                G.append(guidance_from_direction(cols.mean(axis=1), row_idx, coef)); nsig.append(cols.shape[1])
            print(f"  {min(b + a.batch, len(names))}/{len(names)} genes  ({time.time() - t0:.0f}s)", flush=True)
        G = np.array(G)
        u_saved = np.load(UFM1_SAVED)["rna_embedding_guidance"].astype(np.float64)
        u_new = G[names.index("UFM1")]
        cosu = float(u_saved @ u_new / (np.linalg.norm(u_saved) * np.linalg.norm(u_new)))
        print(f"UFM1 rebuilt vs saved guidance: cosine {cosu:.6f}, norm {np.linalg.norm(u_new):.4f} vs {np.linalg.norm(u_saved):.4f}")
        if cosu < 0.9999:
            sys.exit("REFUSING to save: the rebuild does not reproduce the locked UFM1 direction.")
        os.makedirs(OUT_DIR, exist_ok=True)
        np.savez_compressed(a.directions, guidance=G, genes=np.array(names), n_sigs=np.array(nsig),
                            ufm1_cosine_to_saved=cosu, min_cell_lines=a.min_cell_lines)
        print("saved ->", a.directions)
        return
    # ---- score ----
    if not os.path.exists(a.primary_result):
        sys.exit("REFUSING: the primary result does not exist.")
    if os.path.exists(a.out):
        sys.exit(f"REFUSING: {a.out} exists.")
    if sha(a.primary_script) != a.primary_sha:
        sys.exit("REFUSING: primary script hash differs from --primary-sha.")
    core = {"np": np}
    exec(open(a.primary_script).read().split("# --- BEGIN CORE ---")[1].split("# --- END CORE ---")[0], core)
    z = np.load(a.cohort, allow_pickle=True)
    ids = [str(c) for c in z["case_ids"]]
    E = core["unit"](z["embeddings"].astype(np.float64))
    if not np.isfinite(E).all():
        sys.exit("REFUSING: non-finite cohort rows.")
    d = np.load(a.directions, allow_pickle=True)
    G, genes = d["guidance"], [str(x) for x in d["genes"]]
    case = ids.index(a.ufm1_id)
    res = score_directions(E, G, case, core["loo_cos"], core["perm_p"], core["rank_of"])
    st = np.array([r[0] for r in res]); ui = genes.index("UFM1")
    other = np.delete(st, ui)
    out = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "n_comparison_genes": int(len(other)), "min_cell_lines": int(d["min_cell_lines"]),
           "ufm1": {"statistic": float(st[ui]), "rank_among_cases": res[ui][1], "p": res[ui][2]},
           "fraction_of_genes_with_statistic_ge_ufm1": float((other >= st[ui]).mean()),
           "comparison_statistic_quantiles": {q: float(np.quantile(other, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
           "comparison_p_below_0.05": int(sum(r[2] <= 0.05 for i, r in enumerate(res) if i != ui)),
           "script_sha256": sha(__file__), "primary_script_sha256": a.primary_sha, "descriptive_only": True}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tmp = a.out + ".tmp"; json.dump(out, open(tmp, "w"), indent=2); os.replace(tmp, a.out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
