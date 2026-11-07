"""Does the mapped UFM1 direction match the measured embedding shift after UFM1 knockdown?  (post hoc validation)"""
import json
import sys
import numpy as np

EMB, DIRS, UFM1 = sys.argv[1], sys.argv[2], sys.argv[3]
OUTJ = sys.argv[4] if len(sys.argv) > 4 else "perturbseq_validation.json"
unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)

z = np.load(EMB, allow_pickle=True)
ids = [str(c) for c in z["case_ids"]]
E = unit(z["embeddings"].astype(np.float64))
kind = np.array([i.split("|")[0] for i in ids]); gene = np.array([i.split("|")[1] for i in ids])
ctrl, pm = np.where(kind == "control")[0], np.where(kind == "pert")[0]
print("controls", len(ctrl), "perturbations", len(pm))

d = np.load(DIRS, allow_pickle=True)
G, genes = unit(d["guidance"].astype(np.float64)), [str(x) for x in d["genes"]]
u = unit(np.load(UFM1)["rna_embedding_guidance"].astype(np.float64))
if "UFM1" in genes:
    genes_o = [g for g in genes if g != "UFM1"]; G = G[[genes.index(g) for g in genes_o]]; genes = genes_o
D = np.vstack([u[None], G]); dnames = ["UFM1"] + genes

mu_c = E[ctrl].mean(0)
S_raw = E[pm] - mu_c
pn = list(gene[pm])
res = {"n_controls": int(len(ctrl)), "n_perturbations": int(len(pm)), "n_directions": int(len(dnames))}

def rank_frac(vals, i):
    return float((vals >= vals[i]).sum() / len(vals))

for label, S in (("raw_shift", S_raw), ("centered_shift", S_raw - S_raw.mean(0))):
    r = {}
    c_u = unit(S) @ u
    for g in ("UFM1", "UBA5", "UFL1"):
        if g in pn:
            i = pn.index(g)
            r[g] = {"cos_with_UFM1_direction": float(c_u[i]), "rank": int((c_u >= c_u[i]).sum()),
                    "rank_fraction_among_perturbations": rank_frac(c_u, i)}
    # null: control pseudobulks against the mean of the other controls
    Ec = E[ctrl]; null = np.array([(unit(Ec[k] - np.delete(Ec, k, 0).mean(0)) @ u) for k in range(len(ctrl))])
    r["control_null_cos_mean_sd"] = [float(null.mean()), float(null.std())]
    # direction specificity: for each shared gene, rank of its own direction among all directions
    C = unit(S) @ D.T
    shared = [g for g in pn if g in dnames]
    fr = []
    for g in shared:
        i, j = pn.index(g), dnames.index(g)
        fr.append(rank_frac(C[i], j))
    fr = np.array(fr)
    r["gene_matching"] = {"n_shared_genes": int(len(shared)), "median_rank_fraction": float(np.median(fr)),
                          "fraction_le_0.05": float((fr <= 0.05).mean()), "expected_under_chance": 0.05,
                          "UFM1_rank_fraction": float(fr[shared.index("UFM1")]) if "UFM1" in shared else None}
    res[label] = r
json.dump(res, open(OUTJ, "w"), indent=1)
print(json.dumps(res, indent=1))
