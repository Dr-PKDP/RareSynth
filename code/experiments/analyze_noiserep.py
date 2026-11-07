"""Analysis of the noise-draw replicates of the steering comparator."""
import json
import os
import sys
from collections import defaultdict

import numpy as np

MARGIN, NBOOT = 0.05, 10000


def classify(lo, hi):
    if hi <= -MARGIN:
        return "advantage beyond margin"
    if lo >= MARGIN:
        return "steered Gaussian closer"
    if lo > -MARGIN and hi < MARGIN:
        return "within margin (not equivalence)"
    return "inconclusive"


def boot(d_by_ckpt, resample_ckpt, rng):
    ck = sorted(d_by_ckpt)
    est = np.mean([np.mean(d_by_ckpt[c]) for c in ck])
    out = np.empty(NBOOT)
    for i in range(NBOOT):
        cks = rng.choice(ck, size=len(ck)) if resample_ckpt else ck
        out[i] = np.mean([np.mean(rng.choice(d_by_ckpt[c], size=len(d_by_ckpt[c]))) for c in cks])
    return est, float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def analyze(lines, verbose=True):
    un = {(r["seed"], r["mode"], r["rep"]): r for r in lines if r["scale"] == "unguided"}
    cells = defaultdict(lambda: defaultdict(list))
    for r in lines:
        if r["scale"] == "unguided":
            continue
        u = un.get((r["seed"], r["mode"], r["rep"]))
        for arm in ("T_translate", "G_ascent", "P_zascent"):
            if arm not in r:
                continue
            cells[(r["mode"], r["scale"], arm, "level")][r["seed"]].append(r["generator"]["c2st"] - r[arm]["c2st"])
            if u:
                inc = (r["generator"]["c2st"] - u["generator"]["c2st"]) - (r[arm]["c2st"] - u["gaussian_base"]["c2st"])
                cells[(r["mode"], r["scale"], arm, "increase")][r["seed"]].append(inc)
    rng = np.random.default_rng(0)
    rows = []
    for key in sorted(cells):
        d = {c: np.asarray(v) for c, v in cells[key].items()}
        est, lo, hi = boot(d, False, rng)
        _, lo2, hi2 = boot(d, True, rng) if len(d) > 1 else (est, lo, hi)
        n = {c: len(v) for c, v in d.items()}
        rows.append({"mode": key[0], "scale": key[1], "arm": key[2], "kind": key[3], "est": est, "lo": lo, "hi": hi,
                     "lo_ckpt": lo2, "hi_ckpt": hi2, "n_draws": n, "class": classify(lo, hi), "class_sens": classify(lo2, hi2)})
    if verbose:
        for part, title in ((lambda r: r["kind"] == "level" and r["arm"] != "P_zascent", "PRIMARY (levels; T and G)"),
                            (lambda r: r["kind"] == "level" and r["arm"] == "P_zascent", "EXPLORATORY: arm P (levels)"),
                            (lambda r: r["kind"] == "increase", "EXPLORATORY: increase contrast")):
            print(f"\n=== {title} ===")
            for r in filter(part, rows):
                print(f"{r['mode']:>4} {r['scale']:>3} {r['arm']:<12} D {r['est']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"
                      f"  (+ckpt resampling [{r['lo_ckpt']:+.3f}, {r['hi_ckpt']:+.3f}])  draws/ckpt {sorted(r['n_draws'].values())}"
                      f"  -> {r['class']}  | sensitivity: {r['class_sens']}")
    return rows


def selftest():
    rng = np.random.default_rng(1)
    lines = []
    for s in range(3):
        for r in range(10):
            for scale, true_gap in ((150, -0.08), (200, -0.02)):
                g = 0.80 + rng.normal(0, 0.03)
                lines.append({"seed": s, "mode": "cond", "rep": r, "scale": scale, "generator": {"c2st": g},
                              "T_translate": {"c2st": g - true_gap + rng.normal(0, 0.02)}})
            lines.append({"seed": s, "mode": "cond", "rep": r, "scale": "unguided", "generator": {"c2st": 0.6},
                          "gaussian_base": {"c2st": 0.5}})
    rows = {(x["scale"], x["kind"]): x for x in analyze(lines, verbose=False)}
    a, b = rows[(150, "level")], rows[(200, "level")]
    assert a["class"] == "advantage beyond margin" and abs(a["est"] + 0.08) < 0.03, a
    assert b["class"] == "within margin (not equivalence)" and abs(b["est"] + 0.02) < 0.03, b
    assert classify(-0.09, -0.04) == "inconclusive" and classify(0.06, 0.1) == "steered Gaussian closer"
    print("SELFTEST PASSED:", {k: (round(v["est"], 3), v["class"]) for k, v in rows.items() if k[1] == "level"})


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        p = next((a for a in sys.argv[1:] if not a.startswith("--")), None) or \
            "/data/pduttapramanik/raresynth/runs/steering_comparator/noiserep/draws.jsonl"
        analyze([json.loads(x) for x in open(p) if x.strip()])
