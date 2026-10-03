"""Apply docs/ANALYSIS_PLAN.md to docs/results/compare/runs.jsonl (exact permutation, Holm, decide)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from doom_esn import credence_stats as cs

rows = [
    json.loads(line) for line in Path("docs/results/compare/runs.jsonl").read_text().splitlines()
]
by = defaultdict(dict)
for r in rows:
    by[r["arm"]][r["seed"]] = r["heldout_mean"]
for arm, d in by.items():
    cs.check_unique_seeds(list(d))
pts = {arm: np.array([d[s] for s in sorted(d)]) for arm, d in by.items()}
out = {
    "arms": {
        a: {
            "n": len(v),
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
            "median": float(np.median(v)),
            "min": float(v.min()),
            "max": float(v.max()),
            "points": [round(float(x), 1) for x in v],
        }
        for a, v in pts.items()
    }
}
plan = [
    ("reservoir_white", "reservoir_raw"),
    ("window2_white", "window1_white"),
    ("reservoir_white", "window2_white"),
]
ps = [cs.permutation_pvalue(pts[a], pts[b], alternative="two-sided") for a, b in plan]
padj = cs.holm(ps)
res = []
for (a, b), p, pa in zip(plan, ps, padj, strict=True):
    minp = float(cs.holm([cs.min_attainable_p(len(pts[a]), len(pts[b]))] * len(plan))[0])
    res.append(
        {
            "a": a,
            "b": b,
            "diff_means": float(pts[a].mean() - pts[b].mean()),
            "p_raw": float(p),
            "p_holm": float(pa),
            "min_p_holm_possible": minp,
            "decision": cs.decide(float(pa), 0.05, minp),
        }
    )
out["planned"] = res
rnd = [r for r in rows]
Path("docs/results/compare/analysis.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
