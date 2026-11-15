"""帰無対照の関門の測定。版ごとに構造の判定を数える。

- dev / eval: v1.3 の事前登録（docs/PREREG_v1.3_NULL_GATES.md）。対照は実コーパスと同じシード 20 個
- dev14 / eval14: v1.4 の事前登録（docs/PREREG_v1.4_MULTIPLE_TESTING.md）。対照は 100 個
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from math import comb

from babel.decipher.engine import decipher
from babel.evaluate.score import STRUCT_KEYS, score
from babel.experiments.run import DEFAULT_GEN, DEV_SEEDS, TEST_SEEDS
from babel.generator.corpus import build_corpus, null_corpus
from babel.generator.languages import FAMILIES

VARIANTS = {"v1.2": {"null_gates": False}, "v1.3": {"affix_null_alpha": 0.05}, "v1.4": {}}
SPLITS = {   # (実コーパスのシード, 対照のシード, 比べる版)
    "dev": (DEV_SEEDS, DEV_SEEDS, ("v1.2", "v1.3")),
    "eval": (TEST_SEEDS, TEST_SEEDS, ("v1.2", "v1.3")),
    "dev14": (DEV_SEEDS, list(range(100)), ("v1.2", "v1.3", "v1.4")),
    "eval14": (list(range(300, 320)), list(range(300, 400)), ("v1.2", "v1.3", "v1.4")),
}


def _binom_cdf(k, n, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k, n, conf=0.95):
    """二項の割合の正確な信頼区間（二分法）。"""
    a = (1 - conf) / 2

    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return round((lo + hi) / 2, 4)

    lower = 0.0 if k == 0 else solve(lambda p: 1 - _binom_cdf(k - 1, n, p) < a)
    upper = 1.0 if k == n else solve(lambda p: _binom_cdf(k, n, p) > a)
    return [lower, upper]


def _real(task):
    fam, seed, level, bnd, names = task
    public, secret = build_corpus(fam, seed, level, boundaries=bnd, **DEFAULT_GEN)
    out = {}
    for name in names:
        cfg = VARIANTS[name]
        st = score(decipher(public, cfg), secret, public)["structure"]
        out[name] = st
    return {"kind": "real", "family": fam, "seed": seed, "level": level, "boundaries": bnd, "structure": out}


def _null(task):
    kind, seed, names = task
    public, _ = null_corpus(seed, kind)
    out = {}
    for name in names:
        cfg = VARIANTS[name]
        st = decipher(public, cfg)["structure"]
        out[name] = sorted(k for k in STRUCT_KEYS if (st.get(k) or {}).get("value") is not None)
    return {"kind": "null", "null": kind, "seed": seed, "claims": out}


def summarize(rows, names):
    s = {}
    for name in names:
        for bnd in (True, False):
            c = Counter()
            for r in rows:
                if r["kind"] == "real" and r["boundaries"] == bnd:
                    for k, v in r["structure"][name].items():
                        c[v] += 1
                        c[f"{k}:{v}"] += 1
            s[f"{name}:boundaries={bnd}"] = dict(sorted(c.items()))
        for kind in ("random", "shuffled", "syllable"):
            rs = [r for r in rows if r["kind"] == "null" and r["null"] == kind]
            n = sum(len(r["claims"][name]) for r in rs)
            hit = sum(1 for r in rs if r["claims"][name])
            s[f"{name}:null={kind}"] = {"runs": len(rs), "claims": n, "per_run": round(n / max(1, len(rs)), 3),
                                        "runs_with_claim": hit, "ci95_runs_with_claim": clopper_pearson(hit, len(rs)),
                                        "by_key": dict(Counter(k for r in rs for k in r["claims"][name]))}
    return s


def main(split="dev", jobs=4):
    seeds, null_seeds, names = SPLITS[split]
    t0 = time.time()
    real = [(f, s, lv, b, names) for f in FAMILIES for s in seeds for lv in "AC" for b in (True, False)]
    nulls = [(k, s, names) for k in ("random", "shuffled", "syllable") for s in null_seeds]
    with mp.Pool(jobs) as pool:
        rows = pool.map(_real, real) + pool.map(_null, nulls)
    out = {"split": split, "seeds": seeds, "null_seeds": null_seeds, "versions": {n: VARIANTS[n] for n in names},
           "seconds": round(time.time() - t0, 1), "summary": summarize(rows, names), "rows": rows}
    path = f"results/gates/gates-{split}.json"
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1))
    print("seconds", out["seconds"], "->", path)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["dev"]))
