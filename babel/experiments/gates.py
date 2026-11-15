"""v1.3 帰無対照の関門の測定（docs/PREREG_v1.3_NULL_GATES.md）。関門あり・なしで構造の判定を数える。"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from collections import Counter

from babel.decipher.engine import decipher
from babel.evaluate.score import STRUCT_KEYS, score
from babel.experiments.run import DEFAULT_GEN, DEV_SEEDS, TEST_SEEDS
from babel.generator.corpus import build_corpus, null_corpus
from babel.generator.languages import FAMILIES

VARIANTS = {"v1.2": {"null_gates": False}, "v1.3": {"null_gates": True}}


def _real(task):
    fam, seed, level, bnd = task
    public, secret = build_corpus(fam, seed, level, boundaries=bnd, **DEFAULT_GEN)
    out = {}
    for name, cfg in VARIANTS.items():
        st = score(decipher(public, cfg), secret, public)["structure"]
        out[name] = st
    return {"kind": "real", "family": fam, "seed": seed, "level": level, "boundaries": bnd, "structure": out}


def _null(task):
    kind, seed = task
    public, _ = null_corpus(seed, kind)
    out = {}
    for name, cfg in VARIANTS.items():
        st = decipher(public, cfg)["structure"]
        out[name] = sorted(k for k in STRUCT_KEYS if (st.get(k) or {}).get("value") is not None)
    return {"kind": "null", "null": kind, "seed": seed, "claims": out}


def summarize(rows):
    s = {}
    for name in VARIANTS:
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
            s[f"{name}:null={kind}"] = {"runs": len(rs), "claims": n, "per_run": round(n / max(1, len(rs)), 3),
                                        "by_key": dict(Counter(k for r in rs for k in r["claims"][name]))}
    return s


def main(split="dev", jobs=4):
    seeds = DEV_SEEDS if split == "dev" else TEST_SEEDS
    t0 = time.time()
    real = [(f, s, lv, b) for f in FAMILIES for s in seeds for lv in "AC" for b in (True, False)]
    nulls = [(k, s) for k in ("random", "shuffled", "syllable") for s in seeds]
    with mp.Pool(jobs) as pool:
        rows = pool.map(_real, real) + pool.map(_null, nulls)
    out = {"split": split, "seeds": seeds, "seconds": round(time.time() - t0, 1), "summary": summarize(rows), "rows": rows}
    path = f"results/gates/gates-{split}.json"
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1))
    print("seconds", out["seconds"], "->", path)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["dev"]))
