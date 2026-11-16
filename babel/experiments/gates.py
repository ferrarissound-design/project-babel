"""帰無対照の関門の測定。版ごとに構造の判定を数える。

- dev / eval: v1.3 の事前登録（docs/PREREG_v1.3_NULL_GATES.md）。対照は実コーパスと同じシード 20 個
- dev18 / eval18: v1.8 の事前登録（docs/PREREG_v1.8_VERB_BINOMIAL.md）。対訳 5% と 25%、eval18 はシード 500〜519
- dev17 / eval17: v1.7 の事前登録（docs/PREREG_v1.7_VERB_RELATIVE.md）。dev17 は開発シード、eval17 はシード 400〜419
- dev16 / eval16: v1.6 の事前登録（docs/PREREG_v1.6_ALIGNMENT_NULL.md）。LEVEL A、通常の対訳と入れ替えた対訳
- dev15: v1.5 の事前登録（docs/PREREG_v1.5_VERB_EVIDENCE.md）の基準 C・D。水準 A・B・C 別の動詞の位置
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

VARIANTS = {"v1.2": {"null_gates": False, "v15_gates": False, "v16_gates": False},
            "v1.3": {"affix_null_alpha": 0.05, "v15_gates": False, "v16_gates": False},
            "v1.4": {"v15_gates": False, "v16_gates": False}, "v1.5": {"v16_gates": False},
            "v1.6": {"v17_gates": False}, "v1.7": {"v18_gates": False}, "v1.8": {}}
SPLITS = {   # (実コーパスのシード, 対照のシード, 比べる版, 実コーパスの水準)
    "dev": (DEV_SEEDS, DEV_SEEDS, ("v1.2", "v1.3"), "AC"),
    "eval": (TEST_SEEDS, TEST_SEEDS, ("v1.2", "v1.3"), "AC"),
    "dev14": (DEV_SEEDS, list(range(100)), ("v1.2", "v1.3", "v1.4"), "AC"),
    "eval14": (list(range(300, 320)), list(range(300, 400)), ("v1.2", "v1.3", "v1.4"), "AC"),
    "dev15": (DEV_SEEDS, DEV_SEEDS, ("v1.4", "v1.5"), "ABC"),     # v1.5 の基準 C・D（docs/PREREG_v1.5_VERB_EVIDENCE.md）
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
        for lv in sorted({r["level"] for r in rows if r["kind"] == "real"}):
            c = Counter(r["structure"][name]["verb_position"] for r in rows
                        if r["kind"] == "real" and r["boundaries"] and r["level"] == lv
                        and "verb_position" in r["structure"][name])
            s[f"{name}:level={lv}:verb_position"] = dict(sorted(c.items()))
        for kind in ("random", "shuffled", "syllable"):
            rs = [r for r in rows if r["kind"] == "null" and r["null"] == kind]
            n = sum(len(r["claims"][name]) for r in rs)
            hit = sum(1 for r in rs if r["claims"][name])
            s[f"{name}:null={kind}"] = {"runs": len(rs), "claims": n, "per_run": round(n / max(1, len(rs)), 3),
                                        "runs_with_claim": hit, "ci95_runs_with_claim": clopper_pearson(hit, len(rs)),
                                        "by_key": dict(Counter(k for r in rs for k in r["claims"][name]))}
    return s


def main(split="dev", jobs=4):
    seeds, null_seeds, names, levels = SPLITS[split]
    t0 = time.time()
    real = [(f, s, lv, b, names) for f in FAMILIES for s in seeds for lv in levels for b in (True, False)]
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


ORDER_KEYS = ("verb_position", "so_order", "adj_noun")


def _real16(task):
    """v1.6 の基準 A〜C: LEVEL A の合成言語を、通常の対訳と入れ替えた対訳で（docs/PREREG_v1.6_ALIGNMENT_NULL.md）。"""
    from babel.decipher.engine import Decipherer
    fam, seed, mismatch, names, frac = task
    public, secret = build_corpus(fam, seed, "A", **dict(DEFAULT_GEN, parallel_frac=frac))
    if mismatch:
        public = dict(public, parallel=Decipherer._permuted(public["parallel"], f"synthetic-mismatch-{fam}-{seed}"))
    out = {}
    for name in names:
        st = score(decipher(public, VARIANTS[name]), secret, public)["structure"]
        out[name] = {k: st.get(k) for k in ORDER_KEYS}
    return {"family": fam, "seed": seed, "mismatch": mismatch, "parallel_frac": frac, "structure": out}


def main16(split="dev16", jobs=4):
    seeds = (DEV_SEEDS if split.startswith("dev") else list(range(300, 320)) if split == "eval16"
             else list(range(400, 420)) if split == "eval17" else list(range(500, 520)))
    names = {"16": ("v1.5", "v1.6"), "17": ("v1.6", "v1.7"), "18": ("v1.7", "v1.8")}[split[-2:]]
    fracs = (0.05, 0.25) if split.endswith("18") else (0.05,)
    t0 = time.time()
    tasks = [(f, s, m, names, fr) for f in FAMILIES for s in seeds for m in (False, True) for fr in fracs]
    with mp.Pool(jobs) as pool:
        rows = pool.map(_real16, tasks)
    summary = {}
    for m, fr in [(m, fr) for m in (False, True) for fr in fracs]:
        rs = [r for r in rows if r["mismatch"] == m and r["parallel_frac"] == fr]
        tag = "" if len(fracs) == 1 else f":parallel={fr}"
        for name in names:
            c = Counter(v for r in rs for v in r["structure"][name].values() if v is not None)
            claimed = sum(1 for r in rs if any(v in ("correct", "wrong") for v in r["structure"][name].values()))
            summary[f"{name}:{'mismatch' if m else 'matched'}{tag}"] = {
                "runs": len(rs), "runs_with_claim": claimed, "counts": dict(c),
                "by_key": {k: dict(Counter(r["structure"][name][k] for r in rs if r["structure"][name][k])) for k in ORDER_KEYS}}
    out = {"split": split, "seeds": seeds, "seconds": round(time.time() - t0, 1), "summary": summary, "rows": rows}
    path = f"results/gates/gates-{split}.json"
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] and sys.argv[1][-2:] in ("16", "17", "18"):
        main16(sys.argv[1])
        sys.exit()
    main(*(sys.argv[1:2] or ["dev"]))
