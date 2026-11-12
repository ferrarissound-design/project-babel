"""転写の統計（記号数・クラス数・1回だけのクラス数）が、どんな「真の字形数」と「クラスの分けすぎ」に両立するか。

正解転写がない画像で、74クラスが過剰な分割なのか、本当に字形が多いのかを、統計だけで区別できるかを調べる。
真の字形 K 種（出現頻度は Zipf 分布、指数 s）から N 記号を引き、各出現を確率 p で「分けすぎ」
（その字形の新しい変種のクラス。変種は字形ごとに最大 6 個）にしたときの、クラス数と1回だけのクラス数の分布を求める。
"""
from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from pathlib import Path


def simulate(K, N, s=1.0, p_split=0.0, max_variants=6, reps=400, seed=0):
    rng = random.Random(f"inv-{K}-{N}-{s}-{p_split}-{seed}")
    w = [1 / (r + 1) ** s for r in range(K)]
    types, hapax = [], []
    for _ in range(reps):
        toks = rng.choices(range(K), w, k=N)
        cls = [(t, rng.randint(1, max_variants)) if rng.random() < p_split else (t, 0) for t in toks]
        c = Counter(cls)
        types.append(len(c))
        hapax.append(sum(1 for v in c.values() if v == 1))
    return types, hapax


def consistent(N, obs_types, obs_hapax, Ks=(10, 15, 20, 30, 40, 60, 80, 120, 200, 400), ss=(0.5, 1.0, 1.5),
               ps=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9), tol=0.05):
    """観測（クラス数・1回だけのクラス数）が、シミュレーションの中央 90% の範囲に入る組み合わせ。"""
    rows = []
    for K in Ks:
        for s in ss:
            for p in ps:
                t, h = simulate(K, N, s, p)
                t.sort()
                h.sort()
                lo, hi = int(len(t) * tol), int(len(t) * (1 - tol)) - 1
                ok = t[lo] <= obs_types <= t[hi] and h[lo] <= obs_hapax <= h[hi]
                rows.append({"true_glyphs": K, "zipf_s": s, "p_oversplit": p, "types_median": statistics.median(t),
                             "types_90": [t[lo], t[hi]], "hapax_median": statistics.median(h), "hapax_90": [h[lo], h[hi]],
                             "consistent": ok})
    return rows


def run(corpus="results/vision/real-img4494/corpus-continuous.json", out="results/vision/real-img4494/inventory.json"):
    d = json.loads(Path(corpus).read_text(encoding="utf-8"))
    toks = [ch for doc in d["documents"] for sent in doc["sentences"] for ch in sent if not ch.isspace()]
    c = Counter(toks)
    N, T, H = len(toks), len(c), sum(1 for v in c.values() if v == 1)
    rows = consistent(N, T, H)
    res = {"observed": {"symbols": N, "classes": T, "hapax_classes": H}, "grid": rows,
           "consistent": [r for r in rows if r["consistent"]],
           "note": "真の字形数・頻度分布・分けすぎの確率の組み合わせのうち、観測と両立するもの（中央 90% の範囲）。"
                   "複数の組み合わせが両立するなら、転写の統計だけでは過剰分割と字形の多さを区別できない"}
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return res


if __name__ == "__main__":
    r = run()
    print(r["observed"])
    for x in r["consistent"]:
        print(x["true_glyphs"], x["zipf_s"], x["p_oversplit"], x["types_90"], x["hapax_90"])
