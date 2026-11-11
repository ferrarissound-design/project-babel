"""解読不能性の検定。語順をシャッフルした対照と比較して、統語構造が存在するかを判定する。"""
from __future__ import annotations

import math
import random
from collections import Counter
from statistics import mean, pstdev


def _cond_entropy(sents, vocab):
    big, uni = Counter(), Counter()
    for s in sents:
        prev = "<s>"
        for w in s:
            w = w if w in vocab else "<rare>"
            big[(prev, w)] += 1
            uni[prev] += 1
            prev = w
    n = sum(big.values())
    return -sum(c / n * math.log2(c / uni[p]) for (p, _), c in big.items()) if n else 0.0


def syntax_test(sents, n_shuffle=6, seed=0):
    """語の並びに、無作為な並べ替えより強い規則性があるか。"""
    cnt = Counter(w for s in sents for w in s)
    vocab = {w for w, c in cnt.items() if c >= 5}
    real = _cond_entropy(sents, vocab)
    rng = random.Random(seed)
    pool = [w for s in sents for w in s]
    vals = []
    for _ in range(n_shuffle):
        rng.shuffle(pool)
        it = iter(pool)
        vals.append(_cond_entropy([[next(it) for _ in s] for s in sents], vocab))
    mu, sd = mean(vals), pstdev(vals) or 1e-9
    z = (mu - real) / sd
    reduction = (mu - real) / mu if mu else 0.0
    return {"entropy": round(real, 4), "shuffled_mean": round(mu, 4), "z": round(z, 2),
            "reduction": round(reduction, 4), "present": bool(z > 5 and reduction > 0.03)}


def morphology_test(share_affixed_tokens, n_affixes):
    """接辞付きと解析された語の割合が十分に高いか。"""
    return {"affixed_token_share": round(share_affixed_tokens, 4), "n_affixes": n_affixes,
            "present": bool(share_affixed_tokens >= 0.2 and n_affixes >= 3)}
