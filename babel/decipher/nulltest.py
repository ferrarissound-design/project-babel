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


def syntax_test_perm(sents, n_shuffle=199, seed=0, alpha=0.01):
    """v1.3（G2）: 置換 p 値で判定する統語検定。z は参考として残す。"""
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
    p = (1 + sum(1 for v in vals if v <= real)) / (n_shuffle + 1)
    return {"entropy": round(real, 4), "shuffled_mean": round(mu, 4), "z": round(z, 2),
            "reduction": round(reduction, 4), "p": round(p, 4), "n_shuffle": n_shuffle,
            "present": bool(p <= alpha and reduction > 0.03)}


def _mi(pairs):
    n = len(pairs)
    if not n:
        return 0.0
    joint = Counter(pairs)
    a = Counter(x for x, _ in pairs)
    b = Counter(y for _, y in pairs)
    return sum(c / n * math.log2(c * n / (a[x] * b[y])) for (x, y), c in joint.items())


def _context_stat(sent_labels):
    pos, prev = [], []
    for labels in sent_labels:
        L = len(labels)
        for i, lab in enumerate(labels):
            pos.append((lab, "first" if i == 0 else ("last" if i == L - 1 else "mid")))
            prev.append((lab, labels[i - 1] if i else "<s>"))
    return _mi(pos) + _mi(prev)


def affix_context_null(sents, infos, n_ctrl=199, seed=0, alpha=0.05):
    """v1.3（G1'）: 解析された接辞が、文中の位置・直前の語の接辞と連動するか。

    札（接辞の列）をトークンの間で入れ替えた対照と比べる。音節の制約だけで「繰り返す語末」が
    できている無作為な語では、札は文脈と連動しない。
    """
    sents = [s for s in sents if len(s) >= 2]
    rng = random.Random(seed)
    out = {}
    for side, idx in (("suffix", 2), ("prefix", 0)):
        labs = [["|".join(infos[w][idx]) if w in infos else "" for w in s] for s in sents]
        flat = [x for s in labs for x in s]
        affixed = sum(1 for x in flat if x)
        if not affixed:
            out[side] = {"real": 0.0, "control_mean": 0.0, "p": 1.0, "affixed_tokens": 0, "significant": False}
            continue
        real = _context_stat(labs)
        lens = [len(s) for s in labs]
        vals = []
        for _ in range(n_ctrl):
            rng.shuffle(flat)
            it = iter(flat)
            vals.append(_context_stat([[next(it) for _ in range(L)] for L in lens]))
        p = (1 + sum(1 for v in vals if v >= real)) / (n_ctrl + 1)
        out[side] = {"real": round(real, 4), "control_mean": round(mean(vals), 4), "p": round(p, 4),
                     "affixed_tokens": affixed, "significant": bool(p <= alpha)}
    out["n_ctrl"] = n_ctrl
    out["method"] = "affix-context MI (G1')"
    return out
