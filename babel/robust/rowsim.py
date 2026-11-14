"""行の組の類似の検定（認識の誤りに強い統計量）。

既存の linepair 族（babel/structure/stats.py）は「2つの行が共有する 2〜4 記号列の数」を数える。連続した記号列の完全一致なので、
記号1つの取り違え・分割・結合で一致が切れる（IMG_4494 の H01 が方式によって消えた仕組み、docs/ROBUST_STRUCTURE.md 第2節）。

ここでは行の組ごとに「順序を保って対応する記号の重みの合計の最大値」（重みが 0/1 なら最長共通部分列の長さ）を統計量にする。

- 帰無モデル: 全体シャッフル（行の長さを保ち、すべての記号の位置を入れ替える）。linepair 族と同じ
- 多重検定: 組ごとに置換分布の平均・標準偏差で標準化し、行の組すべての最大値の置換分布と比べる（Westfall–Young の一段階 maxT）。
  族全体の誤り率（1つでも誤って有意とする確率）を制御する。minP は統計量が離散的で置換回数が少ないと p 値の下限が有意水準を超える
- 支持の下限: 統計量が min_score 未満の組は報告しない（1〜2 記号の一致だけで構造とは呼ばない）

記号の同一性は「重み」で与える。クラス番号の一致（hard）のほか、字形の類似度（soft、0〜1）も同じ計算で扱える。
"""
from __future__ import annotations

import itertools
import random

DEFAULTS = {"permutations": 999, "alpha": 0.05, "min_score": 3.0, "seed": 0, "sd_floor": 0.25, "null": "global"}


def lcs_score(a, b, weight):
    """順序を保った対応の重みの合計の最大値。a・b は項目番号の列、weight(x, y) >= 0。"""
    prev = [0.0] * (len(b) + 1)
    for x in a:
        cur = [0.0]
        for j, y in enumerate(b):
            w = weight(x, y)
            cur.append(max(prev[j] + w if w > 0 else 0.0, prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def lcs_alignment(a, b, weight):
    """最大の対応そのもの。戻り値: [(a の位置, b の位置, 重み), ...]"""
    n, m = len(a), len(b)
    T = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            w = weight(a[i], b[j])
            T[i + 1][j + 1] = max(T[i][j] + w if w > 0 else 0.0, T[i][j + 1], T[i + 1][j])
    out, i, j = [], n, m
    while i > 0 and j > 0:
        w = weight(a[i - 1], b[j - 1])
        if w > 0 and abs(T[i][j] - (T[i - 1][j - 1] + w)) < 1e-12:
            out.append((i - 1, j - 1, w))
            i, j = i - 1, j - 1
        elif T[i][j] == T[i - 1][j]:
            i -= 1
        else:
            j -= 1
    return out[::-1]


def _permutations(rows, n_perm, seed, null="global"):
    """位置の入れ替え（置換ごとに、位置 → 元の位置）。

    global: すべての位置を入れ替える。
    column: 行の先頭から数えた位置が同じ記号どうしだけを入れ替える（「行の何番目にどの記号が来やすいか」は保つ）。
            どの行も同じ書き出しで始まるような、位置の決まりだけで行が似て見える場合を除くための感度分析に使う。
    """
    rng = random.Random(f"rowsim-{seed}")
    n_items = sum(len(r) for r in rows)
    if null == "global":
        order, out = list(range(n_items)), []
        for _ in range(n_perm):
            rng.shuffle(order)
            out.append(list(order))
        return out
    if null != "column":
        raise ValueError(f"帰無モデル {null}")
    by_col, k = {}, 0
    for r in rows:
        for c in range(len(r)):
            by_col.setdefault(c, []).append(k)
            k += 1
    out = []
    for _ in range(n_perm):
        order = list(range(n_items))
        for c in sorted(by_col):
            sh = list(by_col[c])
            rng.shuffle(sh)
            for dst, src in zip(by_col[c], sh):
                order[dst] = src
        out.append(order)
    return out


def _null_python(rows, pairs, perms, weight):
    spans, k = [], 0
    for r in rows:
        spans.append((k, k + len(r)))
        k += len(r)
    flat = [x for r in rows for x in r]
    out = []
    for perm in perms:
        sh = [flat[i] for i in perm]
        out.append([lcs_score(sh[spans[i][0]:spans[i][1]], sh[spans[j][0]:spans[j][1]], weight) for i, j in pairs])
    return out


def _null_numpy(np, rows, pairs, perms, W):
    """置換をまとめて計算する（_null_python と同じ値）。W: 項目×項目の重みの行列。"""
    spans, k = [], 0
    for r in rows:
        spans.append((k, k + len(r)))
        k += len(r)
    flat = np.array([x for r in rows for x in r], dtype=np.int64)
    P = flat[np.array(perms, dtype=np.int64)]                       # 置換 × 位置
    B = P.shape[0]
    out = np.zeros((B, len(pairs)))
    for c, (i, j) in enumerate(pairs):
        A, Bm = P[:, spans[i][0]:spans[i][1]], P[:, spans[j][0]:spans[j][1]]
        m = Bm.shape[1]
        prev = np.zeros((B, m + 1))
        for s in range(A.shape[1]):
            cur = np.zeros((B, m + 1))
            w = W[A[:, s][:, None], Bm]                              # B × m
            for t in range(m):
                diag = np.where(w[:, t] > 0, prev[:, t] + w[:, t], 0.0)
                cur[:, t + 1] = np.maximum(np.maximum(diag, prev[:, t + 1]), cur[:, t])
            prev = cur
        out[:, c] = prev[:, m]
    return out.tolist()


def _standardize(null, obs, n_pairs, sd_floor):
    """組ごとに置換分布の平均と標準偏差で標準化する（行の長さが違う組を同じ物差しにそろえる）。"""
    B = len(null)
    mean = [sum(r[c] for r in null) / B for c in range(n_pairs)]
    sd = [max(sd_floor, (sum((r[c] - mean[c]) ** 2 for r in null) / B) ** 0.5) for c in range(n_pairs)]
    z_obs = [(obs[c] - mean[c]) / sd[c] for c in range(n_pairs)]
    z_max = [max((r[c] - mean[c]) / sd[c] for c in range(n_pairs)) for r in null]
    return mean, sd, z_obs, z_max


def row_similarity(rows, weight=None, weight_matrix=None, cfg=None, use_numpy=True):
    """行の組の類似の検定。

    rows: 行ごとの項目番号の列。項目番号はクラス番号か、記号の出現1つずつに振った番号（weight_matrix の添字）。
    weight: (x, y) → 重み。省略すると「番号が同じなら 1」（クラス番号の一致）。
    weight_matrix: 項目×項目の重みの行列（リストのリストか NumPy の配列）。指定すると weight より優先。
    戻り値: {"pairs": [{"rows": [i, j], "score", "z", "p_raw", "p_adj", "significant", "alignment"}, ...], ...}

    p_adj は、標準化した統計量の「行の組すべての最大値」の置換分布による（maxT）。観測も標本に数える。
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    rows = [list(r) for r in rows]
    pairs = [(i, j) for i, j in itertools.combinations(range(len(rows)), 2) if rows[i] and rows[j]]
    n_items = sum(len(r) for r in rows)
    base = {"statistic": "lcs", "null": cfg["null"] + "-shuffle", "correction": "maxT-studentized", "permutations": cfg["permutations"],
            "alpha": cfg["alpha"], "min_score": cfg["min_score"], "seed": cfg["seed"], "rows": len(rows), "items": n_items}
    if not pairs:
        return {**base, "pairs": [], "n_pairs": 0}
    if weight_matrix is not None:
        def weight(x, y, M=weight_matrix):
            return float(M[x][y])
    elif weight is None:
        def weight(x, y):
            return 1.0 if x == y else 0.0
    obs = [lcs_score(rows[i], rows[j], weight) for i, j in pairs]
    perms = _permutations(rows, cfg["permutations"], cfg["seed"], cfg["null"])
    null = None
    np = None
    if use_numpy:
        try:
            import numpy as np
        except ImportError:
            np = None
    if np is not None and weight_matrix is not None:
        null = _null_numpy(np, rows, pairs, perms, np.asarray(weight_matrix, dtype=np.float64))
    elif np is not None:
        ids = sorted({x for r in rows for x in r})
        idx = {x: k for k, x in enumerate(ids)}
        W = np.array([[weight(x, y) for y in ids] for x in ids], dtype=np.float64)
        null = _null_numpy(np, [[idx[x] for x in r] for r in rows], pairs, perms, W)
    else:
        null = _null_python(rows, pairs, perms, weight)
    null = [[round(v, 9) for v in r] for r in null]
    obs = [round(v, 9) for v in obs]
    mean, sd, z_obs, z_max = _standardize(null, obs, len(pairs), cfg["sd_floor"])
    z_max.sort()
    total = len(null) + 1
    out = []
    for c, (i, j) in enumerate(pairs):
        ge = 1 + sum(1 for r in null if r[c] >= obs[c])
        lo, hi = 0, len(z_max)                                        # z_max < z_obs である置換の数
        while lo < hi:
            mid = (lo + hi) // 2
            if z_max[mid] < z_obs[c] - 1e-9:
                lo = mid + 1
            else:
                hi = mid
        p_adj = (1 + len(z_max) - lo) / total
        sig = p_adj <= cfg["alpha"] and obs[c] >= cfg["min_score"]
        rec = {"rows": [i, j], "score": round(obs[c], 4), "z": round(z_obs[c], 3), "null_mean": round(mean[c], 3),
               "p_raw": round(ge / total, 5), "p_adj": round(p_adj, 5), "significant": sig}
        if sig or ge / total <= cfg["alpha"]:
            rec["alignment"] = [[a, b, round(w, 3)] for a, b, w in lcs_alignment(rows[i], rows[j], weight)]
        out.append(rec)
    out.sort(key=lambda r: (r["p_adj"], -r["z"], r["rows"]))
    return {**base, "pairs": out, "n_pairs": len(pairs)}


def significant_pairs(result):
    return [tuple(p["rows"]) for p in result["pairs"] if p["significant"]]
