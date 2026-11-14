"""同じ画像の2つの転写を、クラス番号に頼らずに突き合わせる（標準ライブラリだけ。画像は使わない）。

クラス番号は転写ごとに別に振られるので、番号そのものは比べられない。比べられるのは「どの位置とどの位置を同じ記号と
みなしたか」という関係（位置の組の同値関係）。そのために、まず2つの転写の行を位置ごとに対応させる。

前提と限界（docs/ROBUST_STRUCTURE.md 第2節）:
- 2つの転写の行が1対1に対応していること（行数と行の順序が同じ）。行の分け方が違う転写は扱えない
- 長さが同じ行は、位置ごとに1対1に対応すると**仮定**する（分割と結合が同じ行の中で打ち消し合っていれば誤る）
- 長さが違う行の対応は、長さが同じ行から学んだクラスの対応を手掛かりにした推定で、記号の位置（枠）がないと確かめられない
"""
from __future__ import annotations

from collections import Counter, defaultdict

GAP = -0.6          # 一方にしかない記号
SPLIT = 0.7         # 一方の1記号が他方の2記号に当たる（分割・結合）
KNOWN = 3.0         # ほかの位置でも対応しているクラスの組
NEW = 1.0           # どちらのクラスも、ほかの位置での対応がまだ分からない
CONFLICT = 0.1      # 少なくとも一方が、ほかの位置では別のクラスに対応している


def _score(x, y, assoc, tot_a, tot_b):
    n = assoc.get((x, y), 0)
    if n > 0:
        return KNOWN
    if tot_a.get(x, 0) == 0 and tot_b.get(y, 0) == 0:
        return NEW
    return CONFLICT


def align_row(a, b, assoc, tot_a, tot_b):
    """戻り値: 列のリスト。列は (a の位置のタプル, b の位置のタプル)。1対1・1対2・2対1・片方だけ。"""
    n, m = len(a), len(b)
    NEG = float("-inf")
    T = [[NEG] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    T[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if T[i][j] == NEG:
                continue
            cand = []
            if i < n and j < m:
                cand.append((i + 1, j + 1, _score(a[i], b[j], assoc, tot_a, tot_b)))
            if i < n and j + 1 < m:
                cand.append((i + 1, j + 2, SPLIT))
            if i + 1 < n and j < m:
                cand.append((i + 2, j + 1, SPLIT))
            if i < n:
                cand.append((i + 1, j, GAP))
            if j < m:
                cand.append((i, j + 1, GAP))
            for ni, nj, s in cand:
                if T[i][j] + s > T[ni][nj]:
                    T[ni][nj] = T[i][j] + s
                    back[ni][nj] = (i, j)
    cols, i, j = [], n, m
    while (i, j) != (0, 0):
        pi, pj = back[i][j]
        cols.append((tuple(range(pi, i)), tuple(range(pj, j))))
        i, j = pi, pj
    return cols[::-1]


def align(rows_a, rows_b, iterations=4):
    """行ごとの対応を推定する。戻り値: {"columns": 行ごとの列, "anchor_rows": 長さが同じ行の番号, "assoc": クラスの対応の数}"""
    if len(rows_a) != len(rows_b):
        raise ValueError(f"行数が違う（{len(rows_a)} と {len(rows_b)}）。行の分け方が違う転写は突き合わせられない")
    anchors = [k for k, (a, b) in enumerate(zip(rows_a, rows_b)) if len(a) == len(b)]
    cols = {k: [((p,), (p,)) for p in range(len(rows_a[k]))] for k in anchors}

    def count(cols, skip=None):
        assoc, ta, tb = Counter(), Counter(), Counter()
        for k, cs in cols.items():
            if k == skip:
                continue
            for pa, pb in cs:
                if len(pa) == 1 and len(pb) == 1:
                    x, y = rows_a[k][pa[0]], rows_b[k][pb[0]]
                    assoc[(x, y)] += 1
                    ta[x] += 1
                    tb[y] += 1
        return assoc, ta, tb
    for _ in range(iterations):
        new = dict(cols)
        for k in range(len(rows_a)):
            if k in anchors:
                continue
            assoc, ta, tb = count(cols, skip=k)             # その行自身の対応は手掛かりにしない
            new[k] = align_row(rows_a[k], rows_b[k], assoc, ta, tb)
        if new == cols:
            break
        cols = new
    assoc, _, _ = count(cols)
    return {"columns": [cols[k] for k in range(len(rows_a))], "anchor_rows": anchors,
            "assoc": sorted(([x, y, n] for (x, y), n in assoc.items()), key=lambda t: (-t[2], t[0], t[1]))}


def one_to_one(alignment, rows=None):
    """1対1に対応した位置の一覧 [(行, a の位置, b の位置)]。rows を指定するとその行だけ。"""
    out = []
    for k, cs in enumerate(alignment["columns"]):
        if rows is not None and k not in rows:
            continue
        for pa, pb in cs:
            if len(pa) == 1 and len(pb) == 1:
                out.append((k, pa[0], pb[0]))
    return out


def _adjusted_rand(x, y):
    n = len(x)
    if n < 2:
        return None

    def c2(v):
        return v * (v - 1) / 2
    nij, ni, nj = Counter(zip(x, y)), Counter(x), Counter(y)
    s = sum(c2(v) for v in nij.values())
    a, b = sum(c2(v) for v in ni.values()), sum(c2(v) for v in nj.values())
    exp = a * b / c2(n)
    mx = (a + b) / 2
    return None if mx == exp else (s - exp) / (mx - exp)


def pair_agreement(rows_a, rows_b, alignment, rows=None):
    """1対1に対応した位置の組について、「同じ記号とみなしたか」が2つの転写で一致するかを数える。"""
    pos = one_to_one(alignment, rows)
    xa = [rows_a[k][i] for k, i, _ in pos]
    xb = [rows_b[k][j] for k, _, j in pos]
    n11 = n10 = n01 = n00 = 0
    only_a, only_b = [], []
    for p in range(len(pos)):
        for q in range(p + 1, len(pos)):
            sa, sb = xa[p] == xa[q], xb[p] == xb[q]
            if sa and sb:
                n11 += 1
            elif sa:
                n10 += 1
                only_a.append([list(pos[p]), list(pos[q])])
            elif sb:
                n01 += 1
                only_b.append([list(pos[p]), list(pos[q])])
            else:
                n00 += 1
    return {"positions": len(pos), "pairs": n11 + n10 + n01 + n00, "same_in_both": n11, "same_only_in_a": n10, "same_only_in_b": n01,
            "different_in_both": n00, "adjusted_rand": None if (v := _adjusted_rand(xa, xb)) is None else round(v, 4),
            "a_same_confirmed_by_b": round(n11 / (n11 + n10), 4) if n11 + n10 else None,
            "b_same_confirmed_by_a": round(n11 / (n11 + n01), 4) if n11 + n01 else None,
            "pairs_same_only_in_a": only_a, "pairs_same_only_in_b": only_b}


def bracket(rows_a, rows_b, alignment):
    """2つの転写の食い違いを挟む2つの転写を、a の切り出しのまま作る。

    fine:   1対1に対応した位置は「両方が同じとみなした」ときだけ同じクラス（a と b のクラスの組をクラスにする）
    coarse: 「どちらかが同じとみなした」位置を同じクラスにまとめる（推移的に閉じる）
    1対1に対応しない位置は a のクラスのまま（b の情報がない）。
    どちらの転写でも有意な規則は、2つの方式の分類の食い違いに左右されていない。
    """
    b_of = {}
    for k, i, j in one_to_one(alignment):
        b_of[(k, i)] = rows_b[k][j]
    fine_id, fine = {}, []
    for k, r in enumerate(rows_a):
        fine.append([fine_id.setdefault((x, b_of.get((k, i), ("a-only", x))), len(fine_id)) for i, x in enumerate(r)])
    parent = {}

    def find(v):
        parent.setdefault(v, v)
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    for k, r in enumerate(rows_a):
        for i, x in enumerate(r):
            find(("a", x))
            if (k, i) in b_of:
                parent[find(("a", x))] = find(("b", b_of[(k, i)]))
    coarse_id = {}
    coarse = [[coarse_id.setdefault(find(("a", x)), len(coarse_id)) for x in r] for r in rows_a]
    return {"fine": fine, "coarse": coarse, "fine_classes": len(fine_id), "coarse_classes": len(coarse_id)}


def hapax_check(rows_a, rows_b, alignment):
    """a で1回しか出ないクラスの位置が、b では繰り返すクラスに当たっているか（a の分けすぎの候補）。逆も数える。"""
    ca = Counter(x for r in rows_a for x in r)
    cb = Counter(x for r in rows_b for x in r)
    pos = one_to_one(alignment)
    a_hapax = [(k, i, j) for k, i, j in pos if ca[rows_a[k][i]] == 1]
    b_hapax = [(k, i, j) for k, i, j in pos if cb[rows_b[k][j]] == 1]
    by_b = defaultdict(list)
    for k, i, j in a_hapax:
        by_b[rows_b[k][j]].append([k, i])
    return {"a_hapax_classes": sum(1 for v in ca.values() if v == 1), "b_hapax_classes": sum(1 for v in cb.values() if v == 1),
            "a_hapax_aligned": len(a_hapax), "a_hapax_repeated_in_b": sum(1 for k, i, j in a_hapax if cb[rows_b[k][j]] > 1),
            "b_hapax_aligned": len(b_hapax), "b_hapax_repeated_in_a": sum(1 for k, i, j in b_hapax if ca[rows_a[k][i]] > 1),
            "hapax_in_both": sum(1 for k, i, j in a_hapax if cb[rows_b[k][j]] == 1)}
