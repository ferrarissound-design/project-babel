"""数詞の推定。

意味の手がかりを使わず、「一覧形式の文書の最終行（合計行）の数量は、それ以前の行の数量の和になる」
という算術的制約から連立一次方程式を立て、数詞語の値を厳密解（有理数）で求める。
値の絶対的な単位は、既知の数詞や対訳がある場合はそれで固定し、ない場合は最小の正整数解を取る。
"""
from __future__ import annotations

import random
from collections import Counter
from fractions import Fraction
from math import gcd

UNIT = "__unit__"


def list_documents(docs):
    """docs: [(doc_id, [文ごとの正規化語形列], 欠損を含むか)]。一覧形式の文書と、その定型語を返す。"""
    out = []
    for did, sents, damaged in docs:
        if len(sents) < 4:
            continue
        body = sents[:-1]
        df = Counter(w for s in body for w in set(s))
        const = {w for w, c in df.items() if c >= 0.6 * len(body)}
        if len(const) >= 2:
            out.append((did, sents, const, damaged))
    return out


def find_sum_marker(lists, all_counts):
    """合計行にだけ現れる語（合計標識）。"""
    final = Counter()
    for _, sents, const, _ in lists:
        for w in set(sents[-1]):
            final[w] += 1
    best = None
    for w, c in sorted(final.items()):
        if c >= 3 and c / max(1, all_counts[w]) >= 0.7 and (best is None or c > final[best]):
            best = w
    return best


def _overlaps(w, const):
    """w が定型語のどれかと語幹を共有していそうか（品目名の別の語形を数詞候補から外す）。"""
    for c in const:
        m = min(len(w), len(c))
        if m < 3:
            continue
        pre = 0
        while pre < m and w[pre] == c[pre]:
            pre += 1
        suf = 0
        while suf < m and w[-1 - suf] == c[-1 - suf]:
            suf += 1
        if max(pre, suf) >= max(3, 0.6 * m):
            return True
    return False


class LinearSystem:
    """同次連立一次方程式を1本ずつ追加し、矛盾する式を棄却する。"""

    def __init__(self, variables, has_unit):
        self.vars = sorted(variables) + ([UNIT] if has_unit else [])
        self.col = {v: i for i, v in enumerate(self.vars)}
        self.has_unit = has_unit
        self.rows: list[dict] = []      # 既約な行（先頭係数1）。pivot 列 -> 行
        self.pivots: list[int] = []
        self.accepted = self.redundant = self.rejected = 0

    def _reduce(self, row):
        row = dict(row)
        for p, r in zip(self.pivots, self.rows):
            if p in row:
                f = row[p]
                for c, v in r.items():
                    nv = row.get(c, 0) - f * v
                    if nv == 0:
                        row.pop(c, None)
                    else:
                        row[c] = nv
        return row

    def add(self, coeffs: dict):
        row = self._reduce({self.col[v]: Fraction(c) for v, c in coeffs.items() if c and v in self.col})
        if not row:
            self.accepted += 1
            self.redundant += 1
            return True
        unit = self.col.get(UNIT)
        if (self.has_unit and set(row) == {unit}) or (not self.has_unit and len(self.rows) + 1 >= len(self.vars)):
            self.rejected += 1
            return False
        p = min(row)
        f = row[p]
        row = {c: v / f for c, v in row.items()}
        for r in self.rows:
            if p in r:
                g = r[p]
                for c, v in row.items():
                    nv = r.get(c, 0) - g * v
                    if nv == 0:
                        r.pop(c, None)
                    else:
                        r[c] = nv
        self.rows.append(row)
        self.pivots.append(p)
        self.accepted += 1
        return True

    def solve(self):
        """決定できた変数の値 {var: Fraction} と、単位が相対的かどうかを返す。"""
        pivset = set(self.pivots)
        free = [c for c in range(len(self.vars)) if c not in pivset]
        unit = self.col.get(UNIT)
        if self.has_unit:
            ref = unit if unit in free else None
            if ref is None:
                return {}, False
        else:
            if not free:
                return {}, True
            dep = Counter()
            for p, r in zip(self.pivots, self.rows):
                others = [c for c in r if c != p]
                if len(others) == 1:
                    dep[others[0]] += 1
            ref = max(free, key=lambda c: (dep[c], -c))
        vals = {self.vars[ref]: Fraction(1)}
        for p, r in zip(self.pivots, self.rows):
            others = [c for c in r if c != p]
            if not others:
                vals[self.vars[p]] = Fraction(0)
            elif others == [ref]:
                vals[self.vars[p]] = -r[ref]
        vals.pop(UNIT, None)
        return vals, not self.has_unit


def _validate(vals, relative):
    pos = {k: v for k, v in vals.items() if v != 0}
    if len(pos) < 4:
        return None
    if relative:
        if any(v < 0 for v in pos.values()):
            pos = {k: -v for k, v in pos.items()}
        den = 1
        for v in pos.values():
            den = den * v.denominator // gcd(den, v.denominator)
        ints = {k: int(v * den) for k, v in pos.items()}
        g = 0
        for v in ints.values():
            g = gcd(g, abs(v))
        pos = {k: Fraction(v // max(1, g)) for k, v in ints.items()}
    if any(v <= 0 or v.denominator != 1 or v > 5000 for v in pos.values()):
        return None
    if relative and min(pos.values()) != 1:
        return None     # 数え上げの体系なら最小の単位 1 を含むはず（含まない解は誤った式による偽解とみなす）
    return {k: int(v) for k, v in pos.items()}


def infer_base(values):
    """単独の数詞語の値から基数を推定: 「b 以上の値がすべて b の倍数」となる最小の b。"""
    vs = sorted(set(values))
    scores = {}
    for b in range(2, 61):
        big = [v for v in vs if v >= b]
        if b in vs and len(big) >= 3 and all(v % b == 0 for v in big) and all(i in vs for i in range(1, min(b, 6))):
            scores[b] = len(big)
    if not scores:
        return None, {}
    return min(scores), scores


def solve_numerals(docs, is_inflected, all_counts, anchors=(), extra_equations=(), enable_solver=True):
    """
    docs: [(doc_id, 文ごとの語形列, 欠損有無)]
    is_inflected(w): 名詞・動詞クラスに属する語なら True（数詞候補から外す）
    anchors: [(語形, 値)]  既知の数詞
    extra_equations: [(語形の列, 値)]  対訳から得た「この語群の和 = 値」
    """
    lists = list_documents(docs)
    marker = find_sum_marker(lists, all_counts)
    result = {"list_documents": len(lists), "sum_marker": marker, "values": {}, "status": "undetermined",
              "variables": [], "equations": 0, "rejected": 0, "redundant": 0, "relative": None, "base": None, "confidence": 0.0,
              "reason": ""}
    if not enable_solver:
        result["reason"] = "数詞ソルバー無効（アブレーション）"
        return result
    if marker is None:
        result["reason"] = "合計行を示す標識が見つからない"
        return result
    # 数詞候補 = 合計行に現れる語（標識・定型語・品目名の別形を除く）。人名などは合計行に現れない。
    lists = [x for x in lists if marker in x[1][-1]]
    result["list_documents"] = len(lists)
    variables = set()
    for did, sents, const, damaged in lists:
        for w in sents[-1]:
            if w != marker and w not in const and not _overlaps(w, const):
                variables.add(w)
    # 数詞に付く形成要素（例: 「〜十」）を見つけ、その基体（一の位の語など）も候補に加える
    words_in_lists = {w for _, sents, _, _ in lists for s in sents for w in s}
    for _ in range(2):
        aff = Counter()
        for w in variables:
            for k in range(2, len(w) - 1):
                aff[("suf", w[k:])] += 1
                aff[("pre", w[:k])] += 1
        for (side, x), c in sorted(aff.items()):
            if c < 3:
                continue
            if x in words_in_lists and x != marker and len(x) >= 2:
                variables.add(x)            # 形成要素そのものが単独の数詞（「十」など）として現れる
            for w in sorted(words_in_lists - variables):
                if w == marker or len(w) < 2:
                    continue
                if (side == "suf" and w + x in variables) or (side == "pre" and x + w in variables):
                    variables.add(w)        # 複合数詞の基体
                elif side == "suf" and w.endswith(x) and w[: -len(x)] in variables:
                    variables.add(w)        # 既知の基体＋形成要素
                elif side == "pre" and w.startswith(x) and w[len(x):] in variables:
                    variables.add(w)
    eqs, soft = [], []      # soft: 欠損のある文書の式（数詞が欠けていれば誤った式になる）
    for did, sents, const, damaged in lists:
        co = Counter()
        for s in sents[:-1]:
            for w in s:
                if w in variables:
                    co[w] += 1
        for w in sents[-1]:
            if w in variables:
                co[w] -= 1
        co = {w: c for w, c in co.items() if c}
        if co:
            (soft if damaged else eqs).append(co)
    has_unit = bool(anchors) or bool(extra_equations)
    pre = []
    for w, v in anchors:
        variables.add(w)
        pre.append({w: 1, UNIT: -v})
    for ws, v in extra_equations:
        co = Counter(ws)
        variables.update(co)
        co[UNIT] = -v
        pre.append(dict(co))
    result["variables"] = sorted(variables)
    if len(eqs) + len(pre) < 3:
        result["reason"] = "合計行のある一覧文書が足りない"
        return result

    def run(order):
        sysm = LinearSystem(variables, has_unit)
        for e in pre + order:
            sysm.add(e)
        vals, rel = sysm.solve()
        vals = _validate(vals, rel)
        sat = 0
        if vals:
            for e in eqs + soft + pre:
                if all(w in vals or w == UNIT for w in e):
                    sat += sum(c * (1 if w == UNIT else vals[w]) for w, c in e.items()) == 0
        return sat, sysm, vals, rel

    def key(r):
        return (r[2] is not None, r[0], len(r[2] or ()))

    # 欠損のない式を先に入れ、欠損のある式は順序を変えながら追加して、最も多くの式を満たす解を選ぶ
    # （誤った式が早い段階で入ると偽解になるため、順序を無作為化した合意で頑健にする）
    base_order = sorted(eqs, key=lambda e: (len(e), sorted(e.items())))
    soft_order = sorted(soft, key=lambda e: (len(e), sorted(e.items())))
    best = run(base_order)
    rng = random.Random(0)
    n_all = len(eqs) + len(soft) + len(pre)
    for trial in range(30 if soft else 0):
        if best[2] and len(best[2]) >= 0.9 * len(variables) and best[0] >= 0.8 * n_all:
            break
        order = soft_order[:]
        rng.shuffle(order)
        cand = run(base_order + order)
        if key(cand) > key(best):
            best = cand
    base_order = base_order + [e for e in soft_order
                               if best[2] and all(w in best[2] for w in e)
                               and sum(c * best[2][w] for w, c in e.items()) == 0]
    # 値が決まりきらない場合: 数詞の形成要素について競合仮説（加法 / 乗法×k）を立て、合計式との整合で選ぶ
    result["affix_hypotheses"] = []
    if best[2] is None or len(best[2]) < 0.9 * len(variables):
        aff = Counter()
        for w in variables:
            for k in range(1, len(w)):
                if w[:k] in variables and len(w) - k >= 2:
                    aff[("suf", w[k:])] += 1
                if w[k:] in variables and k >= 2:
                    aff[("pre", w[:k])] += 1
        for (side, x), c in sorted(aff.items(), key=lambda kv: (-kv[1], kv[0]))[:3]:
            if c < 3:
                continue
            pairs = [(w[: -len(x)], w) if side == "suf" else (w[len(x):], w) for w in sorted(variables)
                     if (w.endswith(x) if side == "suf" else w.startswith(x)) and
                     (w[: -len(x)] if side == "suf" else w[len(x):]) in variables]
            hyps = [("mul", k, [{comp: 1, b: -k} for b, comp in pairs]) for k in range(2, 151)]
            if x in variables:
                hyps.append(("add", x, [{comp: 1, b: -1, x: -1} for b, comp in pairs]))
            scored = []
            saved = pre
            for kind, k, cons in hyps:
                pre = saved + cons
                r = run(base_order)
                if r[2] is not None:
                    scored.append((r[0], len(r[2]), kind, k, r, cons))
            pre = saved
            scored.sort(key=lambda t: (-t[0], -t[1], str(t[3])))
            if scored and scored[0][0] >= 0.7 * len(base_order) and (len(scored) == 1 or scored[0][:2] > scored[1][:2]):
                top = scored[0]
                pre = saved + top[5]
                if key(top[4]) >= key(best):
                    best = top[4]
                result["affix_hypotheses"].append({"affix": x, "side": side, "chosen": [top[2], top[3]],
                                                   "satisfied": top[0], "rivals": [[t[2], t[3], t[0]] for t in scored[1:4]]})
            elif scored:
                result["affix_hypotheses"].append({"affix": x, "side": side, "chosen": None,
                                                   "rivals": [[t[2], t[3], t[0]] for t in scored[:4]]})
    sat, sysm, vals, rel = best
    result["satisfied"] = sat
    result.update(equations=sysm.accepted, rejected=sysm.rejected, redundant=sysm.redundant, relative=rel)
    if vals is None:
        result["reason"] = "方程式系が矛盾、または正の整数解がない"
        return result
    conf = (1 - 0.5 ** sysm.redundant) * min(1.0, sat / max(1, len(eqs) + len(pre)))
    if rel:
        conf *= 0.85        # 単位（最小値=1）の仮定を含む
    base, scores = infer_base(vals.values())
    result.update(values=vals, status="solved", confidence=round(min(0.98, conf), 3), base=base,
                  base_scores=scores, undetermined=sorted(v for v in variables if v not in vals))
    return result
