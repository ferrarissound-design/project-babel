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
        # b ≥ 4 かつ 1〜(b−1) がすべて揃うときだけ基数として主張する（値が少ないと {1,2,4,6} から「2進法」と誤る）
        if b >= 4 and b in vs and len(big) >= 3 and all(v % b == 0 for v in big) and all(i in vs for i in range(1, b)):
            scores[b] = len(big)
    if not scores:
        return None, {}
    return min(scores), scores


def solve_numerals(docs, is_inflected, all_counts, anchors=(), extra_equations=(), enable_solver=True, seed_variables=()):
    """数詞候補の補完を厳しめ・緩めの2通りで試し、より多くの式を満たす方を採る（競合仮説の比較）。

    seed_variables: 合計行がないときに、対訳などの別の証拠から得た数詞候補。
    """
    results = [_solve(docs, is_inflected, all_counts, anchors, extra_equations, enable_solver, r, set(seed_variables))
               for r in (0.85, 0.6)]
    if not enable_solver:
        return results[0]
    results.sort(key=lambda r: (r["status"] == "solved", r["confidence"], r.get("satisfied", 0)), reverse=True)
    return results[0]


def _solve(docs, is_inflected, all_counts, anchors, extra_equations, enable_solver, anchor_ratio, seed_variables=frozenset()):
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
    if marker is None and not seed_variables:
        result["reason"] = "合計行を示す標識が見つからず、対訳にも数量の手がかりがない"
        return result
    if marker is not None:
        # 数詞候補 = 合計行に現れる語（標識・定型語・品目名の別形を除く）。人名などは合計行に現れない。
        lists = [x for x in lists if marker in x[1][-1]]
        result["list_documents"] = len(lists)
        variables = set()
        for did, sents, const, damaged in lists:
            for w in sents[-1]:
                if w != marker and w not in const and not _overlaps(w, const):
                    variables.add(w)
        words_in_lists = {w for _, sents, _, _ in lists for s in sents for w in s}
    else:
        # 合計行がない: 対訳から得た候補を出発点にする。会計の式は立たないので、方程式は対訳からだけ立てる
        result["source"] = "parallel"
        variables = set(seed_variables)
        words_in_lists = {w for _, sents, _, _ in lists for s in sents for w in s} | variables
    # 数詞に付く形成要素（例: 「〜十」）を見つけ、その基体（一の位の語など）も候補に加える
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
    # 位置による補完: 一覧の各行で、数詞はその文書の定型語（品目名）の決まった側に隣接する。
    # 同じ位置に立つ未登録の語は、合計行に現れなかっただけの数詞とみなす（人名は反対側に立つ）。
    # 合計行がない場合は出発点の候補が少ないので、1回の一致で位置を推定し、繰り返して広げる
    min_adj, ratio, rounds = (2, anchor_ratio, 2) if marker is not None else (1, anchor_ratio / 2, 5)
    for _ in range(rounds):
        added = set()
        for did, sents, const, damaged in lists:
            adj = Counter()
            for s_ in sents[:-1]:
                for i, w in enumerate(s_):
                    if w in const and w not in variables:
                        if i > 0 and s_[i - 1] in variables:
                            adj[(w, -1)] += 1
                        if i + 1 < len(s_) and s_[i + 1] in variables:
                            adj[(w, 1)] += 1
            occ = Counter(w for s_ in sents[:-1] for w in s_ if w in const)
            # 錨にする定型語: ほぼ常に数詞と隣り合うもの（動詞のように他の語とも隣り合う語は使わない）
            good = sorted(((n / occ[w], n, w, sd) for (w, sd), n in adj.items() if n >= min_adj and n / occ[w] >= ratio),
                          reverse=True)
            if not good:
                continue
            _, _, anchor, side = good[0]
            for s_ in sents[:-1]:
                for i, w in enumerate(s_):
                    j = i + side
                    if w == anchor and 0 <= j < len(s_) and s_[j] not in variables and s_[j] not in const \
                            and s_[j] != marker and not _overlaps(s_[j], const):
                        added.add(s_[j])
        if not added:
            break
        variables |= added
    eqs, soft = [], []      # soft: 欠損のある文書の式（数詞が欠けていれば誤った式になる）
    for did, sents, const, damaged in (lists if marker is not None else []):
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
        if marker is None:
            co = Counter(w for w in ws if w in variables)   # 候補を広げた後の集合で式を立てる（文の数詞を漏らさない）
            if not co:
                continue
        else:
            co = Counter(ws)
            variables.update(co)
        co[UNIT] = -v
        pre.append(dict(co))
    result["variables"] = sorted(variables)
    if marker is None:
        # 対訳の式は数詞の見落としで誤ることがあるので、既知の数詞以外は合意型の解法にかける。
        # 固定順の eqs に入れると最初に入った誤った式が値を決めてしまうので、順序を変えて試す soft に入れる
        soft = [e for e in pre if len(e) > 2 or not any(w for w, _ in anchors if w in e)]
        pre = [e for e in pre if e not in soft]
    if len(eqs) + len(soft) + len(pre) < 3:
        result["reason"] = "合計行のある一覧文書が足りない" if marker is not None else "数量つきの対訳が足りない"
        return result

    def run(order):
        sysm = LinearSystem(variables, has_unit)
        for e in pre + order:
            sysm.add(e)
        vals, rel = sysm.solve()
        vals = _validate(vals, rel)
        sat = checkable = 0
        if vals:
            for e in eqs + soft + pre:
                if all(w in vals or w == UNIT for w in e):
                    checkable += 1
                    sat += sum(c * (1 if w == UNIT else vals[w]) for w, c in e.items()) == 0
        sysm.checkable = checkable
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
    best_soft = soft_order if marker is None else None
    if marker is None and soft:
        best = run(base_order + soft_order)
    for trial in range(30 if soft else 0):
        if best[2] and len(best[2]) >= 0.9 * len(variables) and best[0] >= 0.8 * n_all:
            break
        order = soft_order[:]
        rng.shuffle(order)
        cand = run(base_order + order)
        if key(cand) > key(best):
            best, best_soft = cand, order
    if marker is None:
        # 合計行がない場合、対訳の式は次の段階（形成要素の仮説）の材料でもある。値が決まっていない語を含む式も
        # 捨てずに、合意で最もよく式を満たした並び順のまま渡す（矛盾する式は、その順序で後ろのものが棄却される）
        base_order = base_order + list(best_soft or [])
    else:
        base_order = base_order + [e for e in soft_order
                                   if best[2] and all(w in best[2] for w in e)
                                   and sum(c * best[2][w] for w, c in e.items()) == 0]
    # 値が決まりきらない場合: 数詞の形成要素について競合仮説（加法 / 乗法×k）を立て、合計式との整合で選ぶ
    result["affix_hypotheses"] = []
    if best[2] is None or len(best[2]) < 0.9 * len(variables) or best[0] < 0.6 * n_all:
        aff = Counter()
        for w in variables:
            for k in range(1, len(w)):
                if w[:k] in variables and len(w) - k >= 2:
                    aff[("suf", w[k:])] += 1
                if w[k:] in variables and k >= 2:
                    aff[("pre", w[:k])] += 1
        cands = [(side, x) for (side, x), c in sorted(aff.items(), key=lambda kv: (-kv[1], kv[0])) if c >= 3][:3]

        def pairs_of(side, x):
            return [(w[: -len(x)], w) if side == "suf" else (w[len(x):], w) for w in sorted(variables)
                    if (w.endswith(x) if side == "suf" else w.startswith(x)) and
                    (w[: -len(x)] if side == "suf" else w[len(x):]) in variables]

        def cons(side, x, hyp):
            if hyp is None:
                return []
            if hyp == "add":
                return [{comp: 1, b: -1, x: -1} for b, comp in pairs_of(side, x)]
            out = [{comp: 1, b: -hyp} for b, comp in pairs_of(side, x)]
            if has_unit and x in variables:
                out.append({x: 1, UNIT: -hyp})      # 「x 倍」の形成要素が単独で現れるなら、それ自体が x（例: 十＝10）
            return out

        saved = pre
        # 形成要素が単独の語としても現れるなら、まず加法（例: 十＋三）を仮定し、1つずつ他の仮説と入れ替えて比べる
        choice = {c: ("add" if c[1] in variables else None) for c in cands}

        def evaluate(ch):
            nonlocal pre
            pre = saved + [e for c in cands for e in cons(c[0], c[1], ch[c])]
            r = run(base_order)
            pre = saved
            if marker is None and r[2]:
                # 位取りの単位は、それに掛けられる語の値より大きい（「三」の語があるのに「三倍」の単位はない）。
                # 対訳だけの少ない式では、この一般的な性質がないと誤った倍数の仮説を区別できない
                for c in cands:
                    k = ch[c]
                    if isinstance(k, int) and any(r[2].get(b, 0) >= k for b, _ in pairs_of(c[0], c[1])):
                        return (0, r[1], None, r[3])
            return r

        cur = evaluate(choice)
        for c in cands:
            options = [None] + (["add"] if c[1] in variables else []) + list(range(2, 151))
            scored = []
            for h in options:
                if h == choice[c]:
                    scored.append((key(cur), h, cur))
                    continue
                r = evaluate({**choice, c: h})
                scored.append((key(r), h, r))
            scored.sort(key=lambda t: (t[0], str(t[1])), reverse=True)
            top = scored[0]
            if marker is None:
                # 対訳だけの式は少なく、誤った仮説でも検算できる式をすべて満たしてしまうことがある。
                # 「満たした式の数」は値を多く決める仮説ほど増えるので使わず、「矛盾した式の数」で比べる。
                # 矛盾が最少の仮説のうち、仮説なしより多くの式を説明するものがちょうど1つのときだけ採用する
                def viol(t):
                    return t[2][1].checkable - t[2][0] if t[0][0] else None
                valid = [t for t in scored if t[0][0]]
                none_chk = next((t[2][1].checkable for t in valid if t[1] is None), 0)
                if valid:
                    best_v = min(viol(t) for t in valid)
                    winners = [t for t in valid if viol(t) == best_v and t[1] is not None and t[2][1].checkable > none_chk]
                else:
                    winners = []
                unique = len(winners) == 1
                if unique:
                    top = winners[0]
                elif valid:
                    top = next((t for t in valid if t[1] is None), top)
                    unique = top[1] is None
                result.setdefault("ambiguous_hypotheses", []).append(
                    {"affix": c[1], "zero_violation_rivals": [str(t[1]) for t in winners][:5]}) if len(winners) > 1 else None
            else:
                unique = len(scored) == 1 or top[0] > scored[1][0]
            if top[0][0] and unique:
                choice[c], cur = top[1], top[2]
            result["affix_hypotheses"].append({
                "affix": c[1], "side": c[0], "chosen": choice[c] if top[0][0] and unique else None,
                "rivals": [[str(t[1]), t[0][1]] for t in scored[1:4]]})
        if key(cur) > key(best):
            best = cur
            pre = saved + [e for c in cands for e in cons(c[0], c[1], choice[c])]
    sat, sysm, vals, rel = best
    result["satisfied"] = sat
    result.update(equations=sysm.accepted, rejected=sysm.rejected, redundant=sysm.redundant, relative=rel)
    if vals is None:
        result["reason"] = "方程式系が矛盾、または正の整数解がない"
        return result
    if marker is not None:
        conf = (1 - 0.5 ** max(sysm.redundant, sat - len(vals))) * min(1.0, sat / max(1, len(eqs) + len(pre)))
        unreliable = sat < 0.5 * len(eqs) or conf < 0.45
    else:
        # 対訳だけでは未知数が式より多く、全部は決まらない。検算できた式だけを分母にして部分解を評価する
        chk = getattr(sysm, "checkable", 0)
        conf = (1 - 0.5 ** max(0, sat - len(anchors))) * (sat / chk if chk else 0.0)
        unreliable = chk < 3 or sat < 0.9 * chk or conf < 0.45
        result["checkable_equations"] = chk
    if unreliable:
        result["reason"] = "解は得られたが、満たされる式が少なく信頼できない"
        result["confidence"] = round(conf, 3)
        return result
    if rel:
        conf *= 0.85        # 単位（最小値=1）の仮定を含む
    base, scores = infer_base(vals.values())
    result.update(values=vals, status="solved", confidence=round(min(0.98, conf), 3), base=base,
                  base_scores=scores, undetermined=sorted(v for v in variables if v not in vals))
    return result
