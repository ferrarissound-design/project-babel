"""対訳（部分的・構造化）と原文の形態素の対応付け。

双方向の IBM Model 1（EM）で「語幹↔概念」「接辞↔文法素性」の対応確率を求め、
両方向の事後確率の積で対応の強さを測る。信頼度は対応の一貫性と証拠量から計算し、
証拠の足りない語は主張しない（判定不能）。
"""
from __future__ import annotations

from collections import Counter, defaultdict

NULL = "<null>"


def flatten(tr):
    """構造化対訳 → (概念トークン[(token, slot)], 素性トークン, 数値のリスト)"""
    concepts, feats, counts = [], [], []
    if tr.get("pred"):
        concepts.append((tr["pred"], "pred"))
    if tr.get("tense"):
        feats.append("T:" + tr["tense"])
    if tr.get("neg"):
        feats.append("NEG")
    if tr.get("cond"):
        concepts.append(("if", "func"))
    if tr.get("kind") == "total":
        concepts.append(("total", "func"))

    def np(a, role):
        feats.append("R:" + role)
        if a.get("head"):
            for part in a["head"].split("+"):
                concepts.append((part, "head:" + role))
        if a.get("pl"):
            feats.append("N:PL")
        if a.get("adj"):
            concepts.append((a["adj"], "adj:" + role))
        if a.get("title"):
            concepts.append((a["title"], "title:" + role))
        if a.get("deriv"):
            feats.append("D:" + a["deriv"])
        if a.get("count") is not None:
            counts.append((a["count"], role))
        if a.get("poss"):
            np(a["poss"], "possessor")

    for a in tr.get("args", []):
        np(a, a["role"])
    return concepts, feats, counts


def ibm1(pairs, allowed, n_iter=12):
    """pairs: [(src_units, tgt_tokens)]。t[src][tgt] = P(tgt | src) を返す。"""
    t = defaultdict(dict)
    for src, tgt in pairs:
        for s in list(src) + [NULL]:
            for c in tgt:
                if allowed(s, c):
                    t[s][c] = 1.0
    for _ in range(n_iter):
        cnt = defaultdict(lambda: defaultdict(float))
        for src, tgt in pairs:
            srcs = list(src) + [NULL]
            for c in tgt:
                z = sum(t[s].get(c, 0.0) * (0.3 if s == NULL else 1.0) for s in srcs)
                if z <= 0:
                    continue
                for s in srcs:
                    p = t[s].get(c, 0.0) * (0.3 if s == NULL else 1.0)
                    if p:
                        cnt[s][c] += p / z
        for s, row in cnt.items():
            z = sum(row.values())
            t[s] = {c: v / z for c, v in row.items()}
    return t


def posterior(t, src, tgt):
    """post[c][s] = 各 tgt トークン c が src 単位 s から生成された事後確率。"""
    out = {}
    srcs = list(src) + [NULL]
    for c in tgt:
        z = sum(t[s].get(c, 0.0) * (0.3 if s == NULL else 1.0) for s in srcs)
        out[c] = {s: (t[s].get(c, 0.0) / z if z > 0 else 0.0) for s in src}
    return out


class Aligner:
    def __init__(self, items, mode="em", kappa=0.7):
        """items: [{'units': [(unit, word_index)], 'concepts': [(token, slot)], 'feats': [...]}]"""
        self.items = items
        self.kappa = kappa
        self.mode = mode
        use_em = mode == "em"
        self.n_tgt = Counter()
        fwd = [(sorted({u for u, _ in it["units"]}),
                sorted({"C:" + c for c, _ in it["concepts"]} | {"F:" + f for f in it["feats"]})) for it in items]

        def allowed(s, c):       # 接辞は文法素性にだけ対応できる（語幹はどちらにも対応できる）
            return not (s.startswith("A:") and c.startswith("C:"))

        self.A = defaultdict(lambda: defaultdict(float))
        self.joint = defaultdict(Counter)
        self.n = Counter()
        self.links = []          # 文ごとの (unit, token) -> 強さ
        if use_em:
            t1 = ibm1(fwd, allowed)
            t2 = ibm1([(tg, sr) for sr, tg in fwd], lambda c, s: c == NULL or allowed(s, c))
        for src, tgt in fwd:
            link = {}
            self.n_tgt.update(tgt)
            if use_em:
                p1 = posterior(t1, src, tgt)
                p2 = posterior(t2, tgt, src)
            for s in src:
                self.n[s] += 1
                for c in tgt:
                    if not allowed(s, c):
                        continue
                    self.joint[s][c] += 1
                    a = p1[c][s] * p2[s][c] if use_em else 1.0 / max(1, len(tgt))
                    if a > 0:
                        self.A[s][c] += a
                        link[(s, c)] = a
            self.links.append(link)

    def best(self, unit, prefix):
        """unit に最もよく対応するトークン（prefix='C:' or 'F:'）と信頼度、次点。"""
        if self.mode == "em":
            row = sorted(((v, c) for c, v in self.A.get(unit, {}).items() if c.startswith(prefix)), reverse=True)
        elif self.mode == "count":      # 共起回数が最大の概念
            row = sorted(((float(n), c) for c, n in self.joint[unit].items() if c.startswith(prefix)), reverse=True)
        else:                           # Dice 係数
            row = sorted(((2.0 * n / (self.n[unit] + self.n_tgt[c]), c) for c, n in self.joint[unit].items()
                          if c.startswith(prefix)), reverse=True)
        if not row:
            return None, 0.0, []
        v, c = row[0]
        nj, ns = self.joint[unit][c], self.n[unit]
        if self.mode == "em":
            conf = v / (nj + self.kappa) * min(1.0, 2.0 * nj / ns)
        else:
            conf = nj / ns if self.mode == "count" else v
        alts = [(c2[2:], round(v2 / (self.joint[unit][c2] + self.kappa), 3)) for v2, c2 in row[1:3]]
        return c[2:], round(min(0.99, conf), 3), alts

    def functions(self, unit, thr=0.3):
        """接辞（または機能語）が担う文法素性の集合。"""
        out = {}
        for c, v in self.A.get(unit, {}).items():
            if c.startswith("F:") and self.joint[unit][c] >= 2 and v / self.n[unit] >= thr:
                out[c[2:]] = round(v / self.n[unit], 3)
        return out
