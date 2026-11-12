"""構造解析の採点。

1. 設計した規則（gold rules）: 正しく発見 / 見逃し / 判定不能（記述的観察にだけ出た）に分ける
2. 出された仮説の真偽: 母集団オラクル（分析に使っていない大量の行）で、その規則性が成り立つかを確かめる
   - 無作為の生成器では、どの仮説も偽（規則が存在しない）
   - 行の組の仮説は、2行が設計上の語（2記号以上）を共有するときだけ真
   - 接辞の仮説は、設計した接辞の一覧（gold rules）にあるときだけ真。一覧のないコーパスでは「検証不能」
3. 単語境界: 各方式の境界 F1（正解の境界は採点でだけ使う）
"""
from __future__ import annotations

import math
import random
from collections import Counter

from babel.structure.stats import Expectation, SubstitutionFamily, hypergeom_logsf, shuffle_within, surprise

STRICT_P = 3.0          # オラクルで「成り立つ」とみなす強さ: -log10 p ≥ 3
MIN_RATIO = 1.5         # 期待値の何倍以上か


def _ids(pop):
    sym = {}
    lines = []
    for ws, _ in pop:
        lines.append(tuple(sym.setdefault(ch, len(sym)) for w in ws for ch in w))
    return lines, sym


class Oracle:
    def __init__(self, generator, null=False):
        self.null = null
        self.gen = generator
        pop = generator.sample(2000, which="population") if generator.max_lines is None else generator.sample(0, "population")
        self.lines, self.sym = _ids(pop)
        self.exp = Expectation(self.lines)
        self.rexp = Expectation([ln[::-1] for ln in self.lines])
        self.gold_words = generator.gold_words()
        self.gold_affixes = {(r["side"], tuple(r["symbols"])) for r in generator.gold_rules() if r["family"] == "affix"}
        self.has_affix_gold = any(r["family"] == "affix" for r in generator.gold_rules()) or generator.name in (
            "random_uniform", "random_zipf", "embedded_repeat", "periodic")
        self._subst = None

    def seq(self, names):
        if any(n not in self.sym for n in names):
            return None
        return tuple(self.sym[n] for n in names)

    def _strong(self, k, mu):
        return k >= MIN_RATIO * mu and surprise(k, mu) >= STRICT_P

    def verify(self, key, analysis_rows):
        """戻り値: True / False / None（検証不能）。"""
        if self.null:
            return False
        fam = key["family"]
        if fam == "linepair":
            pairs = key.get("pairs") or [key["lines"]]
            ok = sum(self._pair_true(analysis_rows[i], analysis_rows[j]) for i, j in pairs)
            return ok * 2 >= len(pairs)
        if fam == "affix":
            if not self.has_affix_gold:
                return None
            return (key["side"], tuple(key["symbols"])) in self.gold_affixes
        s = self.seq(key["symbols"])
        if s is None:
            return False
        if fam == "ngram":
            k = sum(1 for ln in self.lines for i in range(len(ln) - len(s) + 1) if ln[i:i + len(s)] == s)
            return self._strong(k, self.exp.ngram(s))
        if fam == "position":
            n = len(s)
            if key["side"] == "^":
                k = sum(1 for ln in self.lines if ln[:n] == s)
                return self._strong(k, self.exp.edge(s))
            k = sum(1 for ln in self.lines if len(ln) >= n and ln[-n:] == s)
            return self._strong(k, self.rexp.edge(s[::-1]))
        if fam == "gap":
            d = key["distance"]
            a = s[0]
            k = sum(1 for ln in self.lines for i in range(len(ln) - d) if ln[i] == a and ln[i + d] == a)
            return self._strong(k, self.exp.gap(a, d))
        if fam == "cooccurrence":
            a, b = s
            N = len(self.lines)
            na = sum(1 for ln in self.lines if a in ln)
            nb = sum(1 for ln in self.lines if b in ln)
            both = sum(1 for ln in self.lines if a in ln and b in ln)
            mu = na * nb / N
            return both >= 1.3 * mu and -hypergeom_logsf(both, N, na, nb) / math.log(10) >= STRICT_P
        if fam == "dispersion":
            x = s[0]
            tot = sum(len(ln) for ln in self.lines)
            p = sum(ln.count(x) for ln in self.lines) / tot
            qs = [1 - (1 - p) ** len(ln) for ln in self.lines]
            E = sum(qs)
            sd = math.sqrt(sum(q * (1 - q) for q in qs)) or 1e-9
            obs = sum(1 for ln in self.lines if x in ln)
            if key["side"] == "even":
                return obs >= 1.05 * E and (obs - E) / sd >= 3
            return obs <= 0.9 * E and (E - obs) / sd >= 3
        if fam == "substitution":
            return self._substitution(s)
        return None

    def _pair_true(self, ri, rj):
        """2行が、設計上の語（または同じ文型）を共有するか。語の内部にある3記号以上の共通部分列も認める（形態素の共有）。"""
        if ri[1] is not None and ri[1] == rj[1]:
            return True
        si, sj = "".join(ri[0]), "".join(rj[0])
        for w in self.gold_words:
            if len(w) >= 2 and w in si and w in sj:
                return True
        if not self.gen.has_words:
            return False

        def inner(ws, n=3):
            out = set()
            for w in ws:
                for i in range(len(w) - n + 1):
                    out.add(w[i:i + n])
            return out
        return bool(inner(ri[0]) & inner(rj[0]))

    def _substitution(self, s):
        if self._subst is None:
            sub = self.lines[:600]
            fam = SubstitutionFamily(sub, {})
            obs = fam.compute(sub)
            rng = random.Random(0)
            null = [fam.compute(shuffle_within(sub, rng)) for _ in range(10)]
            self._subst = {}
            for k in set(obs) | {k for n in null for k in n}:
                vals = [n.get(k, 0.0) for n in null]
                mu = sum(vals) / len(vals)
                sd = max(0.05, math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)))
                self._subst[k] = (obs.get(k, 0.0) - mu) / sd >= 3
        return self._subst.get(tuple(sorted(s)), False)


def match_gold(rule, key):
    if rule["family"] != key["family"]:
        return False
    g, h = rule["symbols"], key.get("symbols") or []
    if rule["family"] == "position":
        if rule["side"] != key["side"]:
            return False
        return h[:len(g)] == g if rule["side"] == "^" else h[-len(g):] == g
    if rule["family"] == "ngram":
        def contains(a, b):
            return any(a[i:i + len(b)] == b for i in range(len(a) - len(b) + 1))
        return len(h) >= 2 and (contains(h, g) or contains(g, h))
    if rule["family"] == "gap":
        return h == g and key["distance"] % rule["distance"] == 0
    if rule["family"] == "substitution":
        return set(h) <= set(g)
    if rule["family"] in ("affix", "dispersion"):
        return rule["side"] == key["side"] and h == g
    return False


def score_analysis(analysis, generator, rows, oracle):
    rules = generator.gold_rules()
    hyps = analysis["hypotheses"]
    desc = analysis["descriptive_observations"]
    found, missed, undecided = [], [], []
    for r in rules:
        if any(match_gold(r, h["key"]) for h in hyps):
            found.append(r["label"])
        elif any(match_gold(r, d["key"]) for d in desc):
            undecided.append(r["label"])
        else:
            missed.append(r["label"])
    verdicts = Counter()
    per_family = {}
    false_examples = []
    for h in hyps:
        v = oracle.verify(h["key"], rows)
        tag = "true" if v else "unverifiable" if v is None else "false"
        verdicts[tag] += 1
        pf = per_family.setdefault(h["family"], Counter())
        pf[tag] += 1
        if tag == "false" and len(false_examples) < 5:
            false_examples.append(h["observation"])
    return {"gold_found": found, "gold_missed": missed, "gold_undecided": undecided,
            "hypotheses": len(hyps), "true": verdicts["true"], "false": verdicts["false"],
            "unverifiable": verdicts["unverifiable"], "per_family": {k: dict(v) for k, v in per_family.items()},
            "false_examples": false_examples,
            "tiers": dict(Counter(h["tier"] for h in hyps)),
            "false_by_tier": dict(Counter(h["tier"] for h in hyps if oracle.verify(h["key"], rows) is False)),
            "verdict": analysis["verdict"]["code"]}
