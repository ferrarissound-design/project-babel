"""比較用の単純な手法。解読エンジンと同じ出力形式で差し替えられる。

freq   : 頻度だけ。頻出する語頭・語末の文字列を接辞とみなして切り落とす。意味は共起回数の最大値。
ngram  : n-gram 統計。後続文字の種類数（successor variety）の山で切る。意味は Dice 係数。
どちらも表記揺れの正規化・パラダイム分析・EM・数詞ソルバーを使わない。
"""
from __future__ import annotations

from collections import Counter, defaultdict

CONFIGS = {
    "engine": {},
    "baseline_freq": {"use_variants": False, "morph_method": "freq", "align_method": "count", "pos_constraint": False,
                      "use_numeral_solver": False, "abstain": False, "use_metadata": False},
    "baseline_ngram": {"use_variants": False, "morph_method": "successor", "align_method": "dice", "pos_constraint": False,
                       "use_numeral_solver": False, "use_metadata": False},
    "no_variants": {"use_variants": False},
    "no_morph": {"use_morph": False},
    "no_em": {"align_method": "count"},
    "no_numeral_solver": {"use_numeral_solver": False},
    "no_abstain": {"abstain": False},
    "no_pos_constraint": {"pos_constraint": False},
    "no_metadata": {"use_metadata": False},
}


class _SimpleMorph:
    enable = True

    def __init__(self, counts):
        self.counts = dict(counts)
        self.compounds = {}
        self.cache = {}
        self._fit()
        self.affixes = {}
        n = {"suffix": 0, "prefix": 0}
        for w, c in self.counts.items():
            p, _, x = self.analyze(w)
            for tag, lst in (("P:", p), ("S:", x)):
                for a in lst:
                    e = self.affixes.setdefault(tag + a, [0, 0])
                    e[0] += 1
                    e[1] += c
            n["suffix"] += c * bool(x)
            n["prefix"] += c * bool(p)
        self.support = n
        lo, hi = sorted(n.values())
        self.position = "none" if hi == 0 else "both" if lo >= 0.6 * hi else max(n, key=lambda k: n[k])

    def cuts(self, w):
        p, s, x = self.analyze(w)
        out, pos = [], 0
        for a in p:
            pos += len(a)
            out.append(pos)
        pos += len(s)
        for a in x:
            out.append(pos)
            pos += len(a)
        return sorted(set(c for c in out if 0 < c < len(w)))

    def allomorph_candidates(self):
        return []


class FreqMorph(_SimpleMorph):
    """語末・語頭の頻出文字列（1〜3文字）の上位を接辞とみなす。"""

    def _fit(self):
        suf, pre = Counter(), Counter()
        for w in self.counts:
            for k in (1, 2, 3):
                if len(w) - k >= 3:
                    suf[w[-k:]] += 1
                    pre[w[:k]] += 1
        self.sufs = {a for a, _ in suf.most_common(12)}
        self.pres = {a for a, _ in pre.most_common(12)}
        top_s = sum(c for _, c in suf.most_common(12))
        top_p = sum(c for _, c in pre.most_common(12))
        self.side = "suffix" if top_s >= top_p else "prefix"

    def analyze(self, w):
        if w not in self.cache:
            res = ([], w, [])
            for k in (3, 2, 1):
                if len(w) - k < 3:
                    continue
                if self.side == "suffix" and w[-k:] in self.sufs:
                    res = ([], w[:-k], [w[-k:]])
                    break
                if self.side == "prefix" and w[:k] in self.pres:
                    res = ([w[:k]], w[k:], [])
                    break
            self.cache[w] = res
        return self.cache[w]


class SuccessorMorph(_SimpleMorph):
    """Harris の successor variety: 語頭からの文字列に続く文字の種類数が山になる所で切る（前後両方向）。"""

    def _fit(self):
        self.fwd, self.bwd = defaultdict(set), defaultdict(set)
        for w in self.counts:
            for i in range(1, len(w)):
                self.fwd[w[:i]].add(w[i])
                self.bwd[w[i:]].add(w[i - 1])

    def analyze(self, w):
        if w not in self.cache:
            cut_s = cut_p = None
            for i in range(3, len(w)):          # 接尾辞境界: 語幹 w[:i] の後続の多様性
                v = len(self.fwd[w[:i]])
                if v >= 4 and v > len(self.fwd[w[:i - 1]]) and (i + 1 >= len(w) or v >= len(self.fwd[w[:i + 1]])):
                    cut_s = i
                    break
            end = cut_s or len(w)
            for i in range(end - 3, 0, -1):     # 接頭辞境界
                v = len(self.bwd[w[i:]])
                if v >= 4 and v > len(self.bwd[w[i + 1:]]) and v >= len(self.bwd[w[i - 1:]]):
                    cut_p = i
                    break
            start = cut_p or 0
            self.cache[w] = ([w[:start]] if start else [], w[start:end], [w[end:]] if end < len(w) else [])
        return self.cache[w]


def make_morphology(method, counts):
    return {"freq": FreqMorph, "successor": SuccessorMorph}[method](counts)
