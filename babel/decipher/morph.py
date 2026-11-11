"""形態素の推定（パラダイム法）。

考え方: 接辞は「少数の短い形が多数の語幹に付く」閉じたクラスで、語幹は「多数の形が少数の接辞を取る」
開いたクラスである。語の各分割点について、語幹候補が取る語尾集合の質を比べ、最も多くの
生産的な語尾を説明する分割を選ぶ。接尾辞側と接頭辞側（語を反転して同じ処理）を独立に行う。
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import median

MIN_STEM, MAX_AFF, MIN_P, R_MIN = 3, 7, 3, 0.2


class Side:
    """片側（接尾辞側）の解析。接頭辞側は反転した語彙を渡す。"""

    def __init__(self, vocab):
        self.vocab = set(vocab)
        ends = defaultdict(set)
        nend = Counter()
        for w in self.vocab:
            for k in range(MIN_STEM, len(w) + 1):
                if len(w) - k <= MAX_AFF:
                    ends[w[:k]].add(w[k:])
                    if k < len(w):
                        nend[w[k:]] += 1
        self.ends = {s: e for s, e in ends.items() if len(e) >= 2}
        stems_of = defaultdict(list)
        for s, e in self.ends.items():
            for f in e:
                stems_of[f].append(s)
        self.P = {f: len(v) for f, v in stems_of.items()}
        self.nend = nend
        self.valid = set()
        for f, stems in stems_of.items():
            r = self.P[f] / max(1, nend[f])
            if not f or self.P[f] < MIN_P or r < R_MIN or (len(f) == 1 and (r < 0.5 or self.P[f] < 8)):
                continue
            if self.P[f] >= median(len(self.ends[t]) for t in stems):
                self.valid.add(f)
        self.stems_of = {f: set(v) for f, v in stems_of.items()}
        self.reliable: set = set()
        self.bigrams: set = set()
        self._refresh()
        # 2巡目: 支持した語幹の多くが実際にはその接辞を選ばなかった場合（入れ子の断片など）は捨てる
        use = Counter()
        for w in self.vocab:
            for a in self.split(w)[1]:
                use[a] += 1
        keep = {f for f in self.valid if use[f] >= MIN_P and use[f] >= 0.5 * self.P[f]}
        self.valid = {f for f in self.valid if all(a in keep for a in self._parts(f))}
        self._refresh()
        self.use, outer = Counter(), Counter()
        for w in self.vocab:
            chain = self.split(w)[1]
            for a in chain:
                self.use[a] += 1
            if chain:
                outer[chain[-1]] += 1
            self.bigrams.update(zip(chain, chain[1:]))
        # 1形しか現れない語幹にも適用できる、信頼できる接辞
        self.reliable = {f for f in self.use if outer[f] >= 5 and outer[f] / max(1, nend[f]) >= 0.6}
        self._cache.clear()

    def _refresh(self):
        self.good = {s: {f for f in e if f in self.valid or f == ""} for s, e in self.ends.items()}
        self.Q = {s: sum(math.log(self.P[f]) for f in g) for s, g in self.good.items() if len(g) >= 2}
        self._cache: dict = {}

    def _parts(self, f):
        """複合語尾を既知の接辞の連鎖に分解する（例: lar+da）。"""
        best = [f]
        for i in range(1, len(f)):
            a, b = f[:i], f[i:]
            if a not in self.valid or b not in self.valid or self.P[b] < self.P[f]:
                continue
            takers = self.stems_of.get(f, ())
            if sum(1 for t in takers if t in self.stems_of.get(a, ())) >= 0.5 * len(takers):
                cand = self._parts(a) + self._parts(b)
                if len(cand) > len(best):
                    best = cand
        return best

    def split(self, w, top=True):
        """(語幹, 接辞の連鎖[内側→外側]) を返す。"""
        if w in self._cache:
            return self._cache[w]
        best, bq = None, -1.0
        for k in range(MIN_STEM, len(w)):
            s, f = w[:k], w[k:]
            if len(f) > MAX_AFF or f not in self.valid or s not in self.Q or f not in self.good[s]:
                continue
            if self.Q[s] > bq:
                best, bq = k, self.Q[s]
        if best is not None and self.Q.get(w, -1.0) >= bq:
            best = None         # 語全体が語幹として、より多くの語尾を説明する（切らない方がよい）
            self._cache[w] = (w, [])
            return self._cache[w]
        if best is None:
            res = (w, [])
            if top:     # 1形しか現れない語幹: 信頼できる接辞なら切る。内側は実例のある連鎖だけ許す
                for f in sorted(self.reliable, key=len, reverse=True):
                    if w.endswith(f) and len(w) - len(f) >= MIN_STEM + 1:
                        s, chain = self.split(w[: -len(f)], top=False)
                        while not chain:
                            g = next((g for g in sorted(self.reliable, key=len, reverse=True)
                                      if (g, f) in self.bigrams and s.endswith(g) and len(s) - len(g) >= MIN_STEM + 1), None)
                            if g is None:
                                break
                            s, chain = s[: -len(g)], [g]
                        res = (s, chain + [f])
                        break
                self._cache[w] = res
            return res
        stem, chain = self.split(w[:best], top=False)
        res = (stem, chain + self._parts(w[best:]))
        self._cache[w] = res
        return res


class Morphology:
    def __init__(self, vocab_counts: dict[str, int], enable: bool = True):
        self.counts = dict(vocab_counts)
        self.enable = enable
        self.cache: dict = {}
        if not enable:
            self.affixes, self.position, self.compounds = {}, "none", {}
            return
        # 接尾辞側・接頭辞側をそれぞれ全語形で解析し、支持の大きい側を先に確定する。
        # （接頭辞型の言語で接尾辞側を先に走らせると、語根が「語尾」に見えてしまうため）
        suf0, pre0 = Side(self.counts), Side({w[::-1] for w in self.counts})
        n_s = sum(c for w, c in self.counts.items() if suf0.split(w)[1])
        n_p = sum(c for w, c in self.counts.items() if pre0.split(w[::-1])[1])
        self.first = "suffix" if n_s >= n_p else "prefix"
        if self.first == "suffix":
            self.suf = suf0
            self.pre = Side({self.suf.split(w)[0][::-1] for w in self.counts})
        else:
            self.pre = pre0
            self.suf = Side({self.pre.split(w[::-1])[0][::-1] for w in self.counts})
        self.compounds = {}
        n_suf = sum(c for w, c in self.counts.items() if self.analyze(w)[2])
        n_pre = sum(c for w, c in self.counts.items() if self.analyze(w)[0])
        self.support = {"suffix": n_suf, "prefix": n_pre}
        lo, hi = sorted([n_suf, n_pre])
        self.position = "none" if hi == 0 else ("both" if lo >= 0.6 * hi else ("suffix" if n_suf > n_pre else "prefix"))
        self._find_compounds()
        self.affixes = {}
        for w, c in self.counts.items():
            p, s, x = self.analyze(w)
            for a in p:
                self.affixes.setdefault("P:" + a, [0, 0])
                self.affixes["P:" + a][0] += 1
                self.affixes["P:" + a][1] += c
            for a in x:
                self.affixes.setdefault("S:" + a, [0, 0])
                self.affixes["S:" + a][0] += 1
                self.affixes["S:" + a][1] += c

    def analyze(self, w):
        """(接頭辞[外→内], 語幹, 接尾辞[内→外])"""
        if not self.enable:
            return ([], w, [])
        if w in self.cache:
            return self.cache[w]
        if self.first == "suffix":
            s1, sufs = self.suf.split(w)
            r, pres = self.pre.split(s1[::-1])
            core = r[::-1]
            if len(core) < MIN_STEM:
                core, pres = s1, []
        else:
            r, pres = self.pre.split(w[::-1])
            core, sufs = self.suf.split(r[::-1])
            if len(core) < MIN_STEM:
                core, sufs = r[::-1], []
        res = ([p[::-1] for p in reversed(pres)], core, sufs)
        self.cache[w] = res
        return res

    def _find_compounds(self):
        """語幹が既知の語幹2つの連結（1文字の連結辞を許す）なら複合語とみなす。"""
        stem_forms = defaultdict(set)
        stem_freq = Counter()
        for w, c in self.counts.items():
            p, s, x = self.analyze(w)
            stem_forms[s].add((tuple(p), tuple(x)))
            stem_freq[s] += c
        strong = {s for s in stem_forms if len(s) >= 3 and (len(stem_forms[s]) >= 2 or stem_freq[s] >= 4)}
        for s in sorted(stem_forms):
            if len(s) < 6:
                continue
            best = None
            for i in range(3, len(s) - 2):
                for link in (0, 1):
                    a, b = s[:i], s[i + link:]
                    if a in strong and b in strong and a != s and b != s:
                        score = min(stem_freq[a], stem_freq[b]) - link
                        if best is None or score > best[0]:
                            best = (score, a, b)
            if best:
                self.compounds[s] = (best[1], best[2])

    def cuts(self, w):
        """語内の形態素境界位置（文字インデックス）。"""
        p, s, x = self.analyze(w)
        out, pos = [], 0
        for a in p:
            pos += len(a)
            out.append(pos)
        pos += len(s)
        if s in self.compounds:
            out.append(pos - len(self.compounds[s][1]))
        for a in x:
            out.append(pos)
            pos += len(a)
        return sorted(set(c for c in out if 0 < c < len(w)))

    def allomorph_candidates(self):
        """1文字だけ異なり、同じ語幹に決して付かない接辞の組（異形態の候補）。"""
        by = defaultdict(set)
        for w in self.counts:
            p, s, x = self.analyze(w)
            for a in x:
                by["S:" + a].add(s)
            for a in p:
                by["P:" + a].add(s)
        keys = sorted(k for k in by if len(by[k]) >= 4)
        out = []
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if a[0] != b[0] or len(a) != len(b):
                    continue
                if sum(1 for u, v in zip(a, b) if u != v) == 1 and not (by[a] & by[b]):
                    out.append([a, b])
        return out
