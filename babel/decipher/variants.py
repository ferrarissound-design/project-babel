"""表記揺れ・音変化の正規化。

1か所だけ異なる語形ペアを集める。同じ置換 (x↔y) を示すペアのうち、検定可能なものの多くが
「同じ文書にほとんど共起しない」（＝書き手や時代で使い分けられている）なら、その置換を
表記上の交替とみなす。形態論的な交替（格語尾など）は同じ文書内で共起するので除外される。
統合は語形ペア単位で行う（置換を全語に機械的に適用すると、偶然の最小対を潰してしまうため）。
"""
from __future__ import annotations

import difflib
import math
from collections import Counter, defaultdict

LACUNA = "[…]"


def _diff(w1, w2):
    i = 0
    while i < min(len(w1), len(w2)) and w1[i] == w2[i]:
        i += 1
    j = 0
    while j < min(len(w1), len(w2)) - i and w1[-1 - j] == w2[-1 - j]:
        j += 1
    return w1[i:len(w1) - j], w2[i:len(w2) - j], i == 0, j == 0


_EMIN: dict = {}


def _emin(n, p):
    key = (n, round(p, 2))
    if key not in _EMIN:
        q = key[1]
        _EMIN[key] = sum(math.comb(n, k) * q ** k * (1 - q) ** (n - k) * min(k, n - k) for k in range(n + 1))
    return _EMIN[key]


class Normalizer:
    def __init__(self, docs=None, enable=True, ratio=0.3):
        self.canon: dict[str, str] = {}
        self.ops: list[dict] = []
        self.debug: list = []
        self.cache: dict = {}
        self.freq = Counter()
        if docs is None or not enable:
            return
        cnt = defaultdict(Counter)
        for i, toks in enumerate(docs):
            for t in toks:
                if t != LACUNA:
                    self.freq[t] += 1
                    cnt[t][i] += 1

        def stat(w1, w2):
            """文書内での混在度。観測値と、2形が無作為に混ざる場合の期待値を返す。"""
            c1, c2 = cnt[w1], cnt[w2]
            p = self.freq[w1] / (self.freq[w1] + self.freq[w2])
            obs = exp = 0.0
            for d in set(c1) | set(c2):
                n = min(c1[d] + c2[d], 12)
                if n >= 2:
                    obs += min(c1[d], c2[d])
                    exp += _emin(n, p)
            return obs, exp

        buckets = defaultdict(set)
        for w in self.freq:
            for i in range(len(w) + 1):
                for ln in (0, 1, 2):
                    if i + ln <= len(w):
                        buckets[(w[:i], w[i + ln:])].add(w)
        pairs = set()
        for ws in buckets.values():
            if 2 <= len(ws) <= 40:
                ws = sorted(ws)
                pairs.update((a, b) for i, a in enumerate(ws) for b in ws[i + 1:])
        by_op = defaultdict(list)
        for w1, w2 in sorted(pairs):
            x, y, ini, fin = _diff(w1, w2)
            if len(x) > 2 or len(y) > 2 or not (x or y):
                continue
            pos = "final" if fin else ("initial" if ini else "mid")
            if x and y:
                by_op[(x, y)].append((w1, w2, pos))
            else:   # 挿入・脱落: そのままの形と、左文脈1文字を付けた形（g↔gh など）の両方で登録する
                if x:
                    x, y, w1, w2 = y, x, w2, w1
                by_op[(x, y)].append((w1, w2, pos))
                k = len(w2) - len(_diff(w1, w2)[1]) if fin else w2.find(y, len(w1) - len(w1.lstrip()) if False else 0)
                i = len(w1) if fin else next(i for i in range(len(w1) + 1) if w1[:i] + y + w1[i:] == w2)
                del k
                if i > 0:
                    by_op[(w1[i - 1], w1[i - 1] + y)].append((w1, w2, pos))
        n_chars = [sum(len(t) for t in toks if t != LACUNA) for toks in docs]

        def absent_docs(a, b, where):
            """文字列 a が（期待されるのに）現れない文書の集合。b の一部として現れる a は数えない。"""
            counts = []
            for toks in docs:
                c = 0
                for t in toks:
                    if t == LACUNA:
                        continue
                    if where == "final":
                        c += t.endswith(a) and not (a in b and t.endswith(b))
                    elif where == "initial":
                        c += t.startswith(a) and not (a in b and t.startswith(b))
                    else:
                        c += t.count(a) - (t.count(b) if a in b else 0)
                counts.append(c)
            used = [i for i in range(len(docs)) if counts[i] > 0]
            rate = sum(counts) / max(1, sum(n_chars[i] for i in used))    # a を使う文書での出現率
            testable = [i for i in range(len(docs)) if rate * n_chars[i] >= 3]
            lack = {i for i in testable if counts[i] <= 0.2 * rate * n_chars[i]}
            return lack, len(testable)

        for (x, y), ps in sorted(by_op.items()):
            if len(ps) < 3:
                continue
            kinds = {p[2] for p in ps}
            where = kinds.pop() if len(kinds) == 1 and "mid" not in kinds else "any"
            if where != "any" and not ((len(x) <= 1 and len(y) <= 1) or (x and x in y) or (y and y in x)):
                continue        # 語端だけの長い交替は接辞の交替（時制語尾など）の可能性が高い
            best = None
            for a, b, flip in ((x, y, False), (y, x, True)):
                if not a:
                    continue
                lack, n_test = absent_docs(a, b, where)
                if n_test < 10 or len(lack) < max(3, 0.08 * n_test):
                    continue
                # a→b の規則なら、b を含む語形は a が消えた文書に集中し、a を含む語形はそこに現れない
                in_b = tot_b = in_a = tot_a = 0
                for w1, w2, _ in ps:
                    wa, wb = (w2, w1) if flip else (w1, w2)
                    for d, c in cnt[wb].items():
                        tot_b += c
                        in_b += c * (d in lack)
                    for d, c in cnt[wa].items():
                        tot_a += c
                        in_a += c * (d in lack)
                self.debug.append((a, b, where, len(ps), n_test, len(lack), round(in_b / max(1, tot_b), 2),
                                   round(in_a / max(1, tot_a), 2)))
                if tot_b and tot_a and in_b / tot_b >= 0.7 and in_a / tot_a <= 0.15:
                    cand = {"a": a, "b": b, "where": where, "pairs": len(ps), "docs_without_a": len(lack),
                            "b_in_those_docs": round(in_b / tot_b, 3), "a_in_those_docs": round(in_a / tot_a, 3),
                            "examples": [list(p[:2]) for p in ps[:3]]}
                    if best is None or cand["b_in_those_docs"] > best["b_in_those_docs"]:
                        best = cand
            if best:
                self.ops.append(best)
        # 受理した交替は「全文書に規則を適用した形」に書き換える（対になる語形が未出現でも一貫する）
        self.ops = [o for o in self.ops if o["pairs"] >= 6]
        self.ops.sort(key=lambda o: (-len(o["a"]), -o["pairs"], o["a"]))
        for w in self.freq:
            self.canon[w] = self._rewrite(w)

    def _rewrite(self, w):
        for op in self.ops:
            a, b = op["a"], op["b"]
            if op["where"] == "final":
                if w.endswith(a) and len(w) > len(a):
                    w = w[: -len(a)] + b
            elif op["where"] == "initial":
                if w.startswith(a) and len(w) > len(a):
                    w = b + w[len(a):]
            elif len(w.replace(a, b)) >= 2:
                w = w.replace(a, b)
        return w

    def norm(self, word: str):
        """(正規化形, 正規化形の各位置→元の語の位置) を返す。"""
        if word in self.cache:
            return self.cache[word]
        c = self.canon.get(word) or self._rewrite(word)
        if c == word:
            res = (word, list(range(len(word) + 1)))
        else:
            mp = [0] * (len(c) + 1)
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, c, word, autojunk=False).get_opcodes():
                for k in range(i1, i2):
                    mp[k] = j1 + (k - i1) if tag == "equal" else min(j1 + (k - i1), j2)
            mp[len(c)] = len(word)
            res = (c, mp)
        self.cache[word] = res
        return res
