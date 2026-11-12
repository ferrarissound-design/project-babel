"""単語境界のない連続書記からの単語分割。

baseline: 文字遷移確率が低い所で切る（n-gram 法）。
main: 単語ユニグラムモデルの Viterbi 分割を反復再推定（初期値は baseline）。

どちらも「学習（fit）」と「適用（apply）」に分かれている。
- トランスダクティブ: 分割したい全文（train と heldout の両方）で学習して、その同じ文を分割する。
- インダクティブ: train の文だけで学習し、学習した頻度を凍結したまま heldout の文に適用する。
"""
from __future__ import annotations

import math
from collections import Counter


class TransitionModel:
    """文字の遷移確率と、切る位置のしきい値。"""

    def __init__(self, lines, quantile=0.3):
        self.big, self.uni = Counter(), Counter()
        for ln in lines:
            for a, b in zip(ln, ln[1:]):
                self.big[(a, b)] += 1
                self.uni[a] += 1
        probs = sorted(self.big[(a, b)] / self.uni[a] for ln in lines for a, b in zip(ln, ln[1:]))
        self.thr = probs[int(len(probs) * quantile)] if probs else None

    def prob(self, a, b):
        return self.big[(a, b)] / self.uni[a] if self.uni[a] else 0.0

    def apply(self, ln):
        if self.thr is None:
            return [ln]
        words, cur = [], ln[:1]
        for a, b in zip(ln, ln[1:]):
            if self.prob(a, b) <= self.thr and len(cur) >= 2:
                words.append(cur)
                cur = b
            else:
                cur += b
        if cur:
            words.append(cur)
        return words


def transition_segment(lines, quantile=0.3):
    """トランスダクティブ: 与えた全文で学習して、同じ全文を分割する（従来の動作）。"""
    m = TransitionModel(lines, quantile)
    if m.thr is None:
        return [[ln] for ln in lines]
    return [m.apply(ln) for ln in lines]


class UnigramModel:
    """単語ユニグラムの頻度（凍結できる）と、新語の基底確率（文字ユニグラム）。"""

    def __init__(self, cnt, chars, max_len=14, alpha=20.0):
        self.cnt, self.total = cnt, sum(cnt.values())
        self.chars, self.n_chars = chars, sum(chars.values())
        self.max_len, self.alpha = max_len, alpha

    def base(self, w):     # 長い語ほど不利。学習に出てこない文字は確率 0 にならないよう 1 回分を与える
        n = self.n_chars or 1
        return math.prod((self.chars[c] or 1) / n for c in w) * 0.5 ** len(w)

    def apply(self, ln):
        best = [0.0] + [-math.inf] * len(ln)
        back = [0] * (len(ln) + 1)
        for j in range(1, len(ln) + 1):
            for i in range(max(0, j - self.max_len), j):
                w = ln[i:j]
                p = (self.cnt.get(w, 0) + self.alpha * self.base(w)) / (self.total + self.alpha)
                sc = best[i] + math.log(p)
                if sc > best[j]:
                    best[j], back[j] = sc, i
        words, j = [], len(ln)
        while j > 0:
            words.append(ln[back[j]:j])
            j = back[j]
        return words[::-1]


def fit_unigram(lines, n_iter=6, max_len=14, alpha=20.0):
    """lines で反復再推定したユニグラムモデルと、lines の分割を返す。"""
    seg = transition_segment(lines)
    chars = Counter(c for ln in lines for c in ln)
    model = UnigramModel(Counter(), chars, max_len, alpha)
    for _ in range(n_iter):
        model = UnigramModel(Counter(w for s in seg for w in s), chars, max_len, alpha)
        new = [model.apply(ln) for ln in lines]
        if new == seg:
            break
        seg = new
    # 返すモデルは、返す分割（seg）から数えた頻度で凍結する（未知の文への適用に使う）
    return UnigramModel(Counter(w for s in seg for w in s), chars, max_len, alpha), seg


def unigram_segment(lines, n_iter=6, max_len=14, alpha=20.0):
    """トランスダクティブ: 与えた全文で学習して、同じ全文を分割する（従来の動作）。"""
    return fit_unigram(lines, n_iter, max_len, alpha)[1]


def inductive_segment(train_lines, test_lines, n_iter=6, max_len=14, alpha=20.0):
    """インダクティブ: train_lines だけで学習し、凍結したモデルで test_lines を分割する。"""
    model, train_seg = fit_unigram(train_lines, n_iter, max_len, alpha)
    return train_seg, [model.apply(ln) for ln in test_lines]


def inductive_transition(train_lines, test_lines, quantile=0.3):
    m = TransitionModel(train_lines, quantile)
    if m.thr is None:
        return [[ln] for ln in train_lines], [[ln] for ln in test_lines]
    return [m.apply(ln) for ln in train_lines], [m.apply(ln) for ln in test_lines]
