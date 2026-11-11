"""単語境界のない連続書記からの単語分割。

baseline: 文字遷移確率が低い所で切る（n-gram 法）。
main: 単語ユニグラムモデルの Viterbi 分割を反復再推定（初期値は baseline）。
"""
from __future__ import annotations

import math
from collections import Counter


def transition_segment(lines, quantile=0.3):
    big, uni = Counter(), Counter()
    for ln in lines:
        for a, b in zip(ln, ln[1:]):
            big[(a, b)] += 1
            uni[a] += 1
    probs = sorted(big[(a, b)] / uni[a] for ln in lines for a, b in zip(ln, ln[1:]))
    if not probs:
        return [[ln] for ln in lines]
    thr = probs[int(len(probs) * quantile)]
    out = []
    for ln in lines:
        words, cur = [], ln[:1]
        for a, b in zip(ln, ln[1:]):
            if big[(a, b)] / uni[a] <= thr and len(cur) >= 2:
                words.append(cur)
                cur = b
            else:
                cur += b
        if cur:
            words.append(cur)
        out.append(words)
    return out


def unigram_segment(lines, n_iter=6, max_len=14, alpha=20.0):
    seg = transition_segment(lines)
    chars = Counter(c for ln in lines for c in ln)
    n_chars = sum(chars.values())

    def base(w):   # 文字ユニグラムによる新語の基底確率（長い語ほど不利）
        return math.prod(chars[c] / n_chars for c in w) * 0.5 ** len(w)

    for _ in range(n_iter):
        cnt = Counter(w for s in seg for w in s)
        total = sum(cnt.values())
        new = []
        for ln in lines:
            best = [0.0] + [-math.inf] * len(ln)
            back = [0] * (len(ln) + 1)
            for j in range(1, len(ln) + 1):
                for i in range(max(0, j - max_len), j):
                    w = ln[i:j]
                    p = (cnt.get(w, 0) + alpha * base(w)) / (total + alpha)
                    sc = best[i] + math.log(p)
                    if sc > best[j]:
                        best[j], back[j] = sc, i
            words, j = [], len(ln)
            while j > 0:
                words.append(ln[back[j]:j])
                j = back[j]
            new.append(words[::-1])
        if new == seg:
            break
        seg = new
    return seg
