"""行の組の類似を仕込んだ文（正解つき）。実画像 IMG_4494 の転写と同じ規模（13 行、行の長さ 6〜13、異なり記号が多い）で作る。

- 背景の行: 語彙 V の Zipf 分布から独立に引く（行の間に仕込んだ関係はない）
- 仕込んだ組: 一方の行を写し、各位置を確率 1 − keep で別の記号に替え、ときどき1記号を抜くか足す（同じ枠で一部が変わる行）
- n_pairs = 0 の文は陰性対照（行の間に仕込んだ構造がない）

正解は「仕込んだ行の組」。仕込んでいない組が偶然似ることはある（正解の転写での検定が、それを有意とする率を別に測る）。
"""
from __future__ import annotations

import random


def zipf_sampler(vocab, s, rng):
    order = list(range(vocab))
    rng.shuffle(order)
    weights = [1.0 / (k + 1) ** s for k in range(vocab)]
    return lambda: rng.choices(order, weights)[0]


def planted_text(seed, n_rows=13, vocab=80, zipf=1.0, n_pairs=2, len_range=(6, 13), keep_range=(0.55, 0.85)):
    """戻り値: (行（記号番号のリスト）のリスト, 仕込んだ組 [(i, j)], 詳細)。記号番号は 0〜vocab-1。"""
    rng = random.Random(f"planted-{seed}-{n_rows}-{vocab}-{zipf}-{n_pairs}")
    draw = zipf_sampler(vocab, zipf, rng)
    rows = [[draw() for _ in range(rng.randint(*len_range))] for _ in range(n_rows)]
    slots = list(range(n_rows))
    rng.shuffle(slots)
    pairs, detail = [], []
    for k in range(n_pairs):
        i, j = slots[2 * k], slots[2 * k + 1]
        keep = rng.uniform(*keep_range)
        copy, kept = [], 0
        for x in rows[i]:
            if rng.random() < keep:
                copy.append(x)
                kept += 1
            else:
                y = draw()
                while y == x:
                    y = draw()
                copy.append(y)
        edit = rng.random()
        if edit < 0.2 and len(copy) > len_range[0]:
            del copy[rng.randrange(len(copy))]
        elif edit < 0.4 and len(copy) < len_range[1]:
            copy.insert(rng.randrange(len(copy) + 1), draw())
        rows[j] = copy
        pairs.append((min(i, j), max(i, j)))
        detail.append({"rows": [min(i, j), max(i, j)], "keep": round(keep, 3), "kept_positions": kept, "length": len(rows[i])})
    return rows, sorted(pairs), {"seed": seed, "vocab": vocab, "zipf": zipf, "n_pairs": n_pairs, "planted": detail}
