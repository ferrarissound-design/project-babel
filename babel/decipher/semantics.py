"""意味関係の推定（分布的類似）と、メタデータ（出土地）に基づく地名仮説。"""
from __future__ import annotations

import math
from collections import Counter, defaultdict


def relatedness(sent_stems, min_freq=5, topk=3, role_sensitive=True):
    """sent_stems: 文ごとの [(語幹, 接辞署名)]。語幹ごとに分布の近い語幹を返す。

    特徴は「同じ文に現れる他の語幹 × 自分の接辞署名」。同じ動詞の同じ格の項になる名詞同士が近くなる。
    """
    freq = Counter(s for sent in sent_stems for s, _ in sent)
    vec = defaultdict(Counter)
    feat_freq = Counter()
    for sent in sent_stems:
        for i, (s, sig) in enumerate(sent):
            if freq[s] < min_freq:
                continue
            for j, (o, _) in enumerate(sent):
                if i != j and o != s:
                    f = (o, sig) if role_sensitive else o
                    vec[s][f] += 1
                    feat_freq[f] += 1
    total = max(1, sum(feat_freq.values()))
    ppmi = {}
    for s, row in vec.items():
        tot = sum(row.values())
        w = {}
        for f, c in row.items():
            v = math.log((c / tot) / (feat_freq[f] / total))
            if v > 0:
                w[f] = v
        ppmi[s] = (w, math.sqrt(sum(v * v for v in w.values())) or 1.0)
    out = {}
    keys = sorted(ppmi)
    for s in keys:
        w, norm = ppmi[s]
        sims = []
        for o in keys:
            if o == s:
                continue
            w2, n2 = ppmi[o]
            small, big = (w, w2) if len(w) < len(w2) else (w2, w)
            dot = sum(v * big[f] for f, v in small.items() if f in big)
            if dot > 0:
                sims.append((dot / (norm * n2), o))
        sims.sort(reverse=True)
        out[s] = [[o, round(v, 3)] for v, o in sims[:topk]]
    return out


def toponym_hypotheses(doc_stems, doc_site, eligible, min_docs=3):
    """出土地ごとに、その地の文書に偏って現れる語幹を地名候補とする。

    doc_stems: 文書ごとの語幹集合、doc_site: 文書ごとの出土地、eligible(stem): 候補にしてよい語幹か。
    地元の人名も同じ偏りを示すため、次点との差で信頼度を割り引く。
    """
    df, df_site = Counter(), defaultdict(Counter)
    n_site = Counter(doc_site)
    for stems, site in zip(doc_stems, doc_site):
        for s in stems:
            df[s] += 1
            df_site[site][s] += 1
    out = {}
    for site in sorted(n_site):
        cands = []
        for s, c in df_site[site].items():
            if c >= min_docs and eligible(s):
                prec = c / df[s]
                cover = c / n_site[site]
                if prec >= 0.5:
                    cands.append((prec * math.sqrt(cover), prec, cover, s))
        cands.sort(reverse=True)
        if cands:
            top = cands[0]
            second = cands[1][0] if len(cands) > 1 else 0.0
            conf = top[1] * (0.5 + 0.5 * (1 - second / top[0])) if top[0] > 0 else 0.0   # 次点と僅差なら半減
            out[site] = {"stem": top[3], "confidence": round(min(0.95, conf), 3),
                         "precision": round(top[1], 3), "coverage": round(top[2], 3),
                         "alternatives": [[c[3], round(c[1], 3)] for c in cands[1:3]]}
    return out
