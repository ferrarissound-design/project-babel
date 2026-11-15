"""語形のクラス分けの探索（問題記録 #99）。

方式ごとに語幹 → グループを作り、正解の品詞と比べる（V 値・純度・グループ数・被覆率）。
品詞は採点のためだけに使い、クラス分けには渡さない。ここで使うデータ（実在言語 20、合成言語の開発シード）は開発用。
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from babel.decipher import classes
from babel.decipher.engine import Decipherer
from babel.experiments import real_languages as rl
from babel.generator.corpus import build_corpus
from babel.generator.languages import FAMILIES

UNGROUPED = "<none>"


# ---------------------------------------------------------------- 採点
def _h(counter):
    n = sum(counter.values())
    return -sum(c / n * math.log2(c / n) for c in counter.values() if c) if n else 0.0


def score_grouping(stem_group, infos, sents, word_pos, by_word=False):
    """トークン単位の V 値（Rosenberg & Hirschberg 2007）。グループのない語は1つの「なし」グループに入れる。

    by_word=True なら、stem_group は語形 → グループ（語形単位のクラス分け）。
    """
    cont = defaultdict(Counter)       # group -> pos -> tokens
    covered = total = 0
    for s in sents:
        for w in s:
            if w not in infos or w not in word_pos:
                continue
            pos = word_pos[w].most_common(1)[0][0]
            g = stem_group.get(w if by_word else infos[w][1], UNGROUPED)
            cont[g][pos] += 1
            total += 1
            covered += g != UNGROUPED
    by_pos = defaultdict(Counter)
    for g, c in cont.items():
        for p, n in c.items():
            by_pos[p][g] += n
    n = total
    hc = _h(Counter({p: sum(c.values()) for p, c in by_pos.items()}))
    hk = _h(Counter({g: sum(c.values()) for g, c in cont.items()}))
    hc_k = sum(sum(c.values()) / n * _h(c) for c in cont.values())
    hk_c = sum(sum(c.values()) / n * _h(c) for c in by_pos.values())
    hom = 1 - hc_k / hc if hc else 1.0
    com = 1 - hk_c / hk if hk else 1.0
    v = 2 * hom * com / (hom + com) if hom + com else 0.0
    purity = sum(max(c.values()) for g, c in cont.items() if g != UNGROUPED) / max(1, covered)
    return {"v_measure": round(v, 3), "homogeneity": round(hom, 3), "completeness": round(com, 3),
            "purity_grouped": round(purity, 3), "coverage": round(covered / max(1, total), 3),
            "groups": sum(1 for g in cont if g != UNGROUPED)}


# ---------------------------------------------------------------- 方式
def unionfind(d, min_shared=3, frac=0.1):
    """現行（min_shared=3, frac=0.1）と、併合を厳しくした変種。"""
    bases = defaultdict(set)
    for pre, stem, suf in d.infos.values():
        base = stem
        for a in suf:
            bases["S:" + a].add(base)
            base += a
        base = stem
        for a in reversed(pre):
            bases["P:" + a].add(base)
            base = a + base
    tags = sorted(bases)
    parent = {t: t for t in tags}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            shared = len(bases[a] & bases[b])
            if shared >= max(min_shared, frac * min(len(bases[a]), len(bases[b]))):
                parent[find(a)] = find(b)
    votes = defaultdict(Counter)
    for pre, stem, suf in d.infos.values():
        for t in ["P:" + a for a in pre] + ["S:" + a for a in suf]:
            votes[stem][find(t)] += 1
    return {s: sorted(v.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for s, v in votes.items()}


def _sig(info):
    pre, _, suf = info
    return "|".join(pre) + "_" + "|".join(suf)


def distributional(d, k=16, n_frequent=50, min_count=3, seed=0, iters=25):
    """語幹を、前後の語（頻出語はその語、他は接辞の形）と自分の接辞の形で特徴づけ、球面 k-means で分ける。"""
    cnt = Counter(w for s in d.sents for w in s)
    frequent = {w for w, _ in cnt.most_common(n_frequent)}

    def ctx(w):
        if w is None:
            return "#"
        return "W:" + w if w in frequent else "A:" + _sig(d.infos[w])

    feats = defaultdict(Counter)
    for s in d.sents:
        for i, w in enumerate(s):
            stem = d.infos[w][1]
            feats[stem]["L" + ctx(s[i - 1] if i else None)] += 1
            feats[stem]["R" + ctx(s[i + 1] if i < len(s) - 1 else None)] += 1
            feats[stem]["S" + _sig(d.infos[w])] += 2
    stem_tok = Counter()
    for w, c in cnt.items():
        stem_tok[d.infos[w][1]] += c

    def unit(v):
        nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {f: x / nrm for f, x in v.items()}

    vecs = {s: unit(v) for s, v in feats.items()}
    train = sorted(s for s in vecs if stem_tok[s] >= min_count)
    if len(train) < k:
        return {}
    rng = random.Random(seed)
    # k-means++ の初期化（コサイン距離）
    cents = [dict(vecs[rng.choice(train)])]
    while len(cents) < k:
        dist = [1 - max(sum(x * c.get(f, 0.0) for f, x in vecs[s].items()) for c in cents) for s in train]
        tot = sum(dist) or 1.0
        r, acc = rng.random() * tot, 0.0
        for s, dd in zip(train, dist):
            acc += dd
            if acc >= r:
                cents.append(dict(vecs[s]))
                break
    assign = {}
    for _ in range(iters):
        new = {}
        for s in train:
            v = vecs[s]
            new[s] = max(range(k), key=lambda j: (sum(x * cents[j].get(f, 0.0) for f, x in v.items()), -j))
        if new == assign:
            break
        assign = new
        sums = [Counter() for _ in range(k)]
        for s, j in assign.items():
            for f, x in vecs[s].items():
                sums[j][f] += x * stem_tok[s]
        cents = [unit(c) if c else cents[j] for j, c in enumerate(sums)]
    out = {s: f"D{j}" for s, j in assign.items()}
    for s in vecs:                     # 低頻度の語幹は、いちばん近い重心へ
        if s not in out:
            v = vecs[s]
            out[s] = f"D{max(range(k), key=lambda j: (sum(x * cents[j].get(f, 0.0) for f, x in v.items()), -j))}"
    return out


def distributional_words(d, k=32, n_frequent=200, min_count=3, seed=0, iters=25, use_affix=True):
    """語形単位: 前後の語（頻出語はその語、他は接辞の形）と自分の接辞の形で、語形を球面 k-means で分ける。"""
    cnt = Counter(w for s in d.sents for w in s)
    frequent = {w for w, _ in cnt.most_common(n_frequent)}

    def ctx(w):
        if w is None:
            return "#"
        return "W:" + w if w in frequent else ("A:" + _sig(d.infos[w]) if use_affix else "W:<rare>")

    feats = defaultdict(Counter)
    for s in d.sents:
        for i, w in enumerate(s):
            feats[w]["L" + ctx(s[i - 1] if i else None)] += 1
            feats[w]["R" + ctx(s[i + 1] if i < len(s) - 1 else None)] += 1
            if use_affix:
                feats[w]["S" + _sig(d.infos[w])] += 2
    return _kmeans(feats, cnt, k, min_count, seed, iters)


def _kmeans(feats, weight, k, min_count, seed, iters):
    def unit(v):
        nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {f: x / nrm for f, x in v.items()}

    vecs = {s: unit(v) for s, v in feats.items()}
    train = sorted(s for s in vecs if weight[s] >= min_count)
    if len(train) < k:
        return {}
    rng = random.Random(seed)
    cents = [dict(vecs[rng.choice(train)])]
    while len(cents) < k:
        dist = [1 - max(sum(x * c.get(f, 0.0) for f, x in vecs[s].items()) for c in cents) for s in train]
        tot = sum(dist) or 1.0
        r, acc = rng.random() * tot, 0.0
        for s, dd in zip(train, dist):
            acc += dd
            if acc >= r:
                cents.append(dict(vecs[s]))
                break
    assign = {}
    for _ in range(iters):
        new = {s: max(range(k), key=lambda j: (sum(x * cents[j].get(f, 0.0) for f, x in vecs[s].items()), -j))
               for s in train}
        if new == assign:
            break
        assign = new
        sums = [Counter() for _ in range(k)]
        for s, j in assign.items():
            for f, x in vecs[s].items():
                sums[j][f] += x * weight[s]
        cents = [unit(c) if c else cents[j] for j, c in enumerate(sums)]
    out = {s: f"D{j}" for s, j in assign.items()}
    for s in vecs:
        if s not in out:
            v = vecs[s]
            out[s] = f"D{max(range(k), key=lambda j: (sum(x * cents[j].get(f, 0.0) for f, x in v.items()), -j))}"
    return out


WORD_METHODS = {"wdist_k16": lambda d: distributional_words(d, k=16),
                "wdist_k32": lambda d: distributional_words(d, k=32),
                "wdist_k32_noaffix": lambda d: distributional_words(d, k=32, use_affix=False)}

METHODS = {
    "current": lambda d: classes.paradigm_groups(d.infos)[1],
    "uf_3_0.3": lambda d: unionfind(d, 3, 0.3),
    "uf_5_0.5": lambda d: unionfind(d, 5, 0.5),
    "dist_k8": lambda d: distributional(d, k=8),
    "dist_k16": lambda d: distributional(d, k=16),
    "dist_k32": lambda d: distributional(d, k=32),
}


# ---------------------------------------------------------------- データ
def _norm_pos(dec, word_pos):
    out = defaultdict(Counter)
    for w, c in word_pos.items():
        out[dec.nz.norm(w)[0]].update(c)
    return out


def corpora():
    for fam in FAMILIES:
        for seed in range(5):
            public, secret = build_corpus(fam, seed, "C")
            wp = defaultdict(Counter)
            for d in public["documents"]:
                for si, sent in enumerate(d["sentences"]):
                    for w, t in zip(sent.split(), secret["tokens"][d["id"]][si]):
                        wp[w][t["pos"]] += 1
            yield f"{fam}-{seed}-C", "synthetic", public, wp
    for repo, code in rl.TREEBANKS + rl.TREEBANKS_HELDOUT:
        path, _ = rl.fetch(repo, code)
        sents = rl.read_sentences(path)
        words, _ = rl.cipher(sents)
        wp = defaultdict(Counter)
        for s, ws in zip(sents, words):
            for (_, u, _), c in zip(s, ws):
                wp[c][u] += 1
        yield code, "real", rl.to_public(f"real-{code}", words), wp


def main(out="results/grouping/study-dev.json"):
    rows = []
    for name, kind, public, wp in corpora():
        dec = Decipherer(public).fit()
        npos = _norm_pos(dec, wp)
        row = {"corpus": name, "kind": kind, "methods": {}}
        for m, fn in METHODS.items():
            row["methods"][m] = score_grouping(fn(dec), dec.infos, dec.sents, npos)
        for m, fn in WORD_METHODS.items():
            row["methods"][m] = score_grouping(fn(dec), dec.infos, dec.sents, npos, by_word=True)
        rows.append(row)
        print(name, {m: r["v_measure"] for m, r in row["methods"].items()}, flush=True)
    summary = {}
    for kind in ("synthetic", "real"):
        rs = [r for r in rows if r["kind"] == kind]
        summary[kind] = {m: {k: round(sum(r["methods"][m][k] for r in rs) / len(rs), 3)
                             for k in ("v_measure", "homogeneity", "completeness", "purity_grouped", "coverage", "groups")}
                         for m in list(METHODS) + list(WORD_METHODS)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"note": "開発用（結果を見たデータ）", "summary": summary, "rows": rows},
                                    ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def verb_clusters(out="results/grouping/verb-clusters-dev.json"):
    """分布のクラス（語形単位・前後の語のみ・k=32）で、動詞が過半数のクラスと、現行の動詞の選び方が選ぶクラス。"""
    rows = []
    for name, kind, public, wp in corpora():
        dec = Decipherer(public).fit()
        npos = _norm_pos(dec, wp)
        vl = "V" if kind == "synthetic" else "VERB"
        g = WORD_METHODS["wdist_k32_noaffix"](dec)
        st = defaultdict(lambda: {"tokens": 0, "verb": 0, "sentences": 0, "once": 0})
        for sent in dec.sents:
            k = Counter(g.get(w) for w in sent)
            for c, n in k.items():
                st[c]["sentences"] += 1
                st[c]["once"] += n == 1
            for w in sent:
                st[g.get(w)]["tokens"] += 1
                st[g.get(w)]["verb"] += w in npos and npos[w].most_common(1)[0][0] == vl
        ns = len(dec.sents)
        cl = [{"verb_share": x["verb"] / x["tokens"], "tokens": x["tokens"], "coverage": x["sentences"] / ns,
               "one_per_sentence": x["once"] / max(1, x["sentences"])} for x in st.values()]
        total_verb = sum(x["verb"] for x in st.values())
        verbish = [c for c in cl if c["verb_share"] >= 0.5]
        cands = [c for c in cl if c["coverage"] >= 0.3]
        pick = max(cands, key=lambda c: c["one_per_sentence"]) if cands else None
        rows.append({"corpus": name, "kind": kind, "verb_majority_clusters": len(verbish),
                     "verb_tokens_in_verb_majority_clusters": round(
                         sum(c["tokens"] * c["verb_share"] for c in verbish) / max(1, total_verb), 3),
                     "selected_verb_share": round(pick["verb_share"], 3) if pick else None})
        print(rows[-1], flush=True)
    Path(out).write_text(json.dumps({"note": "開発用", "rows": rows}, ensure_ascii=False, indent=1) + "\n",
                         encoding="utf-8")


SIZE_TREEBANKS = [("UD_Spanish-AnCora", "es_ancora"), ("UD_Czech-CAC", "cs_cac"), ("UD_Turkish-IMST", "tr_imst"),
                  ("UD_English-EWT", "en_ewt"), ("UD_Estonian-EDT", "et_edt")]


def size_effect(out="results/grouping/size-dev.json"):
    """文の数を 1000 → 4000 にしたときの V 値（データ量が足りないことが原因かの確認）。"""
    rows = []
    for repo, code in SIZE_TREEBANKS:
        path, _ = rl.fetch(repo, code)
        for n in (1000, 4000):
            sents = rl.read_sentences(path, n)
            try:
                words, _ = rl.cipher(sents)
            except ValueError as e:              # 文字の種類が換字の記号数を超える
                rows.append({"corpus": code, "sentences": len(sents), "skipped": str(e)})
                continue
            wp = defaultdict(Counter)
            for sent, ws in zip(sents, words):
                for (_, u, _), c in zip(sent, ws):
                    wp[c][u] += 1
            dec = Decipherer(rl.to_public(f"real-{code}", words)).fit()
            npos = _norm_pos(dec, wp)
            rows.append({"corpus": code, "sentences": len(sents), "tokens": sum(map(len, sents)),
                         "current": score_grouping(METHODS["current"](dec), dec.infos, dec.sents, npos),
                         "wdist_k32_noaffix": score_grouping(WORD_METHODS["wdist_k32_noaffix"](dec), dec.infos,
                                                             dec.sents, npos, by_word=True)})
            print(rows[-1]["corpus"], n, rows[-1]["current"]["v_measure"], rows[-1]["wdist_k32_noaffix"]["v_measure"],
                  flush=True)
    Path(out).write_text(json.dumps({"note": "開発用", "rows": rows}, ensure_ascii=False, indent=1) + "\n",
                         encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:2] == ["size"]:
        size_effect()
    elif sys.argv[1:2] == ["verbs"]:
        verb_clusters()
    else:
        main(*sys.argv[1:2])
