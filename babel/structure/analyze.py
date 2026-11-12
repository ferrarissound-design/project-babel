"""構造解析の本体: python -m babel analyze corpus.json --out analysis.json

手順
1. 記述統計（頻度・n-gram・最長反復・接頭部と接尾部・前後の文脈・共起・位置・行の類似度）
2. 6つの仮説の族を置換検定（帰無モデルと多重検定の扱いは stats.py の説明を参照）
3. 単語境界の候補（boundaries.py）と、最上位の分割に依存する接辞の検定
4. 安定性: (a) 行の部分標本（70% を非復元抽出）で同じ仮説が再現されるか = 標本の揺れ
           (b) 転写を攪乱（誤分類・分割・結合）しても再現されるか = 画像認識・転写の誤差
   (a) と (b) は別々に報告し、混同しない
5. 判定: 多重検定を通り、支持する観測が最小数以上のものだけを「仮説」とする。
   通らなかったものは「記述的観察」として少数だけ並べる（偶然と区別できない、と明記）。
"""
from __future__ import annotations

import math
import random
import time
from collections import Counter, defaultdict

from . import VERSION
from . import boundaries as bd
from .corpus import Corpus
from .hypotheses import INTERPRETATIONS, KIND_TEXT, NEEDED, Builder, components, templates
from .stats import FAMILIES, longest_repeat, ngrams, repeated_ngrams, run_tests, surprise

DEFAULTS = {
    "alpha": 0.05,
    "permutations": None,          # None: コーパスの大きさで自動（下の _auto）
    "max_n": 8,
    "max_gap": 8,
    "families": [f.name for f in FAMILIES] + ["affix"],
    "subsamples": 10,              # 標本の揺れ（行の部分標本）
    "subsample_fraction": 0.7,
    "subsample_permutations": 99,
    "perturbations": 6,            # 転写の誤差への感度（0 で無効）
    "perturbation_rate": 0.05,
    "perturbation_permutations": 199,
    "compression_reps": 19,
    "affix_permutations": 199,
    "stable_threshold": 0.5,
    "max_descriptive": 12,
    "seed": 0,
}

SHORT_LINES, SHORT_SYMBOLS = 30, 600     # これ以下を「短いコーパス」として扱う


def _auto(cfg, corpus):
    cfg = dict(cfg)
    n = corpus.n_symbols
    if cfg["permutations"] is None:
        cfg["permutations"] = 999 if n <= 800 else 499 if n <= 2000 else 199
    return cfg


# ------------------------------------------------------------------------------------------------ 記述統計

def describe(corpus: Corpus, max_n=8, top=15):
    lines = corpus.lines
    freq = Counter(s for ln in lines for s in ln)
    N = sum(freq.values())
    lens = sorted(len(ln) for ln in lines)
    rep = repeated_ngrams(lines, 2, max_n)
    by_n = defaultdict(list)
    for g, k in rep.items():
        by_n[len(g)].append((k, g))
    lrs_len, lrs = longest_repeat(lines)
    # 接頭部・接尾部を共有する行（2記号以上）
    pre, suf = Counter(), Counter()
    for ln in lines:
        for n in range(2, min(4, len(ln)) + 1):
            pre[ln[:n]] += 1
            suf[ln[-n:]] += 1
    # 前後の文脈の多様さ（後続・先行の異なり数）
    succ, prec = defaultdict(set), defaultdict(set)
    for ln in lines:
        for a, b in zip(ln, ln[1:]):
            succ[a].add(b)
            prec[b].add(a)
    # 位置（行内の相対位置の平均）
    relpos = defaultdict(list)
    for ln in lines:
        for i, s in enumerate(ln):
            relpos[s].append(i / (len(ln) - 1) if len(ln) > 1 else 0.5)
    # 共起
    cooc = Counter()
    for ln in lines:
        ss = sorted(set(ln))
        for a in range(len(ss)):
            for b in range(a + 1, len(ss)):
                cooc[(ss[a], ss[b])] += 1
    # 行の類似度（2記号列の Jaccard）
    sets = [set(ngrams(ln, 2)) for ln in lines]
    sims = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            u = len(sets[i] | sets[j])
            if u:
                v = len(sets[i] & sets[j]) / u
                if v > 0:
                    sims.append((v, i, j))
    sims.sort(reverse=True)
    docs = defaultdict(list)
    for li, info in enumerate(corpus.line_info):
        docs[info.get("doc")].append(li)
    doc_sim = []
    dk = sorted(docs, key=str)
    dsets = {d: set(g for li in docs[d] for g in ngrams(lines[li], 2)) for d in dk}
    for a in range(len(dk)):
        for b in range(a + 1, len(dk)):
            u = len(dsets[dk[a]] | dsets[dk[b]])
            if u:
                doc_sim.append((len(dsets[dk[a]] & dsets[dk[b]]) / u, dk[a], dk[b]))
    doc_sim.sort(reverse=True)
    nm = corpus.names
    return {
        "lines": len(lines), "documents": len(docs), "symbols_total": N, "symbol_types": len(freq),
        "type_token_ratio": round(len(freq) / N, 3) if N else None,
        "hapax_types": sum(1 for v in freq.values() if v == 1),
        "mean_count_per_type": round(N / len(freq), 2) if freq else None,
        "line_length": {"min": lens[0] if lens else 0, "median": lens[len(lens) // 2] if lens else 0,
                        "max": lens[-1] if lens else 0},
        "top_symbols": [{"symbol": corpus.name(s), "count": k} for s, k in freq.most_common(top)],
        "repeated_ngrams": {str(n): {"types": len(v), "top": [{"symbols": nm(g), "count": k}
                                                               for k, g in sorted(v, key=lambda x: (-x[0], x[1]))[:8]]}
                            for n, v in sorted(by_n.items())},
        "longest_repeat": {"length": lrs_len, "symbols": nm(lrs) if lrs else []},
        "shared_line_prefixes": [{"symbols": nm(g), "lines": k} for g, k in pre.most_common(8) if k >= 2],
        "shared_line_suffixes": [{"symbols": nm(g), "lines": k} for g, k in suf.most_common(8) if k >= 2],
        "context_variety": [{"symbol": corpus.name(s), "count": freq[s], "distinct_next": len(succ[s]),
                             "distinct_prev": len(prec[s])} for s, _ in freq.most_common(top)],
        "mean_relative_position": [{"symbol": corpus.name(s), "count": freq[s],
                                    "mean": round(sum(relpos[s]) / len(relpos[s]), 3)}
                                   for s, _ in freq.most_common(top)],
        "top_cooccurrences": [{"symbols": nm(k), "lines": v} for k, v in cooc.most_common(8) if v >= 2],
        "most_similar_lines": [{"lines": [i, j], "bigram_jaccard": round(v, 3)} for v, i, j in sims[:8]],
        "most_similar_documents": [{"documents": [a, b], "bigram_jaccard": round(v, 3)} for v, a, b in doc_sim[:5]],
    }


# ------------------------------------------------------------------------------------------------ 接辞

def _affix_stats(types, maxlen=2):
    tset = set(types)
    out = {}
    for side in ("suffix", "prefix"):
        for L in range(1, maxlen + 1):
            ends = defaultdict(set)          # 語幹 → 付く接辞
            for w in tset:
                if len(w) > L:
                    stem, aff = (w[:-L], w[-L:]) if side == "suffix" else (w[L:], w[:L])
                    ends[stem].add(aff)
            cnt = Counter()
            for stem, affs in ends.items():
                if len(affs) >= 2 or stem in tset:      # 語幹が別の接辞とも、単独でも現れる
                    for a in affs:
                        cnt[a] += 1
            for a, k in cnt.items():
                if k >= 2:
                    out[(side, a)] = k
    return out


def affix_test(words, B, seed=0):
    """分割候補の語の型（異なり語）で、接辞の候補ごとに「別の接辞とも現れる語幹」の数を数え、
    語の型ごとに記号を並べ替えた帰無モデル（語の長さと構成を保つ）と比べる（標準化 maxT）。"""
    types = sorted(set(words))
    obs = _affix_stats(types)
    rng = random.Random(seed)
    samples = []
    for _ in range(B):
        sh = []
        for w in types:
            x = list(w)
            rng.shuffle(x)
            sh.append(tuple(x))
        samples.append(_affix_stats(sh))
    keys = set(obs)
    for s in samples:
        keys.update(s)
    mu, sd = {}, {}
    for k in keys:
        vals = [s.get(k, 0) for s in samples] + [obs.get(k, 0)]
        m = sum(vals) / len(vals)
        mu[k], sd[k] = m, max(0.5, math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)))
    maxes = sorted(max(((v - mu[k]) / sd[k] for k, v in s.items()), default=0.0) for s in samples)
    res = {}
    for k, v in obs.items():
        z = (v - mu[k]) / sd[k]
        res[k] = {"stat": z, "value": v, "null_mean": mu[k],
                  "p_family": (1 + sum(1 for m in maxes if m >= z - 1e-12)) / (B + 1),
                  "p_raw": (1 + sum(1 for s in samples if s.get(k, 0) >= v)) / (B + 1)}
    return {"keys": res}


# ------------------------------------------------------------------------------------------------ 転写の攪乱

def perturb(corpus: Corpus, rate, rng):
    """誤分類（別の記号に置換）・分割（1記号→2記号）・結合（隣接2記号→1記号）を起こした転写を作る。
    画像由来で一致度の低い記号は、誤分類されやすいとして3倍の確率で選ぶ。"""
    symbols = list(corpus.symbols)
    ids = {s: i for i, s in enumerate(symbols)}
    freq = Counter(s for ln in corpus.lines for s in ln)
    pool = [s for s, k in freq.items() for _ in range(k)]
    low = {tuple(x) for x in corpus.source.get("low_confidence_positions") or []}

    def new_sym(name):
        if name not in ids:
            ids[name] = len(symbols)
            symbols.append(name)
        return ids[name]
    lines, origins = [], []
    for li, (ln, org) in enumerate(zip(corpus.lines, corpus.origins)):
        out, o2, i = [], [], 0
        while i < len(ln):
            r = rate * (3 if (li, org[i]) in low else 1)
            u = rng.random()
            if u < r:                                   # 誤分類
                out.append(rng.choice(pool))
                o2.append(org[i])
            elif u < r * 1.5:                           # 分割
                out += [new_sym(symbols[ln[i]] + "~a"), new_sym(symbols[ln[i]] + "~b")]
                o2 += [org[i], org[i]]
            elif u < r * 2 and i + 1 < len(ln):         # 結合
                out.append(new_sym(symbols[ln[i]] + "+" + symbols[ln[i + 1]]))
                o2.append(org[i])
                i += 1
            else:
                out.append(ln[i])
                o2.append(org[i])
            i += 1
        lines.append(tuple(out))
        origins.append(o2)
    return corpus.copy_with_lines(lines, origins, symbols)


# ------------------------------------------------------------------------------------------------ 本体

def _canon(fam, key, corpus):
    """部分標本・攪乱した転写の間で同じ仮説を照合するための名前（記号名・元の行番号）。"""
    if fam == "linepair":
        return (fam, tuple(sorted(corpus.line_info[i].get("_orig", i) for i in key)))
    if fam == "position":
        return (fam, key[0], tuple(corpus.names(key[1])))
    if fam == "gap":
        return (fam, corpus.name(key[0]), key[1])
    if fam == "ngram":
        return (fam, tuple(corpus.names(key)))
    if fam in ("cooccurrence", "substitution"):
        return (fam, tuple(sorted(corpus.names(key))))
    if fam == "dispersion":
        return (fam, key[0], corpus.name(key[1]))
    return (fam, key)


def key_json(fam, key, corpus, pairs=None):
    """仮説の対象を機械が読める形で（評価・照合用）。"""
    if fam == "linepair" and pairs is not None:
        orig = [corpus.line_info[i].get("_orig", i) for i in key]
        return {"family": fam, "lines": orig,
                "pairs": [[corpus.line_info[i].get("_orig", i), corpus.line_info[j].get("_orig", j)] for i, j in pairs]}
    if fam == "dispersion":
        return {"family": fam, "side": key[0], "symbols": [corpus.name(key[1])]}
    if fam == "ngram":
        return {"family": fam, "symbols": corpus.names(key)}
    if fam == "position":
        return {"family": fam, "side": key[0], "symbols": corpus.names(key[1])}
    if fam == "gap":
        return {"family": fam, "symbols": [corpus.name(key[0])], "distance": key[1]}
    if fam in ("cooccurrence", "substitution"):
        return {"family": fam, "symbols": corpus.names(key)}
    if fam == "linepair":
        return {"family": fam, "lines": [corpus.line_info[i].get("_orig", i) for i in key]}
    if fam == "affix":
        return {"family": fam, "side": key[0], "symbols": corpus.names(key[1])}
    return {"family": fam}


def _significant(res, corpus, thr, min_support):
    out = set()
    for fam, r in res.items():
        if fam.startswith("_"):
            continue
        for key, v in r["keys"].items():
            if v["p_family"] <= thr and _support_of(fam, key, v, corpus) >= min_support.get(fam, 3):
                out.add(_canon(fam, key, corpus))
    return out


def _support_of(fam, key, v, corpus):
    if fam == "linepair":
        return v.get("value", 0)
    if fam == "cooccurrence":
        return sum(1 for ln in corpus.lines if key[0] in ln and key[1] in ln)
    if fam == "substitution":
        return min(sum(ln.count(key[0]) for ln in corpus.lines), sum(ln.count(key[1]) for ln in corpus.lines))
    if fam == "dispersion":
        return sum(1 for ln in corpus.lines if key[1] in ln) if key[0] == "even" else sum(ln.count(key[1]) for ln in corpus.lines)
    if fam == "ngram":
        return sum(1 for ln in corpus.lines for g in ngrams(ln, len(key)) if g == key)
    if fam == "position":
        side, seq = key
        n = len(seq)
        return sum(1 for ln in corpus.lines if len(ln) >= n and (ln[:n] if side == "^" else ln[-n:]) == seq)
    if fam == "gap":
        s, d = key
        return sum(1 for ln in corpus.lines for i in range(len(ln) - d) if ln[i] == s and ln[i + d] == s)
    return 0


STAT_KIND = {"ngram": "-log10 Poisson 上側確率（行内シャッフルでの期待値）", "position": "-log10 Poisson 上側確率（行内シャッフルでの期待値）",
             "gap": "-log10 Poisson 上側確率（行内シャッフルでの期待値）", "cooccurrence": "同じ行に現れる回数を置換分布で標準化した z",
             "dispersion": "置換分布で標準化した z", "substitution": "文脈の残差コサインを置換分布で標準化した z",
             "linepair": "共有する2〜4記号列の数を置換分布で標準化した z（行の組の最良値）",
             "affix": "別の形とも現れる語幹の数を置換分布で標準化した z"}
NULL_TEXT = {"ngram": "行内シャッフル", "position": "行内シャッフル", "gap": "行内シャッフル", "substitution": "行内シャッフル",
             "cooccurrence": "行の長さの層ごとに、各記号を含む行の数を保った独立配置", "dispersion": "全体シャッフル", "linepair": "全体シャッフル",
             "affix": "語の型ごとの並べ替え"}


def _consolidate(sig_keys, corpus, cfg):
    """有意な鍵を、人が読む仮説の単位にまとめる。

    - n-gram: 長さが上限の列が1記号ずつずれて同じ回数で続くものは、1本の長い反復列につなぐ。
      長い列に含まれ、その回数の 80% 以上がその長い列の中での出現である短い列は、長い列の一部（contains）にする
    - 行頭・行末: 同じ回数でより長い列に含まれるものは長い列にまとめる
    - 行の組: 連結成分ごとに1つの仮説（似た行のまとまり）にする
    """
    items = []
    ng = sorted(sig_keys.get("ngram", []), key=lambda e: (-len(e[1]), -e[3]))
    cnt = {e[1]: e[3] for e in ng}
    ent = {e[1]: e for e in ng}
    mx = cfg["max_n"]
    longest = [k for k in cnt if len(k) == mx]
    nxt = {}
    for k in longest:
        for k2 in longest:
            if k2 != k and k[1:] == k2[:-1] and cnt[k2] == cnt[k]:
                nxt[k] = k2
    has_prev = set(nxt.values())
    chains = []
    for k in longest:
        if k in has_prev and any(nxt.get(p) == k for p in longest):
            continue
        chain, cur, seen = [k], k, {k}
        while cur in nxt and nxt[cur] not in seen:
            cur = nxt[cur]
            seen.add(cur)
            chain.append(cur)
        chains.append(chain)
    in_chain = {k for c in chains for k in c}
    kept = []          # (seq, members)
    def occ(seq):
        n = len(seq)
        return sum(1 for ln in corpus.lines for i in range(len(ln) - n + 1) if ln[i:i + n] == seq)
    for c in chains:
        seq = c[0] + tuple(x[-1] for x in c[1:])
        if len(c) > 1 and occ(seq) == cnt[c[0]]:
            kept.append((seq, c))
        else:                       # 連続して見えても、実際に1本の列として現れない場合はつながない
            in_chain.difference_update(c)
    for fam, key, v, supp in ng:
        if key in in_chain:
            continue
        host = None
        for seq, members in kept:
            if len(seq) > len(key) and any(seq[i:i + len(key)] == key for i in range(len(seq) - len(key) + 1)):
                if cnt[members[0]] >= 0.8 * supp:
                    host = (seq, members)
                    break
        if host:
            host[1].append(key)
        else:
            kept.append((key, [key]))
    for seq, members in kept:
        best = min((ent[m] for m in members if m in ent), key=lambda e: e[2]["p_family"])
        items.append({"fam": "ngram", "key": seq, "v": best[2], "canons": [_canon("ngram", best[1], corpus)],
                      "contains": [m for m in members if m != seq and len(m) < len(seq) and m not in in_chain]})
    pos = sig_keys.get("position", [])
    for fam, key, v, supp in pos:
        if any(k2[0] == key[0] and len(k2[1]) > len(key[1]) and s2 == supp and
               (k2[1][:len(key[1])] == key[1] if key[0] == "^" else k2[1][-len(key[1]):] == key[1])
               for _, k2, _, s2 in pos):
            continue
        items.append({"fam": "position", "key": key, "v": v, "canons": [_canon("position", key, corpus)]})
    pairs = sig_keys.get("linepair", [])
    if pairs:
        pv = {e[1]: e[2] for e in pairs}
        for comp in components([e[1] for e in pairs]):
            mem = set(comp)
            ps = [k for k in pv if k[0] in mem]
            best = min((pv[k] for k in ps), key=lambda v: v["p_family"])
            items.append({"fam": "linepair", "key": tuple(sorted(comp)), "v": best, "pairs": ps,
                          "canons": [_canon("linepair", k, corpus) for k in ps]})
    for fam in cfg["families"]:
        if fam in ("ngram", "position", "linepair"):
            continue
        for _, key, v, supp in sig_keys.get(fam, []):
            items.append({"fam": fam, "key": key, "v": v, "canons": [_canon(fam, key, corpus)] if fam != "affix" else []})
    return items


def _min_support():
    return {f.name: f.min_support for f in FAMILIES}


def _detectability(res, cfg, n_fam, corpus):
    """このデータ量で、2記号の連接が「偶然でない」と判定されるには何回の出現が必要かの目安。

    n-gram の族の閾値（置換分布の最大統計量の 1 - alpha/族数 分位）を、典型的な頻度の記号の組の
    期待値に当てはめて、必要な最小の出現回数を求める。
    """
    from .stats import Expectation
    thr = cfg["alpha"] / n_fam
    out = {"family_threshold_p": round(thr, 5), "min_attainable_p": round(1 / (cfg["permutations"] + 1), 5),
           "permutations_sufficient": 1 / (cfg["permutations"] + 1) <= thr}
    nm = res.get("ngram", {}).get("null_max")
    if not nm:
        return out
    T = nm[min(len(nm) - 1, int(math.ceil((1 - thr) * len(nm))) - 1)]
    exp = Expectation(corpus.lines)
    freq = Counter(s for ln in corpus.lines for s in ln)
    ranked = [s for s, _ in freq.most_common()]
    need = {}
    picks = {"two_most_frequent_symbols": ranked[:2], "two_median_frequency_symbols": ranked[len(ranked) // 2:len(ranked) // 2 + 2]}
    for label, pair in picks.items():
        if len(pair) < 2:
            continue
        e = exp.ngram(tuple(pair))
        k = 2
        while k < 500 and surprise(k, e) < T:
            k += 1
        need[label] = {"symbols": corpus.names(pair), "expected_count_under_null": round(e, 3), "occurrences_needed": k}
    out["ngram_family_threshold_statistic"] = round(T, 3)
    out["bigram_occurrences_needed"] = need
    return out


def analyze(corpus: Corpus, config: dict | None = None) -> dict:
    t0 = time.time()
    cfg = _auto(dict(DEFAULTS, **(config or {})), corpus)
    for i, info in enumerate(corpus.line_info):
        info.setdefault("_orig", i)
    lines = corpus.lines
    if not lines or corpus.n_symbols < 4:
        raise ValueError("解析には4記号以上の記号列が必要")
    short = len(lines) <= SHORT_LINES or corpus.n_symbols <= SHORT_SYMBOLS
    fams = [f for f in cfg["families"] if f != "affix"]
    n_fam = len(cfg["families"])
    thr = cfg["alpha"] / n_fam
    minsup = _min_support()
    minsup["affix"] = 3
    timing = {}

    t = time.time()
    desc = describe(corpus, cfg["max_n"])
    timing["describe"] = time.time() - t

    t = time.time()
    test_cfg = {"max_n": cfg["max_n"], "max_gap": cfg["max_gap"], "permutations": cfg["permutations"], "families": fams}
    res = run_tests(lines, test_cfg, rng_seed=cfg["seed"])
    timing["tests"] = time.time() - t

    t = time.time()
    seg = bd.candidates(lines, seed=cfg["seed"], n_subsamples=5 if len(lines) >= 4 else 0,
                        compression_reps=cfg["compression_reps"])
    best = {"method": "consensus"}       # 接辞の検定は方式の過半数が切る分割を使う（記述長最小は短いコーパスで「切らない」に偏る）
    words = [w for ln, cs in zip(lines, seg["segmentations"]["consensus"]) for w in bd.words_from_cuts(ln, cs)]
    if "affix" in cfg["families"]:
        res["affix"] = affix_test(words, cfg["affix_permutations"], cfg["seed"])
    timing["boundaries"] = time.time() - t

    # ------------------------------------------------------------ 安定性: 標本の揺れ
    t = time.time()
    sig_main = _significant({k: v for k, v in res.items() if k != "affix"}, corpus, thr, minsup)
    rng = random.Random(cfg["seed"] + 1)
    sub_hits, sub_trials = Counter(), Counter()
    n_sub = cfg["subsamples"] if len(lines) >= 5 else 0
    sub_cfg = dict(test_cfg, permutations=cfg["subsample_permutations"])
    for r in range(n_sub):
        m = max(3, int(round(len(lines) * cfg["subsample_fraction"])))
        idx = sorted(rng.sample(range(len(lines)), m))
        sub = corpus.subset(idx)
        sres = run_tests(sub.lines, sub_cfg, rng_seed=cfg["seed"] + 100 + r, with_global=False)
        # 部分標本は小さいので、家族単位の補正（maxT）のみで判定し、族の数での Bonferroni はかけない
        sig = _significant(sres, sub, cfg["alpha"], minsup)
        kept = {sub.line_info[i]["_orig"] for i in range(len(idx))}
        for h in sig_main:
            if h[0] == "linepair" and not set(h[1]) <= kept:
                continue
            sub_trials[h] += 1
            sub_hits[h] += h in sig
    timing["subsample_stability"] = time.time() - t

    # ------------------------------------------------------------ 安定性: 転写の誤差
    t = time.time()
    per_hits, per_trials = Counter(), 0
    per_cfg = dict(test_cfg, permutations=cfg["perturbation_permutations"])
    prng = random.Random(cfg["seed"] + 2)
    for r in range(cfg["perturbations"]):
        pc = perturb(corpus, cfg["perturbation_rate"], prng)
        pres = run_tests(pc.lines, per_cfg, rng_seed=cfg["seed"] + 200 + r, with_global=False)
        sig = _significant(pres, pc, thr, minsup)
        per_trials += 1
        for h in sig_main:
            per_hits[h] += h in sig
    timing["transcription_stability"] = time.time() - t

    # ------------------------------------------------------------ 仮説の記録
    b = Builder(corpus)
    hyps, descriptive, below_support = [], [], []
    sig_keys = defaultdict(list)
    for fam, r in res.items():
        if fam.startswith("_"):
            continue
        for key, v in r["keys"].items():
            supp = _support_of(fam, key, v, corpus) if fam != "affix" else v["value"]
            entry = (fam, key, v, supp)
            if v["p_family"] <= thr:
                if supp >= minsup.get(fam, 3):
                    sig_keys[fam].append(entry)
                else:
                    below_support.append(entry)
            elif v["p_raw"] <= cfg["alpha"]:
                descriptive.append(entry)
    items = _consolidate(sig_keys, corpus, cfg)
    pos = sig_keys.get("position", [])
    ng_keys = [it["key"] for it in items if it["fam"] == "ngram"]
    seg_units = {w for w in words}
    for it in items:
        fam, key, v = it["fam"], it["key"], it["v"]
        if fam == "affix":
            side, aff = key
            rec = {"observation": f"分割候補（{best['method']}）の語のうち、{len(aff)}記号の{'接尾部' if side == 'suffix' else '接頭部'} "
                                  f"{' '.join(corpus.names(aff))} が、別の形とも現れる語幹 {v['value']} 種に付く",
                   "evidence": [], "support": v["value"], "support_lines": None,
                   "counterexamples": {"description": "（この接辞の付かない語幹は数えていない）", "count": None},
                   "measures": {"segmentation": best["method"]}, "symbols": corpus.names(aff)}
            stab_s = stab_t = None
            n_trials = 0
        else:
            rec = b.linegroup(key, it["pairs"]) if fam == "linepair" else getattr(b, fam)(key)
            canons = it["canons"]
            tr = [sub_trials[c] for c in canons if sub_trials[c]]
            stab_s = round(sum(sub_hits[c] / sub_trials[c] for c in canons if sub_trials[c]) / len(tr), 2) if tr else None
            stab_t = round(sum(per_hits[c] for c in canons) / (per_trials * len(canons)), 2) if per_trials else None
            n_trials = max(tr) if tr else 0
        if it.get("contains"):
            rec["contains"] = [" ".join(corpus.names(g)) for g in it["contains"][:10]]
        corro = []
        if fam == "ngram":
            if key in seg_units or any(key == w[:len(key)] or key == w[-len(key):] for w in seg_units if len(w) > len(key)):
                corro.append("単語境界の合議（方式の過半数）が、この列を語またはその端として切り出す")
            if any((k2[0] == "^" and key[:len(k2[1])] == k2[1]) or (k2[0] == "$" and key[-len(k2[1]):] == k2[1])
                   for _, k2, _, _ in pos):
                corro.append("行頭・行末の仮説と重なる")
        if fam == "position":
            if any(k2[:1] == key[1][:1] or k2[-1:] == key[1][-1:] for k2 in ng_keys):
                corro.append("反復する記号列の仮説と重なる")
        if fam == "dispersion" and key[0] == "even":
            if any(k2[0] in ("^", "$") and key[1] in k2[1] for _, k2, _, _ in pos):
                corro.append("行頭・行末の仮説と重なる")
        stable = all(x is None or x >= cfg["stable_threshold"] for x in (stab_s, stab_t))
        unc = _has_uncertain(rec.get("evidence", []))
        limits = []
        if fam != "affix":
            if stab_s is None:
                limits.append("行が少なく、部分標本での再現性を測れない")
            elif stab_s < cfg["stable_threshold"]:
                limits.append(f"行の部分標本（{int(cfg['subsample_fraction'] * 100)}%）で再現したのは {stab_s:.0%} のみ: 一部の行に依存している")
            if stab_t is not None and stab_t < cfg["stable_threshold"]:
                limits.append(f"転写を {cfg['perturbation_rate']:.0%} 攪乱すると再現したのは {stab_t:.0%} のみ: 画像認識・転写の誤差に弱い")
        else:
            limits.append("単語分割の候補に依存する。分割が誤っていれば接辞も見かけのもの（安定性は未測定）")
        if unc:
            limits.append("根拠に、人が不確かと印を付けた記号を含む")
        if short:
            limits.append("短いコーパス: 規則の一般性（別の資料でも成り立つか）は検証できない")
        measured = stab_s is not None
        # 接辞は分割候補に依存し、開発シードでの確認でも誤りが多かったので、常に暫定（C）とする
        tier = "C" if fam == "affix" else "A" if stable and measured and corro else "B" if stable and measured else "C"
        ikey = fam + (key[0] if fam in ("position", "dispersion") else "")
        hyps.append({
            "id": "", "family": fam, "kind": KIND_TEXT[fam], "tier": tier, **rec,
            "key": key_json(fam, key, corpus, it.get("pairs")),
            "statistics": {"p_family_corrected": round(v["p_family"], 5), "p_raw": round(v["p_raw"], 5),
                           "threshold": round(thr, 5), "statistic": round(v["stat"], 3),
                           "statistic_kind": STAT_KIND[fam], "null_model": NULL_TEXT[fam],
                           "permutations": cfg["affix_permutations"] if fam == "affix" else cfg["permutations"]},
            "stability": {"sampling": stab_s, "sampling_trials": n_trials,
                          "transcription": stab_t, "transcription_trials": per_trials if fam != "affix" else 0},
            "corroborated_by": corro,
            "interpretation": {"status": "未検証（観測からは選べない候補の列挙）", "candidates": INTERPRETATIONS[ikey]},
            "limits": limits,
            "needed_data": NEEDED[fam],
        })
    # 接辞は評価（シード 1000〜1009、凍結版 8dd39ac）で偽の率 94% だったため、仮説から外して「実験的な観察」にする。
    # 多重検定の族の数（閾値）は評価時と同じにして、他の族の判定を変えない（docs/ISSUES.md #41）
    experimental = [h for h in hyps if h["family"] == "affix"]
    for h in experimental:
        h["tier"] = "実験的"
        h["limits"].insert(0, "接辞の検定は評価で偽の率が94%だった。規則として扱わない（参考表示）")
    hyps = [h for h in hyps if h["family"] != "affix"]
    tier_order = {"A": 0, "B": 1, "C": 2}
    hyps.sort(key=lambda h: (tier_order[h["tier"]], h["statistics"]["p_family_corrected"], -h["statistics"]["statistic"]))
    for i, h in enumerate(hyps):
        h["id"] = f"H{i + 1:02d}"

    # 記述的観察（多重検定を通らなかったもの）
    descriptive.sort(key=lambda e: e[2]["p_raw"])
    desc_out = []
    for fam, key, v, supp in (below_support + descriptive)[:cfg["max_descriptive"]]:
        if fam == "affix":
            obs = f"接辞の候補 {' '.join(corpus.names(key[1]))}（{key[0]}）"
        else:
            obs = getattr(b, fam)(key)["observation"]
        desc_out.append({"family": fam, "observation": obs, "key": key_json(fam, key, corpus), "p_raw": round(v["p_raw"], 4),
                         "p_family_corrected": round(v["p_family"], 4), "support": supp,
                         "status": "支持する観測が少なすぎる" if (fam, key, v, supp) in below_support else
                         "多重検定の補正後は偶然と区別できない"})

    # 構造のまとめ
    subst_pairs = [tuple(h["symbols"]) for h in hyps if h["family"] == "substitution"]
    tmpl = templates(b, [tuple(p) for h in hyps if h["family"] == "linepair" for p in _local_pairs(h, corpus)])
    glob = res["_global"]
    any_global = any(g["p_holm"] <= cfg["alpha"] for g in glob.values())
    verdict = _verdict(hyps, any_global, short, desc)

    review = _review_points(hyps, seg, corpus, cfg)
    timing["total"] = time.time() - t0
    limits = _data_limits(desc, short, cfg, len(lines))
    out = {
        "schema": "babel-structure-analysis-v1", "version": VERSION,
        "input": {**{k: v for k, v in corpus.source.items() if k != "low_confidence_positions"},
                  "corrections_applied": corpus.corrections_applied,
                  "low_confidence_symbols": len(corpus.source.get("low_confidence_positions") or []),
                  "uncertain_marked": len(corpus.uncertain)},
        "mode": "short-corpus" if short else "standard",
        "config": {k: cfg[k] for k in ("alpha", "permutations", "max_n", "max_gap", "subsamples", "subsample_fraction",
                                        "perturbations", "perturbation_rate", "compression_reps", "seed")},
        "verdict": verdict,
        "statistics": desc,
        "global_tests": glob,
        "hypotheses": hyps,
        "descriptive_observations": desc_out,
        "experimental_observations": experimental,
        "structures": {
            "substitution_sets": [sorted(c) for c in components(subst_pairs)],
            "line_templates": tmpl,
        },
        "boundaries": {k: v for k, v in seg.items() if k != "segmentations"},
        "given_boundaries_agreement": _given_agreement(corpus, seg),
        "detectability": _detectability(res, cfg, n_fam, corpus),
        "data_limits": limits,
        "review_points": review,
        "multiple_testing": {"families": cfg["families"], "per_family_threshold": round(thr, 5),
                             "method": "族ごとに Westfall–Young single-step maxT（置換分布）、族の数で Bonferroni。大域検定は Holm"},
        "seconds": {k: round(v, 2) for k, v in timing.items()},
        "notes": ["観測された規則と、そこから考えられる機能（interpretation）は別物。機能は未検証。",
                  "規則が見つからないことは『無意味な記号列』の証明ではない。検出力（detectability）を併せて読むこと。",
                  "標本の揺れ（stability.sampling）と転写の誤差（stability.transcription）は別に測っている。"],
    }
    return out


def _has_uncertain(x):
    """根拠の入れ子の中まで見て、人が「不確か」と印を付けた記号を含むか（行の組の根拠は line_a / line_b の中に持つ）。"""
    if isinstance(x, dict):
        return bool(x.get("contains_uncertain_symbol")) or any(_has_uncertain(v) for v in x.values())
    if isinstance(x, list):
        return any(_has_uncertain(v) for v in x)
    return False


def _local_pairs(h, corpus):
    back = {info.get("_orig", i): i for i, info in enumerate(corpus.line_info)}
    return [(back[a], back[b]) for a, b in h["key"]["pairs"]]


def _verdict(hyps, any_global, short, desc):
    strong = [h for h in hyps if h["tier"] in ("A", "B")]
    if strong:
        text = f"偶然では説明しにくい規則性を {len(strong)} 件観測した（安定性の低いもの {len(hyps) - len(strong)} 件は別記）。意味・文法機能は未検証"
        code = "structure_observed"
    elif hyps:
        text = f"多重検定を通った規則性が {len(hyps)} 件あるが、いずれも部分標本または転写の攪乱で再現しにくい。暫定的な観測として扱う"
        code = "structure_unstable"
    elif any_global:
        text = "コーパス全体では無作為な並びより規則的だが、個々の規則として特定できるものはない"
        code = "global_only"
    else:
        text = ("このデータ量で検出できる規則性は見つからなかった。これは規則が存在しないことの証明ではない"
                + ("（短いコーパスのため検出力が低い）" if short else ""))
        code = "no_structure_detected"
    return {"code": code, "text": text, "hypotheses": len(hyps), "stable_hypotheses": len(strong),
            "global_structure": any_global}


def _data_limits(desc, short, cfg, n_lines):
    out = []
    if short:
        out.append("短いコーパスモード: 行数 ≤ 30 または記号数 ≤ 600。高次の統計（条件付きエントロピー、語順、語の分布）は記述のみ")
    if desc["type_token_ratio"] and desc["type_token_ratio"] > 0.4:
        out.append(f"記号の異なり数が多い（異なり/延べ = {desc['type_token_ratio']}）。1記号あたり平均 {desc['mean_count_per_type']} 回しか"
                   "現れず、多くの記号について位置・文脈の規則は検定できない")
    if desc["hapax_types"] and desc["symbol_types"] and desc["hapax_types"] / desc["symbol_types"] > 0.5:
        out.append(f"1回しか現れない記号が {desc['hapax_types']} 種（全 {desc['symbol_types']} 種）。"
                   "視覚クラスの分けすぎ（同じ文字が別クラス）の可能性も確認が必要")
    if n_lines < 5:
        out.append("行が5行未満のため、部分標本による安定性は測っていない")
    out.append("文の境界は入力の行（または文）をそのまま使う。改行が文の区切りとは限らない")
    out.append("単語・形態素レベルの置き換え可能性（語の範疇）は、単語境界が不確かなため記号レベルでのみ検定している")
    return out


def _review_points(hyps, seg, corpus, cfg):
    out = []
    for h in hyps:
        if h["tier"] == "C":
            out.append({"what": f"{h['id']}（{h['kind']}）: 安定性が低い", "why": "; ".join(h["limits"][:2]),
                        "where": h["evidence"][:3]})
        if h["family"] == "substitution":
            out.append({"what": f"{h['id']}: 記号 {' と '.join(h['symbols'])} を画像で比較する",
                        "why": "同じ文脈を共有する。別の文字か、同じ文字が2つの視覚クラスに分かれたかを人が判断できる",
                        "where": []})
        if h["family"] == "ngram" and len(h["symbols"]) == 2 and h["measures"]["p_last_given_prefix"] > 0.9 \
                and h["measures"]["p_first_given_suffix"] > 0.9:
            out.append({"what": f"{h['id']}: {' '.join(h['symbols'])} が常に並ぶ",
                        "why": "1つの文字が2つに分割されて認識された可能性がある（画像で確認）", "where": h["evidence"][:3]})
    unsure = []
    for li, shares in enumerate(seg["boundary_vote_share"]):
        for b, v in shares.items():
            if 0.4 <= v <= 0.6:
                unsure.append((li, int(b)))
    if unsure:
        out.append({"what": f"方式によって判断が分かれる単語境界 {len(unsure)} 箇所",
                    "why": "半数前後の方式だけが切る位置", "where": [{"line": li, "cut_before_position": b} for li, b in unsure[:10]]})
    low = corpus.source.get("low_confidence_positions") or []
    if low:
        out.append({"what": f"視覚クラスへの一致度が低い記号 {len(low)} 個", "why": "誤分類の可能性が高い（画像で確認、訂正ファイルで直せる）",
                    "where": [{"line": a, "position": p} for a, p in low[:10]]})
    comp = seg.get("compression_test")
    if comp and comp["p"] > cfg["alpha"]:
        out.append({"what": "単語境界の候補はすべて暫定", "why": f"MDL 分割の圧縮が行内シャッフルと区別できない（p={comp['p']}）", "where": []})
    return out


def _given_agreement(corpus, seg):
    if not corpus.given_boundaries:
        return None
    return {c["method"]: round(bd.boundary_f1(seg["segmentations"][c["method"]], corpus.given_boundaries), 3)
            for c in seg["candidates"]}
