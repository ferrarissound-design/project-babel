"""単語境界の推定（複数の方式を比べ、上位の候補を不確実性つきで出す）。

方式
- transition: 文字遷移確率が低い所で切る（既存 BABEL の babel.decipher.segment.TransitionModel をそのまま使う）
- unigram:    既存 BABEL の単語ユニグラム分割（babel.decipher.segment.unigram_segment をそのまま使う）
- branching:  分岐エントロピー（直前の k 記号の後に来る記号の多様さ）が増える所で切る（Tanaka-Ishii 型）
- mdl:        記述長の利得（Kit & Wilks の DLG）で、隣り合う単位の結合を貪欲に繰り返す。結合の終わった単位の端が境界
- repeats:    2回以上現れる極大反復部分列の両端に票を入れ、票の集まる所で切る
- consensus:  上の方式の過半数が切る所で切る

どの分割も正解として固定しない。各候補に、共通の記述長（2部符号）・他方式との一致度・行の部分標本での再現率を付ける。
さらに「この分割による圧縮が、行内シャッフルした記号列でも同じように得られるか」を置換検定し、
得られるなら、境界は記号の頻度の偏りだけで説明できる（＝境界の統計的根拠がない）と報告する。
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

from babel.decipher import segment as legacy

from .stats import repeated_ngrams, shuffle_within


def _as_str(lines):
    return ["".join(chr(0xF0000 + s) for s in ln) for ln in lines]


def _cuts_from_words(words):
    cuts, k = set(), 0
    for w in words[:-1]:
        k += len(w)
        cuts.add(k)
    return cuts


def words_from_cuts(ln, cuts):
    out, prev = [], 0
    for c in sorted(cuts) + [len(ln)]:
        if c > prev:
            out.append(tuple(ln[prev:c]))
            prev = c
    return out


# --------------------------------------------------------------------------------------------- 各方式

def seg_transition(lines):
    return [_cuts_from_words(ws) for ws in legacy.transition_segment(_as_str(lines))]


def seg_unigram(lines):
    return [_cuts_from_words(ws) for ws in legacy.unigram_segment(_as_str(lines))]


def seg_branching(lines, kmax=3):
    right, left = defaultdict(Counter), defaultdict(Counter)
    for ln in lines:
        for k in range(1, kmax + 1):
            for i in range(len(ln) - k + 1):
                g = ln[i:i + k]
                right[g][ln[i + k] if i + k < len(ln) else None] += 1
                left[g][ln[i - 1] if i > 0 else None] += 1

    def H(c):
        n = sum(c.values())
        return -sum(v / n * math.log2(v / n) for v in c.values()) if n else 0.0

    hr = {g: H(c) for g, c in right.items()}
    hl = {g: H(c) for g, c in left.items()}
    out = []
    for ln in lines:
        cuts = set()
        for b in range(1, len(ln)):
            votes = total = 0
            for k in range(1, kmax + 1):
                if b - k >= 0 and b - k - 1 >= 0:        # 右分岐: k 記号の文脈が1つ進んで多様さが増えたか
                    total += 1
                    votes += hr.get(ln[b - k:b], 0) > hr.get(ln[b - k - 1:b - 1], 0) + 1e-9
                if b + k <= len(ln) and b + k + 1 <= len(ln):   # 左分岐（逆向き）
                    total += 1
                    votes += hl.get(ln[b:b + k], 0) > hl.get(ln[b + 1:b + k + 1], 0) + 1e-9
            if total and votes * 2 > total:
                cuts.add(b)
        out.append(cuts)
    return out


def _corpus_cost(counts, N):
    return N * math.log2(N) - sum(c * math.log2(c) for c in counts.values() if c) if N else 0.0


def seg_mdl(lines, max_iter=None):
    """隣り合う単位の組を、記述長が減る限り結合する。戻り値: (境界, 結合の記録)。"""
    toks = [[(s,) for s in ln] for ln in lines]
    counts = Counter(t for ln in toks for t in ln)
    N = sum(counts.values())
    max_iter = max_iter or max(50, N)
    merges = []
    for _ in range(max_iter):
        pairs = Counter()
        for ln in toks:
            prev = None
            for a, b in zip(ln, ln[1:]):
                if (a, b) == prev:          # "aaa" で重なった組を二重に数えない
                    prev = None
                    continue
                pairs[(a, b)] += 1
                prev = (a, b)
        base = _corpus_cost(counts, N)
        lex_bits = 2 * math.log2(len(counts) + 1)
        best, best_gain = None, 1e-9
        for (a, b), c in pairs.items():
            if c < 2:
                continue
            na, nb = counts[a] - c, counts[b] - c
            if a == b:
                na = counts[a] - 2 * c
                if na < 0:
                    continue
            changed = {a: na, b: nb} if a != b else {a: na}
            N2 = N - c
            old = sum(counts[t] * math.log2(counts[t]) for t in changed if counts[t])
            new = sum(v * math.log2(v) for v in changed.values() if v > 0) + c * math.log2(c)
            cost2 = N2 * math.log2(N2) - (N * math.log2(N) - base - old + new)
            gain = base - cost2 - lex_bits
            if gain > best_gain:
                best, best_gain = (a, b), gain
        if best is None:
            break
        a, b = best
        z = a + b
        merges.append({"unit": z, "gain_bits": round(best_gain, 2)})
        new_toks = []
        for ln in toks:
            out, i = [], 0
            while i < len(ln):
                if i + 1 < len(ln) and ln[i] == a and ln[i + 1] == b:
                    out.append(z)
                    i += 2
                else:
                    out.append(ln[i])
                    i += 1
            new_toks.append(out)
        toks = new_toks
        counts = Counter(t for ln in toks for t in ln)
        N = sum(counts.values())
    return [_cuts_from_words(ln) for ln in toks], merges


def seg_repeats(lines, min_votes=2):
    rep = repeated_ngrams(lines, 2, 64)
    # 極大反復: 1記号伸ばした列が同じ回数で繰り返すものは除く
    longer = defaultdict(int)
    for g, k in rep.items():
        longer[g[1:]] = max(longer[g[1:]], k)
        longer[g[:-1]] = max(longer[g[:-1]], k)
    maximal = {g for g, k in rep.items() if longer.get(g, 0) < k}
    out = []
    for ln in lines:
        votes = Counter()
        for n in range(2, len(ln) + 1):
            for i in range(len(ln) - n + 1):
                if ln[i:i + n] in maximal:
                    votes[i] += 1
                    votes[i + n] += 1
        out.append({b for b, v in votes.items() if 0 < b < len(ln) and v >= min_votes})
    return out


METHODS = {"transition": seg_transition, "unigram": seg_unigram, "branching": seg_branching,
           "mdl": lambda lines: seg_mdl(lines)[0], "repeats": seg_repeats}


# --------------------------------------------------------------------------------------------- 評価量

def description_length(lines, cuts, n_symbols):
    """共通の2部符号の記述長（ビット）: 語彙（各語を記号と終端で書く）＋ 語の列（ユニグラム）。"""
    words = Counter(w for ln, cs in zip(lines, cuts) for w in words_from_cuts(ln, cs))
    N = sum(words.values())
    lex = sum((len(w) + 1) * math.log2(n_symbols + 1) for w in words)
    return lex + _corpus_cost(words, N), len(words), N


def boundary_f1(pred, gold):
    tp = sum(len(p & g) for p, g in zip(pred, gold))
    np_, ng = sum(len(p) for p in pred), sum(len(g) for g in gold)
    if np_ == 0 and ng == 0:
        return 1.0
    prec = tp / np_ if np_ else 0.0
    rec = tp / ng if ng else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def mdl_ratio(lines):
    V = len({s for ln in lines for s in ln})
    cuts, _ = seg_mdl(lines)
    dl, _, _ = description_length(lines, cuts, V)
    base, _, _ = description_length(lines, [set(range(1, len(ln))) for ln in lines], V)
    return dl / base if base else 1.0


def compression_test(lines, reps=19, seed=0):
    """MDL 分割による圧縮率（分割後の記述長 / 1記号＝1語の記述長）を、行内シャッフルと比べる。"""
    rng = random.Random(seed)
    obs = mdl_ratio(lines)
    null = [mdl_ratio(shuffle_within(lines, rng)) for _ in range(reps)]
    p = (1 + sum(1 for v in null if v <= obs + 1e-12)) / (reps + 1)
    return {"observed_ratio": round(obs, 4), "null_mean_ratio": round(sum(null) / len(null), 4) if null else None,
            "null_min_ratio": round(min(null), 4) if null else None, "p": round(p, 4), "reps": reps,
            "description": "MDL 分割で記述長がどれだけ縮むか（小さいほど再利用される単位が多い）。行内シャッフルと比較"}


# --------------------------------------------------------------------------------------------- 候補の生成

def candidates(lines, seed=0, n_subsamples=5, compression_reps=19, top_k=3):
    V = len({s for ln in lines for s in ln})
    segs = {name: fn(lines) for name, fn in METHODS.items()}
    votes = [Counter() for _ in lines]
    for cs in segs.values():
        for i, c in enumerate(cs):
            votes[i].update(c)
    k = len(segs)
    segs["consensus"] = [{b for b, v in vc.items() if v * 2 > k} for vc in votes]
    # 行の部分標本での再現率（MDL・合議）
    rng = random.Random(seed)
    stab = {name: [] for name in ("mdl", "consensus", "transition", "branching", "repeats", "unigram")}
    for _ in range(n_subsamples if len(lines) >= 4 else 0):
        idx = sorted(rng.sample(range(len(lines)), max(2, int(len(lines) * 0.7))))
        sub = [lines[i] for i in idx]
        sub_segs = {name: fn(sub) for name, fn in METHODS.items()}
        sv = [Counter() for _ in sub]
        for cs in sub_segs.values():
            for i, c in enumerate(cs):
                sv[i].update(c)
        sub_segs["consensus"] = [{b for b, v in vc.items() if v * 2 > k} for vc in sv]
        for name in stab:
            stab[name].append(boundary_f1(sub_segs[name], [segs[name][i] for i in idx]))
    names = list(segs)
    agree = {a: {b: round(boundary_f1(segs[a], segs[b]), 3) for b in names if b != a} for a in names}
    out = []
    for name in names:
        dl, n_types, n_tokens = description_length(lines, segs[name], V)
        st = stab.get(name) or []
        out.append({"method": name, "description_length_bits": round(dl, 1), "word_types": n_types,
                    "word_tokens": n_tokens, "mean_agreement_with_others": round(sum(agree[name].values()) / len(agree[name]), 3),
                    "subsample_reproducibility": round(sum(st) / len(st), 3) if st else None,
                    "cuts": [sorted(c) for c in segs[name]]})
    out.sort(key=lambda x: x["description_length_bits"])
    for r, c in enumerate(out):
        c["rank_by_description_length"] = r + 1
    base_none = description_length(lines, [set() for _ in lines], V)[0]
    base_all = description_length(lines, [set(range(1, len(ln))) for ln in lines], V)[0]
    # 位置ごとの境界の確からしさ（方式の票の割合）
    pos = []
    for i, ln in enumerate(lines):
        pos.append({str(b): round(votes[i][b] / k, 2) for b in range(1, len(ln)) if votes[i][b]})
    comp = compression_test(lines, reps=compression_reps, seed=seed) if compression_reps else None
    return {"candidates": out, "top": [c["method"] for c in out[:top_k]], "agreement": agree,
            "baselines_bits": {"no_cuts": round(base_none, 1), "every_symbol": round(base_all, 1)},
            "boundary_vote_share": pos, "compression_test": comp, "segmentations": segs}
