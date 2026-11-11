"""パラダイムのクラス分け・品詞ラベル・語順（意味情報を使わない構造推定）。"""
from __future__ import annotations

from collections import Counter, defaultdict


def paradigm_groups(infos: dict):
    """infos: 正規化語形 -> (接頭辞, 語幹, 接尾辞)。接辞を「同じ基体に付くか」でグループ化する。"""
    bases = defaultdict(set)
    for pre, stem, suf in infos.values():
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
            if shared >= max(3, 0.1 * min(len(bases[a]), len(bases[b]))):
                parent[find(a)] = find(b)
    group_of = {t: find(t) for t in tags}
    votes = defaultdict(Counter)
    for pre, stem, suf in infos.values():
        for t in ["P:" + a for a in pre] + ["S:" + a for a in suf]:
            votes[stem][group_of[t]] += 1
    stem_group = {s: sorted(v.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for s, v in votes.items()}
    return group_of, stem_group


def label_classes(sentences, infos, stem_group, numeral_types=()):
    """sentences: 正規化語形の列の列。語幹→品詞ラベル(N/V/X) と根拠を返す。"""
    stems_in = defaultdict(set)
    for s, g in stem_group.items():
        stems_in[g].add(s)
    stats = {}
    n_sent = max(1, len(sentences))
    for g in sorted(stems_in):
        covered = exactly_one = 0
        for sent in sentences:
            k = sum(1 for w in sent if w in infos and stem_group.get(infos[w][1]) == g)
            covered += k > 0
            exactly_one += k == 1
        stats[g] = {"stems": len(stems_in[g]), "coverage": covered / n_sent,
                    "one_per_sentence": exactly_one / max(1, covered)}
    cands = [g for g in stats if stats[g]["stems"] >= 4 and stats[g]["coverage"] >= 0.3]
    verb = None
    if cands:
        best = max(cands, key=lambda g: (stats[g]["one_per_sentence"], g))
        others = [g for g in cands if g != best]
        if stats[best]["one_per_sentence"] >= 0.7 and others:
            verb = best
    label = {}
    for g in stats:
        label[g] = "V" if g == verb else ("N" if stats[g]["stems"] >= 8 and verb is not None or
                                          (verb is None and stats[g]["stems"] >= 8) else "X")
    stem_pos = {s: label[g] for s, g in stem_group.items()}
    conf = 0.0
    if verb is not None:
        second = max((stats[g]["one_per_sentence"] for g in cands if g != verb), default=0.0)
        conf = max(0.0, min(0.99, 0.5 + (stats[verb]["one_per_sentence"] - second)))
    return stem_pos, {"groups": stats, "verb_group": verb, "verb_confidence": round(conf, 3), "labels": label}


def function_words(sentences, infos, stem_pos, numeral_types):
    """屈折せず高頻度な語を機能語候補とする。"""
    cnt = Counter(w for s in sentences for w in s)
    total = max(1, sum(cnt.values()))
    out = set()
    for w, c in cnt.items():
        if w in infos and w not in numeral_types:
            pre, stem, suf = infos[w]
            if not pre and not suf and stem_pos.get(stem, "X") == "X" and c >= 8 and c / total >= 0.003:
                out.add(w)
    return out


def verb_position(sentences, infos, stem_pos):
    counts = Counter()
    for sent in sentences:
        idx = [i for i, w in enumerate(sent) if w in infos and stem_pos.get(infos[w][1]) == "V"]
        if len(idx) == 1 and len(sent) >= 3:
            i = idx[0]
            counts["initial" if i == 0 else ("final" if i == len(sent) - 1 else "medial")] += 1
    n = sum(counts.values())
    if n < 10:
        return {"value": None, "confidence": 0.0, "alternatives": {}, "n": n}
    alts = {k: round(counts[k] / n, 3) for k in ("initial", "medial", "final")}
    best = max(alts, key=lambda k: alts[k])
    return {"value": best, "confidence": round(alts[best] * min(1.0, n / 40), 3), "alternatives": alts, "n": n}
