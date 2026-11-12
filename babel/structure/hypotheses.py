"""検定の結果から、人が読める仮説の記録を作る。

各仮説は「観測された規則（observation）」と「そこから考えられる機能の候補（interpretation, 未検証）」を分けて持つ。
解釈の候補は、証拠からは選べない可能性の一覧であり、どれかを主張するものではない。
"""
from __future__ import annotations

import difflib
from collections import Counter, defaultdict

from .stats import ngrams

KIND_TEXT = {
    "ngram": "記号列の反復・連接",
    "position": "行頭・行末への集中",
    "gap": "一定間隔での反復",
    "cooccurrence": "同じ行での共起",
    "dispersion": "行への散らばり方（均等・集中）",
    "substitution": "置き換え可能に見える記号",
    "linepair": "似た記号列を共有する行",
    "affix": "共通の接頭部・接尾部",
}

INTERPRETATIONS = {
    "ngram": ["語・語幹・定型句の一部", "接辞", "（2記号の場合）画像認識で1文字が2つに分割された可能性"],
    "position^": ["文頭・行頭の標識", "定型の書き出し", "行番号・見出し", "改行位置の規則（文の始まりとは限らない）"],
    "position$": ["文末・行末の標識", "終止の接辞・定型の結び", "改行位置の規則（文の終わりとは限らない）"],
    "gap": ["定型の枠（X _ _ X）", "反復句・韻律", "表・列の構造"],
    "cooccurrence": ["同じ話題・文型で使われる語の一部", "一致（agreement）", "同じ写字生・同じ文書の癖"],
    "dispersioneven": ["各文に1回ずつ現れる必須の要素（標識・接辞・機能語の一部）", "行・文の区切りに関わる記号", "罫線・装飾の一部を記号と誤認した可能性"],
    "dispersionclump": ["特定の文書・話題に固有の語の一部", "筆記者・時期による字体の違い（転写で別クラスになった）", "表・一覧の一部"],
    "substitution": ["同じ文法的範疇（同じ枠に入る別の記号）", "異体字", "画像認識で同じ文字が2つの視覚クラスに分かれた可能性"],
    "linepair": ["同じ文型・定型文", "同じ句の再録", "（転写の場合）同じ行の重複転写"],
    "affix": ["屈折・派生の接辞", "単語分割の誤りによる見かけの接辞"],
}

NEEDED = {
    "ngram": "同じ記号列が別の文書・別の筆記者でも現れるか。前後の文脈が異なる出現例",
    "position": "改行が文の区切りかどうかの外部情報（句読点・余白・行の長さの分布）、別資料での行頭・行末の記号",
    "gap": "同じ間隔の繰り返しが別の行・別の資料でも起きるか",
    "cooccurrence": "共起する行の数を増やす（現在の行数では偶然との区別が弱い）",
    "dispersion": "行が文の単位かどうか（改行の位置）の確認と、別資料の行での出現回数",
    "substitution": "2つの記号が同じ枠に入る例と、同じ行に並んで現れる例。画像で2つの字形を人が比較すること",
    "linepair": "同じ枠を持つ行がさらにあるか。行の対応部分の画像を人が照合すること",
    "affix": "分割の別候補でも同じ接辞が出るか。接辞の付く語幹が他の接辞とも現れる例",
}


def _fmt(names):
    return " ".join(names)


class Builder:
    def __init__(self, corpus, max_examples=8):
        self.c = corpus
        self.lines = corpus.lines
        self.max_examples = max_examples

    # ------------------------------------------------------------------------------- 位置
    def loc(self, line, start, length=1):
        info = self.c.line_info[line]
        origin = self.c.origins[line][start:start + length]
        d = {"line": line, "start": start, "length": length, "doc": info.get("doc"), "sent": info.get("sent"),
             "symbols": self.c.names(self.lines[line][start:start + length])}
        if self.c.boxes:
            bxs = [self.c.boxes.get((line, o)) for o in origin]
            bxs = [b for b in bxs if b]
            if bxs:
                d["image_box"] = [min(b[0] for b in bxs), min(b[1] for b in bxs), max(b[2] for b in bxs), max(b[3] for b in bxs)]
        if any((line, o) in self.c.uncertain for o in origin):
            d["contains_uncertain_symbol"] = True
        return d

    def occurrences(self, seq):
        n = len(seq)
        out = []
        for li, ln in enumerate(self.lines):
            for i in range(len(ln) - n + 1):
                if ln[i:i + n] == seq:
                    out.append((li, i))
        return out

    # ------------------------------------------------------------------------------- 族ごと
    def ngram(self, g):
        occ = self.occurrences(g)
        pre, suf = g[:-1], g[1:]
        pre_occ = [o for o in self.occurrences(pre) if not (o[1] + len(g) <= len(self.lines[o[0]])
                                                              and self.lines[o[0]][o[1]:o[1] + len(g)] == g)]
        suf_occ = [o for o in self.occurrences(suf) if not (o[1] >= 1 and self.lines[o[0]][o[1] - 1:o[1] - 1 + len(g)] == g)]
        names = self.c.names(g)
        k = len(occ)
        obs = (f"記号列 {_fmt(names)} が {k} 回現れる" if len(g) > 2 else
               f"記号 {names[0]} の直後に {names[1]} が {k} 回現れる")
        lines_with = len({li for li, _ in occ})
        return {
            "observation": obs,
            "evidence": [self.loc(li, i, len(g)) for li, i in occ[:self.max_examples]],
            "support": k,
            "support_lines": lines_with,
            "counterexamples": {
                "description": f"{_fmt(self.c.names(pre))} の後に {names[-1]} が続かない出現、"
                               f"{_fmt(self.c.names(suf))} の前に {names[0]} がない出現",
                "count": len(pre_occ) + len(suf_occ),
                "examples": [self.loc(li, i, len(pre)) for li, i in pre_occ[:4]] +
                            [self.loc(li, i, len(suf)) for li, i in suf_occ[:4]]},
            "measures": {"p_last_given_prefix": round(k / max(1, k + len(pre_occ)), 3),
                         "p_first_given_suffix": round(k / max(1, k + len(suf_occ)), 3), "length": len(g)},
            "symbols": names,
        }

    def position(self, key):
        side, seq = key
        n = len(seq)
        at = [li for li, ln in enumerate(self.lines) if len(ln) >= n and (ln[:n] if side == "^" else ln[-n:]) == seq]
        occ = self.occurrences(seq)
        elsewhere = [(li, i) for li, i in occ if not ((side == "^" and i == 0) or (side == "$" and i + n == len(self.lines[li])))]
        names = self.c.names(seq)
        where = "行頭" if side == "^" else "行末"
        return {
            "observation": f"{_fmt(names)} が {len(self.lines)} 行中 {len(at)} 行の{where}にある",
            "evidence": [self.loc(li, 0 if side == "^" else len(self.lines[li]) - n, n) for li in at[:self.max_examples]],
            "support": len(at),
            "support_lines": len(at),
            "counterexamples": {"description": f"{where}以外での出現", "count": len(elsewhere),
                                "examples": [self.loc(li, i, n) for li, i in elsewhere[:6]],
                                "lines_without": len(self.lines) - len(at)},
            "measures": {"share_of_occurrences_at_edge": round(len(at) / max(1, len(occ)), 3),
                         "share_of_lines": round(len(at) / max(1, len(self.lines)), 3)},
            "symbols": names, "side": side,
        }

    def gap(self, key):
        s, d = key
        pairs, other = [], Counter()
        for li, ln in enumerate(self.lines):
            ps = [i for i, x in enumerate(ln) if x == s]
            for a in range(len(ps)):
                for b in range(a + 1, len(ps)):
                    dd = ps[b] - ps[a]
                    if dd == d:
                        pairs.append((li, ps[a]))
                    elif 2 <= dd <= 8:
                        other[dd] += 1
        name = self.c.name(s)
        return {
            "observation": f"記号 {name} が {d} 記号の間隔で {len(pairs)} 回繰り返される",
            "evidence": [self.loc(li, i, d + 1) for li, i in pairs[:self.max_examples]],
            "support": len(pairs), "support_lines": len({li for li, _ in pairs}),
            "counterexamples": {"description": "同じ記号の他の間隔（2〜8）での組", "count": sum(other.values()),
                                "by_distance": dict(sorted(other.items()))},
            "measures": {"distance": d}, "symbols": [name],
        }

    def cooccurrence(self, key):
        a, b = key
        both = [li for li, ln in enumerate(self.lines) if a in ln and b in ln]
        only_a = sum(1 for ln in self.lines if a in ln and b not in ln)
        only_b = sum(1 for ln in self.lines if b in ln and a not in ln)
        na, nb = self.c.name(a), self.c.name(b)
        ev = []
        for li in both[:self.max_examples]:
            ln = self.lines[li]
            ev.append({"line": li, "doc": self.c.line_info[li].get("doc"), "positions": {na: ln.index(a), nb: ln.index(b)}})
        return {
            "observation": f"記号 {na} と {nb} が同じ行に {len(both)} 回現れる",
            "evidence": ev, "support": len(both), "support_lines": len(both),
            "counterexamples": {"description": "片方だけが現れる行", "count": only_a + only_b,
                                f"only_{na}": only_a, f"only_{nb}": only_b},
            "measures": {"jaccard_lines": round(len(both) / max(1, len(both) + only_a + only_b), 3)},
            "symbols": [na, nb],
        }

    def dispersion(self, key):
        side, x = key
        per_line = [ln.count(x) for ln in self.lines]
        has = sum(1 for k in per_line if k)
        name = self.c.name(x)
        dist = Counter(per_line)
        if side == "even":
            obs = f"記号 {name} が {len(self.lines)} 行中 {has} 行に現れ、偶然より均等に散らばる（1行あたりの回数の分布 {dict(sorted(dist.items()))}）"
            counter = {"description": "記号を含まない行、2回以上含む行", "count": dist.get(0, 0) + sum(v for k, v in dist.items() if k >= 2)}
        else:
            obs = f"記号 {name}（{sum(per_line)} 回）が {has} 行だけに集中する"
            counter = {"description": "（集中の反例は、記号を含む行の外での出現がないこと自体）", "count": None}
        ev = [{"line": li, "doc": self.c.line_info[li].get("doc"), "count": k} for li, k in enumerate(per_line) if k][:self.max_examples]
        return {"observation": obs, "evidence": ev, "support": has, "support_lines": has, "counterexamples": counter,
                "measures": {"lines_with_symbol": has, "occurrences": sum(per_line)}, "symbols": [name], "side": side}

    def linegroup(self, members, pairs):
        """似た行のまとまり（連結成分）。根拠は代表の組の比較。"""
        best = max(pairs, key=lambda p: len(set(ngrams(self.lines[p[0]], 3)) & set(ngrams(self.lines[p[1]], 3))))
        rec = self.linepair(best)
        rec["observation"] = (f"{len(members)} 行（{', '.join(map(str, members[:12]))}{' …' if len(members) > 12 else ''}）が、"
                              f"偶然より多くの2〜4記号列を互いに共有する（有意な行の組 {len(pairs)}）")
        rec["support"] = len(pairs)
        rec["support_lines"] = len(members)
        rec["measures"]["representative_pair"] = list(best)
        return rec

    def substitution(self, key):
        x, y = key
        L, R = defaultdict(set), defaultdict(set)
        for ln in self.lines:
            for i, s in enumerate(ln):
                if s in key:
                    L[s].add(ln[i - 1] if i else "^")
                    R[s].add(ln[i + 1] if i + 1 < len(ln) else "$")

        def nm(v):
            return v if isinstance(v, str) else self.c.name(v)
        sl, sr = L[x] & L[y], R[x] & R[y]
        both_lines = sum(1 for ln in self.lines if x in ln and y in ln)
        nx, ny = self.c.name(x), self.c.name(y)
        return {
            "observation": f"記号 {nx} と {ny} が同じ直前の文脈 {len(sl)} 種・同じ直後の文脈 {len(sr)} 種を共有する",
            "evidence": [{"shared_left": sorted(map(nm, sl)), "shared_right": sorted(map(nm, sr))}],
            "support": len(sl) + len(sr), "support_lines": None,
            "counterexamples": {"description": "片方だけに現れる文脈の数",
                                "count": len((L[x] ^ L[y])) + len((R[x] ^ R[y])),
                                "lines_with_both": both_lines},
            "measures": {"context_overlap": round((len(sl) + len(sr)) / max(1, len(L[x] | L[y]) + len(R[x] | R[y])), 3)},
            "symbols": [nx, ny],
        }

    def linepair(self, key):
        i, j = key
        a, b = self.lines[i], self.lines[j]
        shared = sorted({g for n in (2, 3, 4) for g in set(ngrams(a, n)) & set(ngrams(b, n))}, key=lambda g: (-len(g), g))
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        fixed = [self.c.names(a[m.a:m.a + m.size]) for m in sm.get_matching_blocks() if m.size]
        variable = [{"line_a": self.c.names(a[i1:i2]), "line_b": self.c.names(b[j1:j2])}
                    for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]
        return {
            "observation": f"行 {i} と行 {j} が {len(shared)} 種の2〜4記号列を共有する（一致率 {sm.ratio():.2f}）",
            "evidence": [{"line_a": self.loc(i, 0, len(a)), "line_b": self.loc(j, 0, len(b)),
                          "fixed_parts": fixed, "variable_parts": variable[:6]}],
            "support": len(shared), "support_lines": 2,
            "counterexamples": {"description": "一致しない部分の数", "count": len(variable)},
            "measures": {"sequence_ratio": round(sm.ratio(), 3)},
            "symbols": [],
        }


def templates(builder, pair_keys):
    """有意な行の組を連結成分にまとめ、共通の固定部分と変わる部分を出す。"""
    adj = defaultdict(set)
    for i, j in pair_keys:
        adj[i].add(j)
        adj[j].add(i)
    seen, groups = set(), []
    for v in sorted(adj):
        if v in seen:
            continue
        comp, stack = [], [v]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            stack.extend(adj[u] - seen)
        groups.append(sorted(comp))
    out = []
    lines = builder.lines
    for g in groups:
        base = lines[g[0]]
        keep = [True] * len(base)
        for o in g[1:]:
            sm = difflib.SequenceMatcher(a=base, b=lines[o], autojunk=False)
            mask = [False] * len(base)
            for m in sm.get_matching_blocks():
                for k in range(m.a, m.a + m.size):
                    mask[k] = True
            keep = [x and y for x, y in zip(keep, mask)]
        pattern = []
        for s, k in zip(base, keep):
            tok = builder.c.name(s) if k else "_"
            if tok == "_" and pattern and pattern[-1] == "_":
                continue
            pattern.append(tok)
        out.append({"lines": g, "pattern_on_first_line": " ".join(pattern),
                    "fixed_symbols": sum(keep), "variable_slots": pattern.count("_")})
    return out


def components(pairs):
    adj = defaultdict(set)
    for a, b in pairs:
        adj[a].add(b)
        adj[b].add(a)
    seen, out = set(), []
    for v in sorted(adj, key=str):
        if v in seen:
            continue
        comp, stack = [], [v]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            stack.extend(adj[u] - seen)
        out.append(comp)
    return out
