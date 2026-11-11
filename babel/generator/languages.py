"""3系統の人工言語。語形・接辞形はシードごとに音韻規則から生成する。

KES: 膠着・接尾辞・母音調和・SOV・10進
UMB: 接頭辞型・VSO・20進
SOT: 屈折（融合語尾・2変化類・一致）・SVO（自由語順）・12進・不規則動詞
"""
from __future__ import annotations

import random

from .world import ADJECTIVES, AGENTIVE, COMPOUNDS, VERBS, all_concepts, sem_class

ROLE_CASE = {"agent": "NOM", "theme": "NOM", "patient": "ACC", "recipient": "DAT",
             "location": "LOC", "time": "LOC"}
CASE_FEAT = {"NOM": "R:subject", "ACC": "R:object", "DAT": "R:recipient", "LOC": "R:location",
             "GEN": "R:possessor"}
TENSE_FEAT = {"PAST": "T:PAST", "PRES": "T:PRES", "FUT": "T:FUT"}

PHON = {
    "KES": dict(C=list("ptkbdgmnszlrhy"), V=list("aeiou")),
    "UMB": dict(C=["t", "k", "q", "'", "m", "n", "ng", "f", "s", "sh", "x", "l", "w"],
                V=["a", "i", "u", "a", "i", "u", "aa", "ii", "uu"]),
    "SOT": dict(C=["p", "t", "c", "b", "d", "g", "f", "v", "s", "r", "l", "m", "n", "th"],
                V=["a", "e", "i", "o", "u", "ae"]),
}
# 書記の表記習慣の候補 / 時代による音変化の候補: (種類, 元, 先)
SCRIBE_POOL = {
    "KES": [("sub", "k", "q"), ("sub", "y", "j"), ("sub", "h", "x"), ("final", "r", "rr"), ("sub", "z", "ts")],
    "UMB": [("sub", "sh", "c"), ("sub", "'", ""), ("sub", "ng", "g"), ("sub", "x", "kh"), ("sub", "w", "v")],
    "SOT": [("sub", "c", "k"), ("sub", "v", "w"), ("final", "m", "n"), ("sub", "f", "ph"), ("sub", "ae", "e")],
}
SOUND_POOL = {
    "KES": [("sub", "p", "f"), ("final", "u", "o"), ("sub", "b", "v")],
    "UMB": [("sub", "q", "k"), ("sub", "f", "h"), ("final", "i", "e")],
    "SOT": [("sub", "th", "d"), ("sub", "g", "gh"), ("final", "s", "h")],
}
STRUCTURE = {
    "KES": dict(verb_position="final", so_order="SO", affix_position="suffix", numeral_base=10,
                adj_noun="AN", num_noun="NumN"),
    "UMB": dict(verb_position="initial", so_order="SO", affix_position="prefix", numeral_base=20,
                adj_noun="NA", num_noun="NNum"),
    "SOT": dict(verb_position="medial", so_order="SO", affix_position="suffix", numeral_base=12,
                adj_noun="AN", num_noun="NumN"),
}
FAMILIES = ["KES", "UMB", "SOT"]


def apply_rules(morphs, rules):
    """音変化・表記規則を形態素ごとに適用する（形態素境界を保つ）。"""
    out = [list(m) for m in morphs]
    for kind, x, y in rules:
        if kind == "sub":
            for m in out:
                m[0] = m[0].replace(x, y)
        elif kind == "final" and out and out[-1][0].endswith(x):
            out[-1][0] = out[-1][0][: -len(x)] + y
    return out


class Language:
    def __init__(self, family: str, seed: int):
        assert family in FAMILIES
        self.family, self.seed = family, seed
        self.rng = r = random.Random(f"lang-{family}-{seed}")
        self.C, self.V = PHON[family]["C"], PHON[family]["V"]
        self.used: set[str] = set()
        c = all_concepts()
        self.roots: dict[str, str] = {}
        for con in c["function"]:
            self.roots[con] = self.new_root(1, func=True)
        for con in c["nouns"] + c["verbs"] + c["adjs"]:
            self.roots[con] = self.new_root(r.choice([2, 2, 3]) if family != "SOT" else r.choice([1, 2, 2]))
        for con in c["persons"] + c["places"] + c["gods"]:
            self.roots[con] = self.new_root(r.choice([2, 3]))
        # 同音異義語: 2つの概念が同じ語根を共有する
        nouns = [n for n in c["nouns"] if n not in ("year",)]
        self.homonyms = [tuple(r.sample(nouns, 2)), tuple(r.sample(nouns, 2)),
                         (r.choice(nouns), r.choice(c["verbs"]))]
        for a, b in self.homonyms:
            self.roots[b] = self.roots[a]
        # 後期の語彙交替
        self.late = {con: self.new_root(2) for con in r.sample(c["nouns"][:60] + c["verbs"], 3)}
        self.ncls = {con: r.choice([1, 2]) for con in c["nouns"] + c["persons"] + c["places"] + c["gods"]}
        self.irr = {v: self.new_root(1) for v in ["give", "go", "see", "die"]} if family == "SOT" else {}
        self.make_affixes()
        self.make_numerals()
        self.sound = r.sample(SOUND_POOL[family], 2)       # era>=1 で1つ目、era==2 で2つ目
        self.scribe_rules = r.sample(SCRIBE_POOL[family], 3)

    # ---- 音韻 ----
    def syll(self):
        return self.rng.choice(self.C) + self.rng.choice(self.V)

    def new_root(self, n, func=False):
        for _ in range(10000):
            s = "".join(self.syll() for _ in range(n))
            if self.family == "SOT" or (self.family == "KES" and (func or self.rng.random() < 0.3)):
                s += self.rng.choice(self.C)
            if self.family == "UMB" and func and len(s) < 3:
                s += self.syll()
            if s not in self.used and len(s) >= 2:
                self.used.add(s)
                return s
        raise RuntimeError("root space exhausted")

    def new_affix(self, shapes, vowels=None, cons=None):
        vowels, cons = vowels or self.V, cons or self.C
        for _ in range(10000):
            shape = self.rng.choice(shapes)
            s = "".join(self.rng.choice(cons) if ch == "C" else self.rng.choice(vowels) for ch in shape)
            if s not in self.used:
                self.used.add(s)
                return s
        raise RuntimeError("affix space exhausted")

    def make_affixes(self):
        f, a = self.family, {}
        if f == "KES":  # 母音調和: A = a/e, U = u/i
            for k in ["PL", "ACC", "GEN", "DAT", "LOC", "NEG", "PAST", "PRES", "FUT", "VPL", "AGT", "ACC2"]:
                a[k] = self.new_affix(["CV", "VC", "CVC"], vowels=["A", "U"])
        elif f == "UMB":
            for k in ["ACC", "GEN", "DAT", "LOC", "PL", "PAST", "PRES", "FUT", "AGT", "VPL", "PL2"]:
                a[k] = self.new_affix(["CV"])
        else:
            for cls in (1, 2):
                for case in ["NOM", "ACC", "GEN", "DAT", "LOC"]:
                    for num in ["SG", "PL"]:
                        a[(cls, case, num)] = self.new_affix(["V", "VC", "VC", "VCV"], vowels=list("aeiou"),
                                                             cons=list("smrnt"))
            a[(1, "LOC", "PL")] = a[(1, "DAT", "PL")]   # 語尾の融合（シンクレティズム）
            a[(2, "NOM", "PL")] = a[(2, "GEN", "SG")]
            for t in ["PAST", "PRES", "FUT"]:
                for num in ["SG", "PL"]:
                    a[(t, num)] = self.new_affix(["VC", "VCV", "VCC"], vowels=list("aeiou"), cons=list("ltcdn"))
            a["AGT"] = self.new_affix(["VC"], vowels=list("aeiou"), cons=list("rlc"))
        self.aff = a

    def make_numerals(self):
        n_atoms = {"KES": 9, "UMB": 9, "SOT": 11}[self.family]
        self.units = {i: self.new_root(self.rng.choice([1, 1, 2])) for i in range(1, n_atoms + 1)}
        self.big = {"KES": {10: self.new_root(1), 100: self.new_root(1)},
                    "UMB": {10: self.new_root(1), 20: self.new_root(1)},
                    "SOT": {12: self.new_root(1), 144: self.new_root(1)}}[self.family]

    # ---- 語形成 ----
    def harm(self, template, before):
        if self.family != "KES":
            return template
        last = next((ch for ch in reversed(before) if ch in "aeiou"), "a")
        back = last in "aou"
        return template.replace("A", "a" if back else "e").replace("U", "u" if back else "i")

    def suf(self, key, morphs):
        return self.harm(self.aff[key], "".join(m[0] for m in morphs))

    def root(self, concept, era):
        return self.late[concept] if era == 2 and concept in self.late else self.roots[concept]

    def stem(self, concept, era, deriv=None):
        f = self.family
        if "+" in concept:
            a, b = concept.split("+")
            ma, mb = [self.root(a, era), "stem", a], [self.root(b, era), "stem", b]
            if f == "KES":
                return [ma, mb]
            if f == "UMB":
                return [mb, ma]
            return [ma, ["o", "link", "LINK"], mb]
        m = [self.root(concept, era), "stem", concept]
        if deriv == "AGENT":
            if f == "UMB":
                return [[self.aff["AGT"], "affix", "AGT"], m]
            return [m, [self.suf("AGT", [m]), "affix", "AGT"]]
        return [m]

    def word(self, morphs, concept, pos, feats=(), sem=None):
        return {"morphs": morphs, "concept": concept, "pos": pos, "feats": sorted(feats),
                "sem": sem or sem_class(concept)}

    def noun_class(self, concept, deriv):
        if deriv:
            return 2
        return self.ncls.get(concept.split("+")[-1], 1)

    def noun(self, concept, case, pl, era, deriv=None, bare=False):
        f, ms, feats = self.family, self.stem(concept, era, deriv), set()
        if deriv:
            feats.add("D:AGENT")
        sem = "human" if deriv else next((c[2] for c in COMPOUNDS if "+".join(c[:2]) == concept), None)
        if bare and f != "SOT":
            return self.word(ms, concept, "N", feats, sem)
        if f == "KES":
            if pl:
                ms.append([self.suf("PL", ms), "affix", "PL"])
                feats.add("N:PL")
            if case != "NOM":
                key = "ACC2" if case == "ACC" and era == 2 else case
                ms.append([self.suf(key, ms), "affix", case])
                feats.add(CASE_FEAT[case])
        elif f == "UMB":
            pre = []
            if case != "NOM":
                pre.append([self.aff[case], "affix", case])
                feats.add(CASE_FEAT[case])
            if pl:
                pre.append([self.aff["PL2" if era == 2 else "PL"], "affix", "PL"])
                feats.add("N:PL")
            ms = pre + ms
        else:
            ms.append([self.sot_ending(self.noun_class(concept, deriv), case, pl, era), "affix",
                       f"{case}.{'PL' if pl else 'SG'}"])
            feats.add(CASE_FEAT[case])
            if pl:
                feats.add("N:PL")
        return self.word(ms, concept, "N", feats, sem)

    def sot_ending(self, cls, case, pl, era):
        if era == 2 and case == "LOC" and not pl:   # 後期: 所格単数が与格単数に合流
            case = "DAT"
        return self.aff[(cls, case, "PL" if pl else "SG")]

    def adjective(self, concept, head_concept, case, pl, era, deriv):
        f, ms, feats = self.family, [[self.root(concept, era), "stem", concept]], set()
        if f == "UMB" and pl:
            ms = [[self.aff["PL2" if era == 2 else "PL"], "affix", "PL"]] + ms
            feats.add("N:PL")
        if f == "SOT":
            ms.append([self.sot_ending(self.noun_class(head_concept, deriv), case, pl, era), "affix",
                       f"{case}.{'PL' if pl else 'SG'}"])
            feats.add(CASE_FEAT[case])
            if pl:
                feats.add("N:PL")
        return self.word(ms, concept, "ADJ", feats)

    def number(self, n):
        f, words = self.family, []

        def w(morphs, val):
            words.append(self.word(morphs, f"#{val}", "NUM", (), "numeral"))

        def unit(i):
            return [self.units[i], "stem", f"#{i}"]

        if f == "KES":
            for base, digit in ((100, n // 100), (10, n % 100 // 10)):
                if digit:
                    b = [self.big[base], "stem", f"x{base}"]
                    w([b] if digit == 1 else [unit(digit), b], digit * base)
            if n % 10:
                w([unit(n % 10)], n % 10)
        elif f == "UMB":
            ten = [self.big[10], "stem", "#10"]

            def small(k):
                return [unit(k)] if k < 10 else ([ten] if k == 10 else [ten, unit(k - 10)])
            s, rem = n // 20, n % 20
            if s:
                sc = [self.big[20], "stem", "x20"]
                w([sc] if s == 1 else small(s) + [sc], s * 20)
            if rem:
                w(small(rem), rem)
        else:
            for base, digit in ((144, n // 144), (12, n % 144 // 12)):
                if digit:
                    b = [self.big[base], "stem", f"x{base}"]
                    w([b] if digit == 1 else [unit(digit), b], digit * base)
            if n % 12:
                w([unit(n % 12)], n % 12)
        return words

    def verb(self, concept, tense, neg, pl, era):
        f, feats = self.family, {TENSE_FEAT[tense]}
        root = [self.root(concept, era), "stem", concept]
        pre = []
        if f == "KES":
            ms = [root]
            if neg:
                ms.append([self.suf("NEG", ms), "affix", "NEG"])
                feats.add("NEG")
            ms.append([self.suf(tense, ms), "affix", tense])
            if pl:
                ms.append([self.suf("VPL", ms), "affix", "PL"])
                feats.add("N:PL")
        elif f == "UMB":
            ms = [[self.aff[tense], "affix", tense], root]
            if pl:
                ms.append([self.aff["VPL"], "affix", "PL"])
                feats.add("N:PL")
        else:
            if tense == "PAST" and concept in self.irr:
                root = [self.irr[concept], "stem", concept]
            ms = [root, [self.aff[(tense, "PL" if pl else "SG")], "affix", f"{tense}.{'PL' if pl else 'SG'}"]]
            if pl:
                feats.add("N:PL")
        if neg and f != "KES":
            pre.append(self.word([[self.roots["not"], "stem", "not"]], "not", "FUNC", ("NEG",)))
        return pre, self.word(ms, concept, "V", feats)

    # ---- 統語 ----
    def np(self, np, case, era):
        f = self.family
        pl = bool(np["pl"]) or (np["count"] or 0) > 1
        mark_pl = pl
        if np["count"] is not None:
            if f == "KES":
                mark_pl = False                 # 数詞の後は単数形
            elif f == "SOT" and np["count"] > 1:
                case, mark_pl = "GEN", True     # 数詞の後は属格複数
        head = self.noun(np["head"], case, mark_pl, era, np["deriv"])
        poss = self.np(np["poss"], "GEN", era) if np["poss"] else []
        num = self.number(np["count"]) if np["count"] is not None else []
        adj = [self.adjective(np["adj"], np["head"], case, mark_pl, era, np["deriv"])] if np["adj"] else []
        title = []
        if np["title"]:
            title = [self.noun(np["title"], case, False, era, bare=True)]
        if f == "KES":
            return poss + num + adj + title + [head]
        if f == "UMB":
            return title + [head] + adj + num + poss
        return num + adj + title + [head] + poss

    def render(self, prop, era, rng):
        f = self.family
        parts = {a["role"]: self.np(a["np"], ROLE_CASE[a["role"]], era) for a in prop["args"]}
        if prop["kind"] == "total":
            t = [self.word([[self.roots["total"], "stem", "total"]], "total", "FUNC")]
            return parts["theme"] + t if f == "UMB" else t + parts["theme"]
        if prop["kind"] == "heading":
            return parts["time"]
        subj_np = next(a["np"] for a in prop["args"] if a["role"] == "agent")
        pl = bool(subj_np["pl"]) or (subj_np["count"] or 0) > 1
        pre, v = self.verb(prop["pred"], prop["tense"], prop["neg"], pl, era)
        s, o = parts.get("agent", []), parts.get("patient", [])
        rec, loc = parts.get("recipient", []), parts.get("location", [])
        cond = [self.word([[self.roots["if"], "stem", "if"]], "if", "FUNC")] if prop["cond"] else []
        if f == "KES":
            return cond + loc + s + rec + o + pre + [v]
        if f == "UMB":
            return cond + pre + [v] + s + o + rec + loc
        if o and rng.random() < 0.2:
            return cond + o + pre + [v] + s + rec + loc     # 自由語順（OVS）
        return cond + s + pre + [v] + o + rec + loc

    # ---- 正解情報 ----
    def answer_lexicon(self):
        out = []
        for con, form in sorted(self.roots.items()):
            pos = "V" if con in VERBS else "ADJ" if con in ADJECTIVES else \
                "FUNC" if sem_class(con) == "function" else "N"
            out.append({"form": form, "meaning": con, "pos": pos, "sem": sem_class(con)})
        return out

    def numeral_atoms(self):
        d = {form: v for v, form in self.units.items()}
        d.update({form: v for v, form in self.big.items()})
        return [{"form": k, "value": v} for k, v in sorted(d.items())]

    def describe(self):
        aff = {str(k): v for k, v in self.aff.items()}
        return {"family": self.family, "seed": self.seed, "structure": STRUCTURE[self.family],
                "affixes": aff, "homonyms": [list(h) for h in self.homonyms],
                "late_replacements": sorted(self.late), "sound_changes": self.sound,
                "scribe_rules": self.scribe_rules, "irregular_verbs": sorted(self.irr),
                "agentive_verbs": AGENTIVE}
