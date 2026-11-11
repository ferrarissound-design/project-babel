"""第4系統 NAX（準初見テスト用）。解読器の凍結（コミット c34d6db）より後に作成した。

開発中の3系統に無い性質を意図的に入れてある:
  - 複数は語頭音節の重複（接辞ではなく語幹に依存する形態過程）
  - 主格が有標・対格が無標（有標主格型）
  - 動詞・目的語・主語の語順（VOS）
  - 8進法。倍数は「八＋数字」の順（基数が先）
  - 否定は動詞の後ろの小辞、時制は接尾辞
同じAIが設計しているため独立な初見テストではない。
"""
from __future__ import annotations

from . import languages as L
from .languages import CASE_FEAT, ROLE_CASE, TENSE_FEAT, Language
from .world import COMPOUNDS

FAMILY = "NAX"
L.PHON[FAMILY] = dict(C=["p", "t", "k", "m", "n", "s", "h", "l", "r", "w", "y", "ch"], V=list("aeiou"))
L.SCRIBE_POOL[FAMILY] = [("sub", "ch", "c"), ("sub", "w", "u"), ("sub", "y", "i"), ("final", "h", ""), ("sub", "r", "rh")]
L.SOUND_POOL[FAMILY] = [("sub", "p", "b"), ("sub", "s", "z"), ("final", "e", "i")]
L.STRUCTURE[FAMILY] = dict(verb_position="initial", so_order="OS", affix_position="suffix", numeral_base=8,
                           adj_noun="NA", num_noun="NNum")


class Nax(Language):
    def new_root(self, n, func=False):
        for _ in range(10000):
            s = "".join(self.syll() for _ in range(n))
            if func or self.rng.random() < 0.4:
                s += self.rng.choice(["n", "s", "l", "k"])
            if s not in self.used and len(s) >= 2:
                self.used.add(s)
                return s
        raise RuntimeError("root space exhausted")

    def make_affixes(self):
        a = {}
        for k in ["NOM", "GEN", "DAT", "LOC", "PAST", "PRES", "FUT", "VPL", "AGT", "NOM2"]:
            a[k] = self.new_affix(["VC", "CV", "VCV"])
        self.aff = a

    def make_numerals(self):
        self.units = {i: self.new_root(self.rng.choice([1, 1, 2])) for i in range(1, 8)}
        self.big = {8: self.new_root(1), 64: self.new_root(1)}

    def number(self, n):
        words = []

        def w(morphs, val):
            words.append(self.word(morphs, f"#{val}", "NUM", (), "numeral"))

        for base, digit in ((64, n // 64), (8, n % 64 // 8)):
            if digit:
                b = [self.big[base], "stem", f"x{base}"]
                w([b] if digit == 1 else [b, [self.units[digit], "stem", f"#{digit}"]], digit * base)   # 基数が先
        if n % 8:
            w([[self.units[n % 8], "stem", f"#{n % 8}"]], n % 8)
        return words

    def _redup(self, form):
        i = next((k for k, ch in enumerate(form) if ch in "aeiou"), 0)
        return form[: i + 1]

    def noun(self, concept, case, pl, era, deriv=None, bare=False):
        ms, feats = self.stem(concept, era, deriv), set()
        if deriv:
            feats.add("D:AGENT")
        sem = "human" if deriv else next((c[2] for c in COMPOUNDS if "+".join(c[:2]) == concept), None)
        if bare:
            return self.word(ms, concept, "N", feats, sem)
        if pl:
            ms = [[self._redup(ms[0][0]), "affix", "PL"]] + ms          # 語頭音節の重複
            feats.add("N:PL")
        if case != "ACC":                                               # 対格が無標
            key = "NOM2" if case == "NOM" and era == 2 else case
            ms.append([self.aff[key], "affix", case])
            feats.add(CASE_FEAT[case])
        return self.word(ms, concept, "N", feats, sem)

    def stem(self, concept, era, deriv=None):
        if "+" in concept:
            a, b = concept.split("+")
            return [[self.root(b, era), "stem", b], [self.root(a, era), "stem", a]]     # 主要部が先
        m = [self.root(concept, era), "stem", concept]
        return [m, [self.aff["AGT"], "affix", "AGT"]] if deriv == "AGENT" else [m]

    def adjective(self, concept, head_concept, case, pl, era, deriv):
        return self.word([[self.root(concept, era), "stem", concept]], concept, "ADJ", ())

    def verb(self, concept, tense, neg, pl, era):
        feats = {TENSE_FEAT[tense]}
        ms = [[self.root(concept, era), "stem", concept], [self.aff[tense], "affix", tense]]
        if pl:
            ms.append([self.aff["VPL"], "affix", "PL"])
            feats.add("N:PL")
        post = [self.word([[self.roots["not"], "stem", "not"]], "not", "FUNC", ("NEG",))] if neg else []
        return post, self.word(ms, concept, "V", feats)

    def np(self, np, case, era):
        pl = bool(np["pl"]) or (np["count"] or 0) > 1
        head = self.noun(np["head"], case, pl, era, np["deriv"])
        poss = self.np(np["poss"], "GEN", era) if np["poss"] else []
        num = self.number(np["count"]) if np["count"] is not None else []
        adj = [self.adjective(np["adj"], np["head"], case, pl, era, np["deriv"])] if np["adj"] else []
        title = [self.noun(np["title"], case, False, era, bare=True)] if np["title"] else []
        return poss + [head] + title + adj + num

    def render(self, prop, era, rng):
        parts = {a["role"]: self.np(a["np"], ROLE_CASE[a["role"]], era) for a in prop["args"]}
        if prop["kind"] == "total":
            return parts["theme"] + [self.word([[self.roots["total"], "stem", "total"]], "total", "FUNC")]
        if prop["kind"] == "heading":
            return parts["time"]
        subj = next(a["np"] for a in prop["args"] if a["role"] == "agent")
        post, v = self.verb(prop["pred"], prop["tense"], prop["neg"], bool(subj["pl"]) or (subj["count"] or 0) > 1, era)
        cond = [self.word([[self.roots["if"], "stem", "if"]], "if", "FUNC")] if prop["cond"] else []
        return cond + [v] + post + parts.get("patient", []) + parts.get("recipient", []) + \
            parts.get("location", []) + parts.get("agent", [])


L.HOLDOUT_CLASSES[FAMILY] = Nax
