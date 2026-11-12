"""評価用コーパスの生成器。生成原理の異なる系統を用意する。

各生成器は、言語（シードで決まる）から行を標本し、正解（設計した規則・単語境界）を持つ。
母集団オラクル: 同じ言語から、分析に使った行とは別の行を大量に取り、仮説がその母集団でも成り立つかを確かめる。

系統（principle）
- 内部（このモジュールで設計）: random_uniform, random_zipf, regular, weak_regular, embedded_repeat, periodic, noisy_regular
- 外部の生成原理: pcfg_prefix（再帰的な句構造・接頭辞・動詞が先頭。このモジュールの他の生成器と規則の作り方が違う）
- BABEL 本体の生成器: babel_generator（babel.generator.corpus、解読器の評価用。構造解析のために作ったものではない）
- 第三者が作ったコーパス: external_gpt6（external/inbox の GPT-6 作成コーパス。読むだけで改変しない）
- 実在の言語: real_english（Python 標準ライブラリの The Zen of Python と、パブリックドメインの創世記1章 KJV）
"""
from __future__ import annotations

import contextlib
import io
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONS, VOWS = "bdgklmnprst", "aeiou"


def _stem(rng, lo=2, hi=3):
    n = rng.randint(lo, hi)
    return "".join(rng.choice(CONS) if i % 2 == 0 else rng.choice(VOWS) for i in range(n))


def _unique_stems(rng, k, lo=2, hi=3):
    out = set()
    while len(out) < k:
        out.add(_stem(rng, lo, hi))
    return sorted(out)


class Generator:
    name = ""
    principle = "internal"
    description = ""
    has_words = True
    max_lines = None          # 固定コーパスの行数の上限（None なら無制限に標本できる）

    def __init__(self, seed):
        self.seed = seed
        self.rng = random.Random(f"{self.name}-{seed}")
        self.setup()

    def setup(self):
        pass

    def line(self, rng):
        """1行 → (語のリスト, 文型の番号)。語は記号（1文字）の文字列。"""
        raise NotImplementedError

    def gold_rules(self):
        return []

    def gold_words(self):
        """行の組の仮説を検証するための、設計上の語（2記号以上）。"""
        return set()

    def sample(self, n, which="analysis"):
        rng = random.Random(f"{self.name}-{self.seed}-{which}")
        return [self.line(rng) for _ in range(n)]


# ------------------------------------------------------------------------------------------------ 無作為

class RandomUniform(Generator):
    name, has_words = "random_uniform", False
    description = "20種の記号を一様・独立に並べた行（長さ6〜14）。規則は存在しない"
    ALPHA = "abcdefghijklmnopqrst"

    def line(self, rng):
        return [rng.choice(self.ALPHA) for _ in range(rng.randint(6, 14))], None


class RandomZipf(Generator):
    name, has_words = "random_zipf", False
    description = "30種の記号を Zipf 分布で独立に並べた行。頻度の偏りはあるが並びの規則はない"
    ALPHA = "abcdefghijklmnopqrstuvwxyzABCD"

    def setup(self):
        self.w = [1 / (r + 1) ** 1.1 for r in range(len(self.ALPHA))]

    def line(self, rng):
        return list(rng.choices(self.ALPHA, self.w, k=rng.randint(6, 14))), None


# ------------------------------------------------------------------------------------------------ 規則のある人工言語

class Regular(Generator):
    name = "regular"
    description = "S 主語-x [形容詞] 目的語-q 動詞-v E の固定文型。文頭標識 S・文末標識 E・格接尾辞 x/q・動詞接尾辞 v"

    def setup(self):
        r = self.rng
        self.nouns = _unique_stems(r, 10)
        self.verbs = _unique_stems(r, 6)
        self.adjs = _unique_stems(r, 4, 2, 2)

    def line(self, rng):
        words = ["S", rng.choice(self.nouns) + "x"]
        if rng.random() < 0.5:
            words.append(rng.choice(self.adjs))
        words += [rng.choice(self.nouns) + "q", rng.choice(self.verbs) + "v", "E"]
        return words, "T"

    def gold_rules(self):
        return [{"family": "position", "side": "^", "symbols": ["S"], "label": "文頭標識 S"},
                {"family": "position", "side": "$", "symbols": ["E"], "label": "文末標識 E"},
                {"family": "ngram", "symbols": ["v", "E"], "label": "動詞接尾辞 v の直後に E"},
                {"family": "substitution", "symbols": ["x", "q", "v"], "label": "接尾辞 x/q/v が同じ枠に入る"},
                {"family": "affix", "side": "suffix", "symbols": ["x"], "label": "格接尾辞 x"},
                {"family": "affix", "side": "suffix", "symbols": ["q"], "label": "格接尾辞 q"},
                {"family": "dispersion", "side": "even", "symbols": ["x"], "label": "x は各行に1回（必須）"},
                {"family": "dispersion", "side": "even", "symbols": ["v"], "label": "v は各行に1回（必須）"}]

    def gold_words(self):
        return {n + "x" for n in self.nouns} | {n + "q" for n in self.nouns} | {v + "v" for v in self.verbs} | set(self.adjs)


class WeakRegular(Regular):
    name = "weak_regular"
    description = "Regular の弱い版: 標識は半分の行だけ、語順は無作為、格接尾辞の30%が入れ替わり、記号の5%が雑音"

    def setup(self):
        r = self.rng
        self.nouns = _unique_stems(r, 30)
        self.verbs = _unique_stems(r, 12)
        self.adjs = _unique_stems(r, 8, 2, 2)

    def line(self, rng):
        sx, oq = ("x", "q") if rng.random() > 0.3 else ("q", "x")
        content = [rng.choice(self.nouns) + sx, rng.choice(self.nouns) + oq, rng.choice(self.verbs) + "v"]
        if rng.random() < 0.5:
            content.append(rng.choice(self.adjs))
        rng.shuffle(content)
        words = (["S"] if rng.random() < 0.5 else []) + content + (["E"] if rng.random() < 0.5 else [])
        noisy = ["".join(rng.choice(CONS + VOWS) if rng.random() < 0.05 else ch for ch in w) for w in words]
        return noisy, None


class NoisyRegular(Regular):
    name = "noisy_regular"
    description = "Regular に転写の誤りを模した雑音: 記号の10%を別の記号に置換、5%を欠損"

    def line(self, rng):
        words, t = super().line(rng)
        out = []
        for w in words:
            chars = []
            for ch in w:
                u = rng.random()
                if u < 0.05:
                    continue
                chars.append(rng.choice(CONS + VOWS + "SExqv") if u < 0.15 else ch)
            if chars:
                out.append("".join(chars))
        return out, t


class EmbeddedRepeat(Generator):
    name, has_words = "embedded_repeat", False
    description = "一様な無作為列の約35%の行に、決まった6記号列を1回埋め込んだもの"
    ALPHA = "abcdefghijklmnopqrst"

    def setup(self):
        self.motif = "".join(self.rng.choice(self.ALPHA) for _ in range(6))

    def line(self, rng):
        s = [rng.choice(self.ALPHA) for _ in range(rng.randint(8, 16))]
        if rng.random() < 0.35:
            i = rng.randint(0, len(s) - 6)
            s[i:i + 6] = list(self.motif)
        return s, None

    def gold_rules(self):
        return [{"family": "ngram", "symbols": list(self.motif), "label": "埋め込んだ6記号列"}]

    def gold_words(self):
        return {self.motif}


class Periodic(Generator):
    name, has_words = "periodic", False
    description = "無作為列の 0,4,8,… 番目に記号 P を置いた行（各位置 90%）"
    ALPHA = "abcdefghijklmnop"

    def line(self, rng):
        L = rng.randint(9, 17)
        return ["P" if i % 4 == 0 and rng.random() < 0.9 else rng.choice(self.ALPHA) for i in range(L)], None

    def gold_rules(self):
        return [{"family": "gap", "symbols": ["P"], "distance": 4, "label": "P が4記号おき"}]


# ------------------------------------------------------------------------------------------------ 外部の生成原理

class PCFGPrefix(Generator):
    name, principle = "pcfg_prefix", "external-principle"
    description = ("再帰的な句構造文法（動詞が先頭・接頭辞で標示・等位接続と所有の入れ子）。"
                   "時制接頭辞 Σ/Τ、定冠接頭辞 Δ、複数接頭辞 Π、接続語 Κ、所有標識 Φ。記号はギリシア文字")
    STEMS = "αβγδεζηθικλμνξοπρυχψω"

    def setup(self):
        r = self.rng
        def stems(k):
            out = set()
            while len(out) < k:
                out.add("".join(r.choice(self.STEMS) for _ in range(r.randint(2, 4))))
            return sorted(out)
        self.n, self.v = stems(14), stems(8)

    def np(self, rng, depth=0):
        w = ["Δ" + ("Π" if rng.random() < 0.3 else "") + rng.choice(self.n)]
        if depth < 2 and rng.random() < 0.2:
            w += ["Φ"] + self.np(rng, depth + 1)
        return w

    def clause(self, rng):
        w = [rng.choice("ΣΤ") + rng.choice(self.v)] + self.np(rng)
        if rng.random() < 0.6:
            w += self.np(rng)
        return w

    def line(self, rng):
        w = self.clause(rng)
        if rng.random() < 0.3:
            w += ["Κ"] + self.clause(rng)
        return w, None

    def gold_rules(self):
        return [{"family": "position", "side": "^", "symbols": ["Σ"], "label": "文頭の時制接頭辞 Σ"},
                {"family": "position", "side": "^", "symbols": ["Τ"], "label": "文頭の時制接頭辞 Τ"},
                {"family": "substitution", "symbols": ["Σ", "Τ"], "label": "時制接頭辞 Σ/Τ が同じ枠"},
                {"family": "affix", "side": "prefix", "symbols": ["Δ"], "label": "定冠接頭辞 Δ"}]

    def gold_words(self):
        return set()


# ------------------------------------------------------------------------------------------------ 固定コーパス

class FixedCorpus(Generator):
    """行の集合が決まっているコーパス。シードで分析用の行を選び、残りを母集団（オラクル）に使う。"""
    lines_all: list = []

    def setup(self):
        self.lines_all = self.load()
        self.max_lines = len(self.lines_all) // 2
        order = list(range(len(self.lines_all)))
        random.Random(f"{self.name}-{self.seed}-order").shuffle(order)
        self.order = order
        words = set()
        for ws, _ in self.lines_all:
            words.update(w for w in ws if len(w) >= 2)
        self.words = words

    def load(self):
        raise NotImplementedError

    def sample(self, n, which="analysis"):
        if which == "analysis":
            return [self.lines_all[i] for i in self.order[:min(n, self.max_lines)]]
        return [self.lines_all[i] for i in self.order[self.max_lines:]]

    def gold_words(self):
        return self.words


class BabelGenerator(FixedCorpus):
    name, principle = "babel_generator", "babel-own-generator"
    description = "BABEL 本体の生成器（babel.generator、系統 KES/SOT/… をシードで切り替え、LEVEL C、欠損なし）"
    FAMILIES = ["KES", "UMB", "SOT"]

    def load(self):
        from babel.generator.corpus import build_corpus
        from babel.generator.languages import STRUCTURE
        fams = [f for f in self.FAMILIES if f in STRUCTURE] or sorted(STRUCTURE)
        fam = fams[self.seed % len(fams)]
        self.description += f"（この標本は {fam}）"
        pub, _ = build_corpus(fam, 5000 + self.seed, "C", n_docs=60, lacuna=0.0, boundaries=True)
        return [(s.split(), None) for d in pub["documents"] for s in d["sentences"] if s.strip()]


class ExternalGPT6(FixedCorpus):
    name, principle = "external_gpt6", "third-party"
    description = "第三者（GPT-6）が作成した外部コーパス（external/inbox、読み取りのみ）。シードで2つのコーパスを切り替え"
    SOURCES = ["gpt6-blind-01", "outsider-vso-prefix-7-v1"]

    def load(self):
        src = self.SOURCES[self.seed % len(self.SOURCES)]
        self.description += f"（この標本は {src}）"
        pub = json.loads((ROOT / "external" / "inbox" / src / "corpus.json").read_text(encoding="utf-8"))
        return [(s.split(), None) for d in pub["documents"] for s in d["sentences"] if s.strip()]


def english_lines():
    with contextlib.redirect_stdout(io.StringIO()):
        import this
        zen = "".join(this.d.get(c, c) for c in this.s)
    text = zen.splitlines()[2:] + (Path(__file__).parent / "data" / "genesis1_kjv.txt").read_text(encoding="utf-8").splitlines()
    out = []
    for ln in text:
        words = ["".join(ch for ch in w.lower() if ch.isalpha()) for w in ln.split()]
        words = [w for w in words if w]
        if words:
            out.append((words, None))
    return out


class RealEnglish(FixedCorpus):
    name, principle = "real_english", "real-language"
    description = "実在の言語: The Zen of Python（標準ライブラリ）と創世記1章（KJV、パブリックドメイン）。英字26種、空白を除去"

    def load(self):
        return english_lines()


GENERATORS = {g.name: g for g in (RandomUniform, RandomZipf, Regular, WeakRegular, NoisyRegular, EmbeddedRepeat, Periodic,
                                  PCFGPrefix, BabelGenerator, ExternalGPT6, RealEnglish)}
NULL_GENERATORS = {"random_uniform", "random_zipf"}


def to_corpus(rows, name):
    """解析器への入力: 単語境界を除いた記号列だけ（正解は渡さない）。"""
    return {"language_id": name, "level": "C", "has_word_boundaries": False, "known": [], "parallel": [],
            "documents": [{"id": f"l{i:03d}", "meta": {}, "sentences": ["".join(ws)]} for i, (ws, _) in enumerate(rows)]}


def gold_cuts(rows):
    out = []
    for ws, _ in rows:
        cuts, k = set(), 0
        for w in ws[:-1]:
            k += len(w)
            cuts.add(k)
        out.append(cuts)
    return out
