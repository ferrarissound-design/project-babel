"""Name / loanword anchors: learn a symbol→letter correspondence from translations alone.

Decipherers have long entered unknown scripts through proper names that are
pronounced alike in a known language (royal names in cartouches, place names).
This module does the same with public data only: the English concept strings
of the structured translations and the target symbol strings of the same
sentences. Nothing about the cipher or the language is given.

Model: for an English concept ``c`` and a target word ``w`` the anchor score is
the best semi-global alignment of ``c`` against ``w`` (free trailing
characters of ``w`` for inflection, penalised leading skips/gaps) under
log-ratios log M(x|e)/P(x). M is initialised from position-wise co-occurrence
of letters and symbols over all (concept, word) pairs in translated sentences,
then re-estimated by EM where each concept occurrence chooses one word or NULL.

Limits (reported, not hidden): it only helps when the script is phonographic
and the language shares names or loanwords with English; on unrelated scripts
it should stay silent, which the null-control (mismatched translations) and
per-language results check.
"""
from __future__ import annotations

import math
from collections import defaultdict

MIN_CONCEPT = 4
GAP = -2.5
LEAD_SKIP = -1.5
MAX_LEAD = 2
LACUNA = "[…]"
NULL_SHIFT = 7          # translations moved to another sentence give the chance distribution of anchor scores
NULL_QUANTILE = 0.99


class NameAnchors:
    def __init__(self, sentences, occurrences, iterations=6, prior=0.15, smoothing=2.0, null_control=True):
        """occurrences: [(item_index, role, concept, sentence_index)]."""
        self.sentences = sentences
        self.occ = [o for o in occurrences if o[2] and len(o[2]) >= MIN_CONCEPT and o[2].isalpha()]
        self.iterations, self.prior, self.smoothing = iterations, prior, smoothing
        sym = defaultdict(float)
        for s in sentences:
            for w in s:
                for x in w:
                    sym[x] += 1
        tot = sum(sym.values()) or 1.0
        self.p_sym = {x: v / tot for x, v in sym.items()}
        self.M = None
        self.posterior = {}
        self.null_threshold = None
        self.C, self.T = {}, {}
        self.taught = {}
        self.null_control = null_control
        self.best_scores = []
        self.null_scores = []

    def _candidates(self, concept, sent):
        out = []
        for j, w in enumerate(sent):
            if w == LACUNA or not w.isalpha():
                continue
            if len(w) >= 3 and len(w) >= len(concept) - 2 and len(w) <= len(concept) + 6:
                out.append(j)
        return out

    def _logratio(self, e, x, own=None):
        pb = self.p_sym.get(x, 1e-6)
        if own and e in own[1]:
            # leave-one-out: the letter table without this very (concept, word) pair
            c = self.C.get(e, {}).get(x, 0.0) - own[0].get((e, x), 0.0)
            t = self.T.get(e, 0.0) - own[1][e]
            p = (max(0.0, c) + self.smoothing * pb) / (max(0.0, t) + self.smoothing)
        else:
            row = self.M.get(e)
            p = row.get(x, 0.0) if row else 0.0
        return math.log((p + 1e-4) / (pb + 1e-4))

    def align(self, concept, word, backtrace=False, own=None):
        """Best score of aligning all of ``concept`` to a prefix region of ``word``."""
        n, m = len(concept), len(word)
        neg = -1e9
        # D[i][j]: best score having consumed concept[:i] and word[:j]
        D = [[neg] * (m + 1) for _ in range(n + 1)]
        B = [[None] * (m + 1) for _ in range(n + 1)] if backtrace else None
        for j in range(0, min(MAX_LEAD, m) + 1):
            D[0][j] = LEAD_SKIP * j
        for i in range(1, n + 1):
            e = concept[i - 1]
            for j in range(0, m + 1):
                best, arg = D[i - 1][j] + GAP, (i - 1, j, None)          # letter with no symbol
                if j:
                    s = D[i - 1][j - 1] + self._logratio(e, word[j - 1], own)
                    if s > best:
                        best, arg = s, (i - 1, j - 1, (e, word[j - 1]))
                    s = D[i][j - 1] + GAP                                   # extra symbol
                    if s > best:
                        best, arg = s, (i, j - 1, None)
                D[i][j] = best
                if backtrace:
                    B[i][j] = arg
        # free trailing symbols of the word (inflection), mild preference for short tails
        jbest = max(range(m + 1), key=lambda j: (D[n][j] - 0.3 * (m - j), -j))
        score = D[n][jbest] - 0.3 * (m - jbest)
        if not backtrace:
            return score, None
        pairs, i, j = [], n, jbest
        while i > 0:
            pi, pj, pair = B[i][j]
            if pair:
                pairs.append(pair)
            i, j = pi, pj
        return score, pairs

    def _init(self):
        cnt = defaultdict(lambda: defaultdict(float))
        for _, _, concept, si in self.occ:
            sent = self.sentences[si]
            cands = self._candidates(concept, sent)
            if not cands:
                continue
            w = 1.0 / len(cands)
            for j in cands:
                word = sent[j]
                for k in range(min(len(concept), len(word))):
                    cnt[concept[k]][word[k]] += w
        self.M = self._normalise(cnt)

    def _normalise(self, cnt):
        M = {}
        self.C = {e: dict(row) for e, row in cnt.items()}
        self.T = {e: sum(row.values()) for e, row in cnt.items()}
        for e, row in cnt.items():
            tot = sum(row.values())
            M[e] = {x: (v + self.smoothing * self.p_sym.get(x, 0.0)) / (tot + self.smoothing)
                    for x, v in row.items()}
        return M

    def _own(self, concept, word):
        rec = self.taught.get((concept, word))
        if not rec:
            return None
        p, pairs = rec
        pair_w, tot = defaultdict(float), defaultdict(float)
        for e, x in pairs:
            pair_w[(e, x)] += p
            tot[e] += p
        return pair_w, tot

    def _null_threshold(self):
        """Chance level of the best anchor score, from a full re-fit on translations moved to other sentences.

        The shadow model is trained exactly like the real one (same EM, same leave-one-out scoring), so the
        inflation of scores by fitting is present on both sides of the comparison.
        """
        sents = sorted({si for *_, si in self.occ})
        if len(sents) < 2:
            return float("inf")
        pos = {si: k for k, si in enumerate(sents)}
        shifted = [(item, role, concept, sents[(pos[si] + NULL_SHIFT) % len(sents)])
                   for item, role, concept, si in self.occ]
        shadow = NameAnchors(self.sentences, shifted, self.iterations, self.prior, self.smoothing, null_control=False)
        shadow.fit()
        self.null_scores = sorted(shadow.best_scores)
        if len(self.null_scores) < 20:
            return float("inf")
        return self.null_scores[min(len(self.null_scores) - 1, int(NULL_QUANTILE * len(self.null_scores)))]

    def fit(self):
        if not self.occ:
            return self
        self._init()
        for it in range(self.iterations + 1):
            last = it == self.iterations
            if last:
                self.null_threshold = self._null_threshold() if self.null_control else float("-inf")
            cnt = defaultdict(lambda: defaultdict(float))
            distinct = {}   # each distinct (concept, word) pair teaches the letter table once
            post = {}
            for item, role, concept, si in self.occ:
                sent = self.sentences[si]
                cands = self._candidates(concept, sent)
                if not cands:
                    continue
                scored = [(j, *self.align(concept, sent[j], backtrace=not last,
                                          own=self._own(concept, sent[j]) if last else None)) for j in cands]
                if last:
                    self.best_scores.append(max(sc for _, sc, _ in scored))
                    # only scores above the chance distribution may become anchors
                    scored = [x for x in scored if x[1] > self.null_threshold]
                    if not scored:
                        continue
                # score is a log-likelihood ratio against chance; posterior vs NULL
                mx = max(0.0, max(sc for _, sc, _ in scored))
                z_null = (1 - self.prior) * math.exp(-mx)
                ws = [(j, self.prior / len(cands) * math.exp(sc - mx), pairs) for j, sc, pairs in scored]
                z = z_null + sum(w for _, w, _ in ws)
                for j, w, pairs in ws:
                    p = w / z
                    if p > 1e-3:
                        post[(item, role, j)] = p
                        key = (concept, sent[j])
                        if pairs and p > distinct.get(key, (0.0, None))[0]:
                            distinct[key] = (p, pairs)
            for p, pairs in distinct.values():
                for e, x in pairs:
                    cnt[e][x] += p
            if not last:
                self.M = self._normalise(cnt) if cnt else self.M
                self.taught = distinct
            self.posterior = post
        return self

    def mapping(self, top=1):
        """Most probable letter for each symbol (read-only report)."""
        inv = defaultdict(list)
        for e, row in (self.M or {}).items():
            for x, p in row.items():
                inv[x].append((p * 1.0, e))
        out = {}
        for x, rows in inv.items():
            rows.sort(reverse=True)
            if rows and rows[0][0] > 3 * self.p_sym.get(x, 0.0):
                out[x] = rows[0][1]
        return out
