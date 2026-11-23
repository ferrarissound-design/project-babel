"""Joint latent subject–predicate–object model trained by leave-one-out incremental EM.

Why this replaces the A–E family
--------------------------------
The A–E methods first ask a lexical aligner (IBM Model 1 over three English
concepts) which target stem translates each concept, and only then build
roles from the surviving links. In the PUD development split ~74% of English
predicates and ~88% of objects occur once per language, so a co-occurrence
aligner cannot single them out: candidates are lost before any structural
reasoning starts. Thresholds and soft POS priors cannot repair that.

Here every sentence has a latent triple (s, v, o) of positions (s and o may
be NULL). Its score is a product of experts that are all learned jointly from
the posterior of every other sentence:

* role emission E_r(i): naive-Bayes likelihood ratios of the token's
  distributional features (frequent neighbours, suffix/prefix strings,
  analysed affixes, length, frequency band, relative position) for role r
  against the corpus background. This generalises to words seen once.
* lexical L(c, u): translation evidence of target stem u for English concept
  c, smoothed towards the background so that unseen pairs are neutral (1).
* placement D_r(bucket(i - v)): signed distance of the argument from the
  predicate, including an explicit NULL outcome (pro-drop, oblique object).
* conditional affix agreement: suffix given English tense (predicate) and
  plural flag (arguments), relative to the role's own suffix distribution.

All counts are leave-one-out: while a sentence is re-estimated its own
previous contribution is removed, so a hapax concept cannot confirm itself.
The posterior over triples is exact (O(n^2) per sentence) and the probability
of the best complete triple is reported as the confidence.
"""
from __future__ import annotations

import math
from collections import defaultdict

ROLES = ("S", "V", "O")
NULL = None
LACUNA = "[…]"
UNKNOWN = "<?>"   # role allowed but no translation: lexical expert is neutral
TEMPLATES = ("w", "p", "n", "p2", "n2", "s1", "s2", "s3", "f1", "f2", "ms", "mp", "len", "fq", "rp")
DEFAULTS = {
    "passes": 10,
    "warmup_passes": 2,        # lexical-only passes before structure/class experts are switched on
    "feature_weight": 0.5,     # naive-Bayes tempering of correlated templates
    "feature_smoothing": 4.0,  # Dirichlet mass towards the background for role features
    "lexical_smoothing": 1.0,
    "reverse_smoothing": 1.0,  # P(concept | stem): a stem that absorbs many different concepts is not a translation
    "reverse_lexical": True,
    "placement_smoothing": 1.0,
    "agreement_weight": 0.5,
    "cross_weight": 0.5,       # learned cost of the tokens an argument link crosses to reach the predicate
    "cross_templates": ("w", "s2", "ms"),
    "frequent_word": 3,
    "teacher_power": 0.0,      # structure experts learn from mass weighted by (max marginal)^power (0 = off; tried, worse)
    "anchor_strength": 30.0,   # unary multiplier 1 + strength * P(name/loan anchor)
    "order_seed": 0,           # processing order of incremental EM (0 = corpus order)
    "null_smoothing": 2.0,     # concept-specific rate of an argument not being expressed (pro-drop, oblique)
    "frequent_context": 8,
    "wide_context": True,      # second neighbours (case particles, determiners two words away)
}


def bucket(delta):
    a = abs(delta)
    b = 1 if a == 1 else 2 if a == 2 else 3 if a == 3 else 4 if a <= 5 else 6 if a <= 8 else 9
    return ("+" if delta > 0 else "-") + str(b)


BUCKETS = [s + str(b) for s in "+-" for b in (1, 2, 3, 4, 6, 9)] + ["NULL"]


class Corpus:
    """Token features computed once from the whole public corpus (no translations used)."""

    def __init__(self, sentences, morph, cfg):
        self.sentences = sentences
        self.counts = defaultdict(int)
        for sent in sentences:
            for w in sent:
                if w != LACUNA:
                    self.counts[w] += 1
        self.morph = morph
        self.analysis = {w: morph.analyze(w) for w in self.counts}
        self.stem = {w: a[1] for w, a in self.analysis.items()}
        self.cfg = cfg
        self.features = [[self._features(sent, i) for i in range(len(sent))] for sent in sentences]
        self.bg = defaultdict(float)
        self.bg_tot = defaultdict(float)
        self.bg_types = defaultdict(set)
        self.stem_bg = defaultdict(float)
        n_tok = 0
        for sent, feats in zip(sentences, self.features):
            for w, f in zip(sent, feats):
                if f is None:
                    continue
                n_tok += 1
                self.stem_bg[self.stem[w]] += 1
                for tpl, val in f.items():
                    self.bg[(tpl, val)] += 1
                    self.bg_tot[tpl] += 1
                    self.bg_types[tpl].add(val)
        self.n_tok = max(1, n_tok)
        self.n_stems = max(1, len(self.stem_bg))

    def _features(self, sent, i):
        w = sent[i]
        if w == LACUNA:
            return None
        cf, ff = self.cfg["frequent_context"], self.cfg["frequent_word"]
        c = self.counts

        def ctx(j, edge):
            if j < 0 or j >= len(sent):
                return edge
            x = sent[j]
            return x if x != LACUNA and c[x] >= cf else "<c>"

        pre, _, suf = self.analysis[w]
        f = {
            "w": w if c[w] >= ff else "<r>",
            "p": ctx(i - 1, "<s>"),
            "n": ctx(i + 1, "</s>"),
            **({"p2": ctx(i - 2, "<s>"), "n2": ctx(i + 2, "</s>")} if self.cfg.get("wide_context") else {}),
            "ms": "|".join(suf) or "-",
            "mp": "|".join(pre) or "-",
            "len": str(min(len(w), 12)),
            "fq": str(min(int(math.log2(c[w])), 8)),
            "rp": str(min(9, int(10 * i / len(sent)))),
        }
        for k in (1, 2, 3):
            f["s%d" % k] = w[-k:] if len(w) >= k + 2 else "<short>"
        for k in (1, 2):
            f["f%d" % k] = w[:k] if len(w) >= k + 2 else "<short>"
        return f

    def p_bg(self, tpl, val):
        return (self.bg.get((tpl, val), 0.0) + 0.5) / (self.bg_tot[tpl] + 0.5 * (len(self.bg_types[tpl]) + 1))

    def p_stem(self, u):
        return (self.stem_bg.get(u, 0.0) + 0.5) / (self.n_tok + 0.5 * self.n_stems)


class RoleModel:
    """Leave-one-out incremental EM over latent (s, v, o) triples."""

    def __init__(self, corpus, items, cfg=None):
        """items: [{'sent': index, 'S': concept|None, 'V': concept, 'O': concept|None, 'tense', 'S_pl', 'O_pl'}]."""
        self.c = corpus
        self.items = items
        self.cfg = dict(DEFAULTS, **(cfg or {}))
        self.G = defaultdict(float)
        self.contrib = [None] * len(items)
        self.structure_on = False
        self.posteriors = [None] * len(items)
        self.concept_freq = defaultdict(float)
        for it in items:
            for r in ROLES:
                if it.get(r) not in (None, UNKNOWN):
                    self.concept_freq[it[r]] += 1
        self.concept_total = max(1.0, sum(self.concept_freq.values()))

    # --------------------------------------------------------------- experts
    def emission(self, role, feats):
        if not self.structure_on or feats is None:
            return 1.0
        G, lam = self.G, self.cfg["feature_smoothing"]
        log = 0.0
        for tpl, val in feats.items():
            pb = self.c.p_bg(tpl, val)
            p = (G[("E", role, tpl, val)] + lam * pb) / (G[("Et", role, tpl)] + lam)
            log += math.log(p / pb)
        return math.exp(self.cfg["feature_weight"] * log)

    def agreement(self, role, feats, flag):
        """Suffix given the English tense/plural flag, relative to the role's own suffix distribution."""
        if not self.structure_on or feats is None or flag is None:
            return 1.0
        G, lam = self.G, self.cfg["feature_smoothing"]
        log = 0.0
        for tpl in ("s1", "s2", "ms"):
            val = feats[tpl]
            base = (G[("E", role, tpl, val)] + lam * self.c.p_bg(tpl, val)) / (G[("Et", role, tpl)] + lam)
            p = (G[("A", role, flag, tpl, val)] + lam * base) / (G[("At", role, flag, tpl)] + lam)
            log += math.log(p / base)
        return math.exp(self.cfg["agreement_weight"] * log)

    def lexical(self, concept, stem):
        if concept is None or concept == UNKNOWN:
            return 1.0
        lam = self.cfg["lexical_smoothing"]
        pu = self.c.p_stem(stem)
        forward = ((self.G[("L", concept, stem)] + lam * pu) / (self.G[("Lt", concept)] + lam)) / pu
        if not self.cfg["reverse_lexical"]:
            return forward
        lam_r = self.cfg["reverse_smoothing"]
        pc = (self.concept_freq[concept] + 0.5) / (self.concept_total + 0.5 * len(self.concept_freq))
        reverse = ((self.G[("L", concept, stem)] + lam_r * pc) / (self.G[("Ut", stem)] + lam_r)) / pc
        return forward * reverse

    def crossing(self, role, feats):
        """Likelihood ratio that a token lies between an argument and its predicate."""
        if not self.structure_on or feats is None or not self.cfg["cross_weight"]:
            return 1.0
        G, lam = self.G, self.cfg["feature_smoothing"]
        log = 0.0
        for tpl in self.cfg["cross_templates"]:
            val = feats[tpl]
            pb = self.c.p_bg(tpl, val)
            p = (G[("X", role, tpl, val)] + lam * pb) / (G[("Xt", role, tpl)] + lam)
            log += math.log(p / pb)
        return math.exp(self.cfg["cross_weight"] * log)

    def null_ratio(self, role, item):
        """P(argument unexpressed | its concept, the predicate concept) relative to the role's overall rate."""
        if not self.structure_on:
            return 1.0
        G, lam = self.G, self.cfg["null_smoothing"]
        base = (G[("D", role, "NULL")] + 1.0) / (G[("Dt", role)] + 2.0)
        ratio = 1.0
        for key in ((role, "c", item.get(role)), (role, "v", item.get("V"))):
            if key[2] in (None, UNKNOWN):
                continue
            p = (G[("N",) + key] + lam * base) / (G[("Nt",) + key] + lam)
            ratio *= p / base
        return ratio

    def placement(self, role, b):
        if not self.structure_on:
            return 1.0 / len(BUCKETS)
        lam = self.cfg["placement_smoothing"]
        return (self.G[("D", role, b)] + lam) / (self.G[("Dt", role)] + lam * len(BUCKETS))

    # ------------------------------------------------------------- inference
    def unary(self, item):
        sent = self.c.sentences[item["sent"]]
        feats = self.c.features[item["sent"]]
        out = {}
        for role in ROLES:
            concept = item.get(role)
            if role != "V" and concept is None:
                out[role] = None
                continue
            flag = item.get("tense") if role == "V" else item.get(role + "_pl")
            anchor = (item.get("anchor") or {}).get(role, {})
            rho = self.cfg["anchor_strength"]
            vals = []
            for i, w in enumerate(sent):
                if feats[i] is None:
                    vals.append(0.0)
                    continue
                vals.append(self.emission(role, feats[i]) * self.agreement(role, feats[i], flag)
                            * self.lexical(concept, self.c.stem[w]) * (1.0 + rho * anchor.get(i, 0.0)))
            out[role] = vals
        return out

    def evidence(self, item, triple):
        """Leave-one-out evidence behind each position of a triple (for reporting and calibration)."""
        sent = self.c.sentences[item["sent"]]
        feats = self.c.features[item["sent"]]
        out = {}
        for role, i in zip(("S", "V", "O"), triple, strict=True):
            concept = item.get(role)
            stem = self.c.stem[sent[i]]
            known = concept not in (None, UNKNOWN)
            out[role] = {"lexical_mass": round(self.G[("L", concept, stem)], 3) if known else 0.0,
                         "concept_mass": round(self.G[("Lt", concept)], 3) if known else 0.0,
                         "lexical_ratio": round(self.lexical(concept, stem), 3),
                         "emission_ratio": round(self.emission(role, feats[i]), 3),
                         "anchor": round((item.get("anchor") or {}).get(role, {}).get(i, 0.0), 3)}
        return out

    def posterior(self, item):
        """Exact posterior marginals and the best complete triple for one item."""
        n = len(self.c.sentences[item["sent"]])
        u = self.unary(item)
        rows = {}
        for role in ("S", "O"):
            if u[role] is None:
                rows[role] = None
                continue
            pn = self.placement(role, "NULL") * self.null_ratio(role, item)
            feats = self.c.features[item["sent"]]
            xr = [self.crossing(role, f) for f in feats]
            pl = {b: self.placement(role, b) for b in BUCKETS}
            per_v = []
            for v in range(n):
                width = defaultdict(int)
                for i in range(n):
                    if i != v:
                        width[bucket(i - v)] += 1
                f = [0.0] * n
                cum = 1.0
                for i in range(v - 1, -1, -1):
                    b = bucket(i - v)
                    f[i] = u[role][i] * pl[b] / width[b] * cum
                    cum *= xr[i]
                cum = 1.0
                for i in range(v + 1, n):
                    b = bucket(i - v)
                    f[i] = u[role][i] * pl[b] / width[b] * cum
                    cum *= xr[i]
                per_v.append((f, pn))
            rows[role] = per_v
        zv, total = [0.0] * n, 0.0
        parts = []
        for v in range(n):
            ev = u["V"][v]
            fs, ns = rows["S"][v] if rows["S"] else ([0.0] * n, 1.0)
            fo, no = rows["O"][v] if rows["O"] else ([0.0] * n, 1.0)
            a_s, a_o = sum(fs) + ns, sum(fo) + no
            coll = sum(x * y for x, y in zip(fs, fo))
            z = ev * (a_s * a_o - coll)
            zv[v] = z
            total += z
            parts.append((ev, fs, ns, fo, no, a_s, a_o))
        if total <= 0:
            return None
        pv = [z / total for z in zv]
        ps, po = [0.0] * (n + 1), [0.0] * (n + 1)  # index n = NULL
        joint_s, joint_o = defaultdict(float), defaultdict(float)
        best, best_p = None, 0.0
        triples = {}
        for v, (ev, fs, ns, fo, no, a_s, a_o) in enumerate(parts):
            if zv[v] <= 0:
                continue
            for i in range(n):
                if fs[i]:
                    m = ev * fs[i] * (a_o - fo[i]) / total
                    ps[i] += m
                    joint_s[(v, i)] += m
                if fo[i]:
                    m = ev * fo[i] * (a_s - fs[i]) / total
                    po[i] += m
                    joint_o[(v, i)] += m
            m = ev * ns * a_o / total
            ps[n] += m
            joint_s[(v, None)] += m
            m = ev * no * a_s / total
            po[n] += m
            joint_o[(v, None)] += m
            # complete triples at this v: top-3 of each side resolves the s != o constraint
            ts = sorted(range(n), key=lambda i: -fs[i])[:3]
            to = sorted(range(n), key=lambda i: -fo[i])[:3]
            for i in ts:
                for j in to:
                    if i != j and fs[i] and fo[j]:
                        p = ev * fs[i] * fo[j] / total
                        if p > 1e-4:
                            triples[(i, v, j)] = p
                        if p > best_p:
                            best, best_p = (i, v, j), p
        top = dict(sorted(triples.items(), key=lambda kv: (-kv[1], kv[0]))[:10])
        return {"V": pv, "S": ps, "O": po, "joint_S": joint_s, "joint_O": joint_o, "triples": top,
                "best": best, "best_p": best_p, "has_S": rows["S"] is not None, "has_O": rows["O"] is not None}

    def _contribution(self, item, post):
        sent = self.c.sentences[item["sent"]]
        feats = self.c.features[item["sent"]]
        n = len(sent)
        out = defaultdict(float)
        for role in ROLES:
            if role != "V" and not post["has_" + role]:
                continue
            marg = post[role]
            concept = item.get(role)
            gamma = self.cfg["teacher_power"]
            weight = max(marg) ** gamma if gamma else 1.0
            flag = item.get("tense") if role == "V" else item.get(role + "_pl")
            for i in range(n):
                m = marg[i]
                if m < 1e-6 or feats[i] is None:
                    continue
                ms = m * weight
                for tpl, val in feats[i].items():
                    out[("E", role, tpl, val)] += ms
                    out[("Et", role, tpl)] += ms
                if flag is not None:
                    for tpl in ("s1", "s2", "ms"):
                        out[("A", role, flag, tpl, feats[i][tpl])] += ms
                        out[("At", role, flag, tpl)] += ms
                if concept != UNKNOWN:
                    out[("L", concept, self.c.stem[sent[i]])] += m
                    out[("Lt", concept)] += m
                    out[("Ut", self.c.stem[sent[i]])] += m
            if role != "V":
                pnull = marg[n]
                for key in ((role, "c", item.get(role)), (role, "v", item.get("V"))):
                    if key[2] not in (None, UNKNOWN):
                        out[("N",) + key] += pnull
                        out[("Nt",) + key] += 1.0
                for (v, i), m in post["joint_" + role].items():
                    if m < 1e-6:
                        continue
                    out[("D", role, "NULL" if i is None else bucket(i - v))] += m * weight
                    out[("Dt", role)] += m * weight
                    if i is not None and self.cfg["cross_weight"]:
                        for t in range(min(i, v) + 1, max(i, v)):
                            if feats[t] is None:
                                continue
                            for tpl in self.cfg["cross_templates"]:
                                out[("X", role, tpl, feats[t][tpl])] += m * weight
                                out[("Xt", role, tpl)] += m * weight
        return out

    def _apply(self, contrib, sign):
        G = self.G
        for k, val in contrib.items():
            G[k] += sign * val

    def fit(self):
        cfg = self.cfg
        order = list(range(len(self.items)))
        if cfg["order_seed"]:
            import random
            random.Random(cfg["order_seed"]).shuffle(order)
        for p in range(cfg["passes"]):
            self.structure_on = p >= cfg["warmup_passes"]
            for k in order:
                item = self.items[k]
                if self.contrib[k] is not None:
                    self._apply(self.contrib[k], -1)  # leave this sentence out while re-estimating it
                post = self.posterior(item)
                self.posteriors[k] = post
                if post is None:
                    self.contrib[k] = None
                    continue
                self.contrib[k] = self._contribution(item, post)
                self._apply(self.contrib[k], +1)
        # final scoring pass: every posterior computed with its own contribution removed
        for k, item in enumerate(self.items):
            if self.contrib[k] is not None:
                self._apply(self.contrib[k], -1)
            post = self.posterior(item)
            if post is not None:
                post["evidence"] = {t: self.evidence(item, t) for t in list(post["triples"])[:3]}
            self.posteriors[k] = post
            if self.contrib[k] is not None:
                self._apply(self.contrib[k], +1)
        return self

    # ------------------------------------------------------------- reporting
    def role_profile(self, role, top=8):
        """Most indicative feature values for a role (likelihood ratio vs background, with support)."""
        rows = []
        for key, m in self.G.items():
            if key[0] != "E" or key[1] != role or m < 3:
                continue
            _, _, tpl, val = key
            if val in ("<c>", "<r>", "<short>", "-"):
                continue
            pb = self.c.p_bg(tpl, val)
            p = (m + self.cfg["feature_smoothing"] * pb) / (self.G[("Et", role, tpl)] + self.cfg["feature_smoothing"])
            rows.append((p / pb, tpl, val, m))
        rows.sort(key=lambda r: (-r[0], r[1], r[2]))
        return [{"template": t, "value": v, "ratio": round(x, 2), "support": round(m, 1)} for x, t, v, m in rows[:top]]

    def placement_profile(self, role):
        return {b: round(self.placement(role, b), 3) for b in BUCKETS}

    def lexicon(self, min_mass=1.5, min_share=0.5):
        by_concept = defaultdict(dict)
        for key, m in self.G.items():
            if key[0] == "L" and m > 0:
                by_concept[key[1]][key[2]] = m
        out = []
        for concept, stems in by_concept.items():
            tot = self.G[("Lt", concept)]
            stem, m = max(stems.items(), key=lambda kv: (kv[1], kv[0]))
            if m >= min_mass and tot > 0 and m / tot >= min_share:
                out.append({"concept": concept, "stem": stem, "mass": round(m, 2), "share": round(m / tot, 3),
                            "occurrences": round(tot, 2)})
        out.sort(key=lambda r: (-r["mass"], r["concept"]))
        return out
