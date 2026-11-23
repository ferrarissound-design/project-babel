"""Public-input-only decoding: corpus (+ optional structured translations) -> analysis.

The output keeps three kinds of statement apart:

* ``structure``  – derived from the symbol strings alone (word boundaries,
  affixes). No meaning is asserted.
* sentences with ``basis: "translation"`` – roles and word meanings supported
  by a structured translation of that very sentence.
* sentences with ``basis: "structure"`` – roles proposed for sentences without
  a translation, using only the positional/distributional experts that were
  learned on translated sentences. They are hypotheses about structure, never
  translations, and carry no concept labels.

Every role claim has a confidence (mean posterior of the triple over EM
restarts) and every non-claim has a reason. Confidence is an uncalibrated
model posterior; docs/DECODE_ENGINE.md reports how it relates to measured
precision on development data.
"""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict

from babel.decipher.morph import Morphology
from babel.decipher.segment import unigram_segment

from . import calibration
from .anchors import NameAnchors
from .model import DEFAULTS, LACUNA, ROLES, UNKNOWN, Corpus, RoleModel

VERSION = "2.0.0"
SCHEMA = "babel-decode-v2"
DECODE_DEFAULTS = dict(DEFAULTS, claim_threshold=None, hypothesis_threshold=None, min_translated=20,
                       use_anchors=True, restarts=(1, 2, 3), calibration="auto")
ROLE_NAMES = {"S": "subject", "V": "predicate", "O": "object"}


def _translation_item(index, tr):
    if not isinstance(tr, dict) or not tr.get("pred"):
        return None
    item = {"sent": index, "V": str(tr["pred"]), "S": None, "O": None}
    if tr.get("tense"):
        item["tense"] = str(tr["tense"])
    for a in tr.get("args", []) or []:
        role = {"subject": "S", "object": "O"}.get(a.get("role"))
        if role and a.get("head") and item[role] is None:
            item[role] = str(a["head"])
            item[role + "_pl"] = "PL" if a.get("pl") else "SG"
    return item


def _binomial_upper(k, n, p=1 / 3):
    if n == 0:
        return 1.0
    lf = math.lgamma(n + 1)
    return min(1.0, math.fsum(math.exp(lf - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                                       + i * math.log(p) + (n - i) * math.log(1 - p)) for i in range(k, n + 1)))


class Decoder:
    def __init__(self, public: dict, config: dict | None = None):
        self.pub = public
        self.cfg = dict(DECODE_DEFAULTS, **(config or {}))
        self.warnings = []
        self.calibrator = calibration.load() if self.cfg["calibration"] == "auto" else None
        # threshold: explicit config > the calibrator's development-selected threshold > 0.5 on the raw posterior
        default = self.calibrator["threshold"] if self.calibrator else 0.5
        for key in ("claim_threshold", "hypothesis_threshold"):
            if self.cfg[key] is None:
                self.cfg[key] = default

    # ------------------------------------------------------------ tokenizing
    def _tokenize(self):
        self.keys, raw = [], []
        for d in self.pub.get("documents", []):
            for j, s in enumerate(d.get("sentences", [])):
                self.keys.append((d["id"], j))
                raw.append(s)
        if self.pub.get("has_word_boundaries", True):
            self.sentences = [s.split() for s in raw]
            self.boundary_source = "given"
        else:
            self.sentences = unigram_segment([s.replace(" ", "") for s in raw])
            self.boundary_source = "inferred (unsupervised unigram segmentation)"
        self.index = {k: i for i, k in enumerate(self.keys)}

    def _items(self):
        self.items, self.item_keys, seen = [], [], set()
        for p in self.pub.get("parallel", []) or []:
            key = (p.get("doc"), p.get("sent"))
            if key not in self.index:
                self.warnings.append(f"translation for unknown sentence {key} ignored")
                continue
            if key in seen:
                self.warnings.append(f"duplicate translation for {key}: first kept")
                continue
            item = _translation_item(self.index[key], p.get("translation"))
            if item is None or not self.sentences[self.index[key]]:
                continue
            seen.add(key)
            self.items.append(item)
            self.item_keys.append(key)

    def fit(self):
        t0 = time.time()
        self._tokenize()
        counts = Counter(w for s in self.sentences for w in s if w != LACUNA)
        self.morph = Morphology(dict(counts))
        self.corpus = Corpus(self.sentences, self.morph, self.cfg)
        self._items()
        self.anchors = None
        if self.cfg["use_anchors"] and self.items:
            occ = [(k, r, it[r], it["sent"]) for k, it in enumerate(self.items) for r in ROLES
                   if it.get(r) not in (None, UNKNOWN)]
            self.anchors = NameAnchors(self.sentences, occ).fit()
            for it in self.items:
                it["anchor"] = {}
            for (k, r, j), p in self.anchors.posterior.items():
                self.items[k]["anchor"].setdefault(r, {})[j] = p
        self.trained = len(self.items) >= self.cfg["min_translated"]
        self.models = []
        if self.trained:
            for seed in self.cfg["restarts"]:
                self.models.append(RoleModel(self.corpus, self.items, dict(self.cfg, order_seed=seed)).fit())
        elif self.items:
            self.warnings.append(f"only {len(self.items)} translated sentences (< {self.cfg['min_translated']}): "
                                 "no role model is trained and no roles are claimed")
        self.fit_seconds = round(time.time() - t0, 2)
        return self

    # ----------------------------------------------------------- combining
    @staticmethod
    def _combine(posts, n):
        posts = [p for p in posts if p is not None]
        if not posts:
            return None
        k = len(posts)
        triples = defaultdict(float)
        for p in posts:
            for t, v in p["triples"].items():
                triples[t] += v / k
        marg = {}
        for r in ROLES:
            if r != "V" and not posts[0]["has_" + r]:
                marg[r] = None
                continue
            marg[r] = [sum(p[r][i] for p in posts) / k for i in range(len(posts[0][r]))]
        best = max(triples.items(), key=lambda kv: (kv[1], [-x for x in kv[0]]), default=(None, 0.0))
        agree = sum(1 for p in posts if p["best"] == best[0]) if best[0] else 0
        alts = sorted(triples.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        ev = [p["evidence"][best[0]] for p in posts if best[0] in p.get("evidence", {})]
        evidence = None
        if ev:
            evidence = {r: {f: round(sum(e[r][f] for e in ev) / len(ev), 3) for f in ev[0][r]} for r in ROLES}
        return {"marg": marg, "best": best[0], "p": best[1], "agree": agree, "runs": k, "alts": alts, "n": n,
                "evidence": evidence}

    def _record(self, key, sent, comb, concepts, basis, threshold, complete):
        n = len(sent)
        rec = {"doc": key[0], "sent": key[1], "basis": basis, "tokens": sent}
        if concepts is not None:
            rec["concepts"] = {ROLE_NAMES[r]: concepts[r] for r in ROLES}
        rec["complete_source"] = complete
        rec["corpus_translated"] = len(self.items)
        if comb is None:
            rec.update(triple=None, confidence=0.0, status="undetermined",
                       reason_code="no_model" if not self.trained else "no_posterior",
                       reason="no role model (too few translated sentences)" if not self.trained else "no posterior")
            return rec
        roles = {}
        for r in ROLES:
            m = comb["marg"][r]
            if m is None:
                roles[ROLE_NAMES[r]] = {"absent_in_translation": True}
                continue
            order = sorted(range(n), key=lambda i: (-m[i], i))[:5]
            roles[ROLE_NAMES[r]] = {"candidates": [{"position": i, "token": sent[i], "p": round(m[i], 4)}
                                                   for i in order if m[i] > 0]}
            if r != "V":
                roles[ROLE_NAMES[r]]["p_not_expressed"] = round(m[n], 4)
        rec["roles"] = roles
        rec["confidence"] = round(comb["p"], 4)
        rec["restart_agreement"] = f"{comb['agree']}/{comb['runs']}"
        rec["alternatives"] = [{"positions": list(t), "p": round(v, 4)} for t, v in comb["alts"]]
        if comb["evidence"]:
            rec["evidence"] = {ROLE_NAMES[r]: v for r, v in comb["evidence"].items()}
        best = comb["best"]
        score = comb["p"]
        if self.calibrator is not None and complete and best is not None:
            score = calibration.predict(self.calibrator, calibration.features(rec))
            rec["calibrated_confidence"] = round(score, 4)
        if not complete:
            rec.update(triple=None, status="undetermined", reason_code="incomplete_translation",
                       reason="translation lacks subject or object, so no complete triple is asked for")
        elif best is None:
            rec.update(triple=None, status="undetermined", reason_code="no_triple",
                       reason="no complete triple has posterior mass")
        elif score < threshold:
            kind = "calibrated confidence" if "calibrated_confidence" in rec else "best triple posterior"
            rec.update(triple=None, best_guess=list(best), status="undetermined", reason_code="below_threshold",
                       reason=f"{kind} {score:.2f} below threshold {threshold}")
        else:
            rec.update(triple={"subject": best[0], "predicate": best[1], "object": best[2]},
                       triple_tokens={"subject": sent[best[0]], "predicate": sent[best[1]], "object": sent[best[2]]},
                       status="claimed", reason_code="claimed", reason="confidence at or above threshold")
        return rec

    def grounded(self):
        """One record per translated sentence (basis: translation)."""
        out = []
        for k, (key, item) in enumerate(zip(self.item_keys, self.items, strict=True)):
            sent = self.sentences[item["sent"]]
            comb = self._combine([m.posteriors[k] for m in self.models], len(sent)) if self.models else None
            complete = item["S"] is not None and item["O"] is not None
            out.append(self._record(key, sent, comb, item, "translation", self.cfg["claim_threshold"], complete))
        return out

    def hypotheses(self):
        """Structural role hypotheses for sentences without a translation (no meanings)."""
        if not self.models:
            return []
        done = {it["sent"] for it in self.items}
        out = []
        for i, sent in enumerate(self.sentences):
            if i in done or len(sent) < 3:
                continue
            item = {"sent": i, "S": UNKNOWN, "V": UNKNOWN, "O": UNKNOWN}
            comb = self._combine([m.posterior(item) for m in self.models], len(sent))
            out.append(self._record(self.keys[i], sent, comb, None, "structure", self.cfg["hypothesis_threshold"], True))
        return out

    # ------------------------------------------------------------ reporting
    def structure_report(self):
        aff = sorted(((k, v[0], v[1]) for k, v in self.morph.affixes.items()), key=lambda x: (-x[1], x[0]))[:20]
        rep = {"boundaries": {"source": self.boundary_source,
                              "tokens": sum(len(s) for s in self.sentences),
                              "types": len(self.corpus.counts)},
               "morphology": {"affix_position": self.morph.position,
                              "affixes": [{"affix": a, "stem_types": t, "tokens": c} for a, t, c in aff]}}
        if self.models:
            m = self.models[0]
            rep["role_profiles"] = {ROLE_NAMES[r]: m.role_profile(r) for r in ROLES}
            rep["argument_placement"] = {ROLE_NAMES[r]: m.placement_profile(r) for r in ("S", "O")}
        return rep

    def order_report(self, records):
        votes = Counter()
        for r in records:
            if r["status"] == "claimed":
                t = r["triple"]
                s, v, o = t["subject"], t["predicate"], t["object"]
                votes["initial" if v < min(s, o) else "final" if v > max(s, o) else "medial"] += 1
        so = Counter()
        for r in records:
            if r["status"] == "claimed":
                so["SO" if r["triple"]["subject"] < r["triple"]["object"] else "OS"] += 1
        n = sum(votes[k] for k in ("initial", "medial", "final"))
        best = max(("initial", "medial", "final"), key=lambda k: (votes[k], k))
        p = _binomial_upper(votes[best], n)
        claim = best if n >= 5 and p <= 0.05 / 3 else None
        return {"verb_position_votes": {k: votes[k] for k in ("initial", "medial", "final")},
                "subject_object_votes": dict(so), "n": n, "p_value_vs_uniform": p,
                "verb_position": claim, "basis": "claimed triples of translated sentences",
                "status": "claimed" if claim else "undetermined"}

    def lexicon(self):
        if not self.models:
            return []
        rows = self.models[0].lexicon()
        for r in rows:
            r["basis"] = "translation"
        return rows


def decode(public: dict, config: dict | None = None, hypotheses: bool = True) -> dict:
    d = Decoder(public, config).fit()
    grounded = d.grounded()
    hyp = d.hypotheses() if hypotheses else []
    reasons = Counter(r["reason_code"] for r in grounded if r["status"] != "claimed")
    out = {
        "schema": SCHEMA, "version": VERSION, "language_id": d.pub.get("language_id"),
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.cfg.items()},
        "input": {"documents": len(d.pub.get("documents", [])), "sentences": len(d.sentences),
                  "translated_sentences": len(d.items), "role_model_trained": d.trained},
        "structure": d.structure_report(),
        "lexicon": d.lexicon(),
        "anchors": None,
        "word_order": d.order_report(grounded),
        "sentences": grounded,
        "structural_hypotheses": hyp,
        "summary": {
            "translated": len(grounded),
            "claimed": sum(r["status"] == "claimed" for r in grounded),
            "undetermined": sum(r["status"] != "claimed" for r in grounded),
            "undetermined_reasons": dict(reasons),
            "structural_hypotheses_claimed": sum(r["status"] == "claimed" for r in hyp),
            "structural_hypotheses_total": len(hyp),
        },
        "warnings": d.warnings,
        "fit_seconds": d.fit_seconds,
    }
    if d.anchors is not None and d.anchors.M:
        pairs = sorted(((p, d.items[k][r], d.sentences[d.items[k]["sent"]][j]) for (k, r, j), p in
                        d.anchors.posterior.items() if p >= 0.5), reverse=True)
        out["anchors"] = {"symbols_mapped": len(d.anchors.mapping()), "confident_pairs": len(pairs),
                          "chance_threshold": round(d.anchors.null_threshold, 3)
                          if d.anchors.null_threshold not in (None, float("inf")) else None,
                          "examples": [{"concept": c, "token": w, "p": round(p, 3)} for p, c, w in pairs[:15]],
                          "basis": "letter/symbol correspondence learned from translations (names, loanwords)"}
    return out


__all__ = ["Decoder", "decode", "VERSION", "SCHEMA", "UNKNOWN"]
