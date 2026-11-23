"""Confidence calibration: map public-only evidence of a best triple to P(correct).

The role model's posterior is a naive-Bayes product and is over-confident
(on development data a posterior >= 0.9 was correct about a third of the
time). The calibrator is a small L2 logistic regression over language-neutral
evidence numbers. It is fitted ONLY on the development split (labels from the
development annotations, outside this package) and stored as coefficients in
``calibration.json``; at inference time it reads nothing but the decoder's own
record. Without the file the decoder falls back to the raw posterior.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

PATH = Path(__file__).with_name("calibration.json")
NAMES = ("bias", "log_p", "agree", "margin", "s_marg", "v_marg", "o_marg", "s_null", "o_null", "log_len",
         "s_dist", "o_dist", "v_lex", "s_lex", "o_lex", "s_anchor", "o_anchor", "v_anchor",
         "v_concept", "s_concept", "o_concept", "log_items")


def _marg(rec, role, pos):
    for c in rec["roles"][role].get("candidates", []):
        if c["position"] == pos:
            return c["p"]
    return 0.0


def features(rec):
    """Feature vector (dict) for the best complete triple of a record, or None."""
    if not rec.get("roles") or not rec.get("alternatives"):
        return None
    best = rec["alternatives"][0]["positions"]
    s, v, o = best
    alts = rec["alternatives"]
    second = alts[1]["p"] if len(alts) > 1 else 0.0
    ev = rec.get("evidence") or {}

    def e(role, key):
        return float((ev.get(role) or {}).get(key, 0.0))

    agree_num, agree_den = (rec.get("restart_agreement") or "0/1").split("/")
    roles = rec["roles"]
    return {
        "bias": 1.0,
        "log_p": math.log(max(rec["alternatives"][0]["p"], 1e-6)),
        "agree": int(agree_num) / max(1, int(agree_den)),
        "margin": rec["alternatives"][0]["p"] - second,
        "s_marg": _marg(rec, "subject", s), "v_marg": _marg(rec, "predicate", v), "o_marg": _marg(rec, "object", o),
        "s_null": float(roles["subject"].get("p_not_expressed", 0.0)),
        "o_null": float(roles["object"].get("p_not_expressed", 0.0)),
        "log_len": math.log(len(rec["tokens"])),
        "s_dist": math.log(1 + abs(s - v)), "o_dist": math.log(1 + abs(o - v)),
        "v_lex": math.log1p(e("predicate", "lexical_mass")), "s_lex": math.log1p(e("subject", "lexical_mass")),
        "o_lex": math.log1p(e("object", "lexical_mass")),
        "s_anchor": e("subject", "anchor"), "o_anchor": e("object", "anchor"), "v_anchor": e("predicate", "anchor"),
        "v_concept": math.log1p(e("predicate", "concept_mass")), "s_concept": math.log1p(e("subject", "concept_mass")),
        "o_concept": math.log1p(e("object", "concept_mass")),
        "log_items": math.log1p(rec.get("corpus_translated", 0)),
    }


def vector(feat):
    return [float(feat[n]) for n in NAMES]


def fit(rows, l2=1.0, iters=400, lr=0.5):
    """rows: [(feature dict, label)]. Standardised features, batch gradient descent. Returns a model dict."""
    X = [vector(f) for f, _ in rows]
    y = [float(t) for _, t in rows]
    k = len(NAMES)
    mu = [sum(x[j] for x in X) / len(X) for j in range(k)]
    sd = [math.sqrt(sum((x[j] - mu[j]) ** 2 for x in X) / len(X)) or 1.0 for j in range(k)]
    mu[0], sd[0] = 0.0, 1.0
    Z = [[(x[j] - mu[j]) / sd[j] for j in range(k)] for x in X]
    w = [0.0] * k
    n = len(Z)
    for _ in range(iters):
        g = [0.0] * k
        for z, t in zip(Z, y, strict=True):
            p = 1 / (1 + math.exp(-max(-30, min(30, sum(a * b for a, b in zip(w, z, strict=True))))))
            for j in range(k):
                g[j] += (p - t) * z[j]
        for j in range(k):
            reg = l2 * w[j] if j else 0.0
            w[j] -= lr * (g[j] / n + reg / n)
    return {"names": list(NAMES), "mean": mu, "sd": sd, "weights": w}


def predict(model, feat):
    if model is None or feat is None:
        return None
    z = [(x - m) / s for x, m, s in zip(vector(feat), model["mean"], model["sd"], strict=True)]
    a = sum(w * x for w, x in zip(model["weights"], z, strict=True))
    return 1 / (1 + math.exp(-max(-30, min(30, a))))


def load(path=PATH):
    p = Path(path)
    if not p.exists():
        return None
    model = json.loads(p.read_text(encoding="utf-8"))
    if model.get("names") != list(NAMES):
        raise ValueError("calibration.json was fitted with a different feature list")
    return model
