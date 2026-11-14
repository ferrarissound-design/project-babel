"""bench.py の結果（runs）から指標を計算する。結果の JSON だけを読む（再計算できる）。"""
from __future__ import annotations

from collections import defaultdict

N_MEMBERS = 3  # independent segmentation bases; reclassifications are correlated
FALSE_KINDS = ("unplanted", "same_line", "unmapped")


def _wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def found(run, method):
    """その方式が見つけた組の一覧（kind つき）。vote>=k は票から作る。"""
    if method.startswith("vote>="):
        k = int(method.split(">=")[1])
        return [v for v in run["votes"] if v["votes"] >= k]
    return run["methods"].get(method, [])


def method_names(runs):
    names = list(runs[0]["methods"])
    return names + [f"vote>={k}" for k in range(2, N_MEMBERS + 1)]


def score(runs, method):
    """陰性対照（仕込みなし）での誤発見と、仕込みありでの発見・誤発見。"""
    neg = [r for r in runs if r["n_pairs"] == 0]
    pos = [r for r in runs if r["n_pairs"] > 0]
    neg_any = sum(1 for r in neg if found(r, method))
    neg_n = sum(len(found(r, method)) for r in neg)
    planted = sum(len(r["planted"]) for r in pos)
    hit = sum(sum(1 for f in found(r, method) if f["kind"] == "planted") for r in pos)
    false_pos = sum(sum(1 for f in found(r, method) if f["kind"] in FALSE_KINDS) for r in pos)
    false_all = false_pos + neg_n
    # An unplanted image can contain coincidental structure. Report the oracle
    # signal separately; the unconditional rate is not a pure false-alarm rate.
    oracle_quiet = [r for r in neg if not r["methods"].get("oracle-lcs")]
    quiet_any = sum(bool(found(r, method)) for r in oracle_quiet)
    return {"negative_images": len(neg), "negative_with_any": neg_any, "fwer": round(neg_any / len(neg), 4) if neg else None,
            "fwer_ci95": _wilson(neg_any, len(neg)), "false_per_negative_image": round(neg_n / len(neg), 4) if neg else None,
            "oracle_quiet_negative_images": len(oracle_quiet), "oracle_quiet_with_any": quiet_any,
            "oracle_quiet_rate": round(quiet_any / len(oracle_quiet), 4) if oracle_quiet else None,
            "oracle_quiet_ci95": _wilson(quiet_any, len(oracle_quiet)),
            "positive_images": len(pos), "planted": planted, "planted_found": hit, "recall": round(hit / planted, 4) if planted else None,
            "false_per_positive_image": round(false_pos / len(pos), 4) if pos else None,
            "precision": round(hit / (hit + false_all), 4) if hit + false_all else None,
            "false_total": false_all, "false_kinds": {k: sum(sum(1 for f in found(r, method) if f["kind"] == k) for r in runs)
                                                      for k in FALSE_KINDS}}


def table(runs, methods=None, by=None):
    methods = methods or method_names(runs)
    if by is None:
        return {m: score(runs, m) for m in methods}
    groups = defaultdict(list)
    for r in runs:
        groups[str(r[by])].append(r)
    return {g: {m: score(rs, m) for m in methods} for g, rs in sorted(groups.items())}


def _auc(pos, neg):
    """pos の値が neg の値より大きい確率（同点は 0.5）。"""
    if not pos or not neg:
        return None
    s = 0.0
    for p in pos:
        for q in neg:
            s += 1.0 if p > q else 0.5 if p == q else 0.0
    return round(s / (len(pos) * len(neg)), 4)


def vote_discrimination(runs):
    """どれか1つの方式で有意になった組（基準の行に戻せたもの）について、票の数が真偽を分けるか。"""
    by_votes = defaultdict(lambda: [0, 0])
    pos, neg = [], []
    for r in runs:
        for v in r["votes"]:
            true = v["kind"] == "planted"
            by_votes[v["votes"]][0 if true else 1] += 1
            (pos if true else neg).append(v["votes"])
    return {"candidates": len(pos) + len(neg), "true": len(pos), "false": len(neg), "auc": _auc(pos, neg),
            "by_votes": {str(k): {"true": t, "false": f, "p_true": round(t / (t + f), 4), "ci95": _wilson(t, t + f)}
                         for k, (t, f) in sorted(by_votes.items())}}


def existing_stability_discrimination(runs):
    """既存の babel analyze（improved-v3 の転写）の段階と安定性が、linepair 仮説の真偽を分けるか。"""
    tiers = defaultdict(lambda: [0, 0])
    samp = {True: [], False: []}
    tran = {True: [], False: []}
    for r in runs:
        for d in r.get("ngram_v3_detail", []):
            true = d["kind"] == "planted"
            tiers[d.get("tier") or "?"][0 if true else 1] += 1
            st = d.get("stability") or {}
            if st.get("sampling") is not None:
                samp[true].append(st["sampling"])
            if st.get("transcription") is not None:
                tran[true].append(st["transcription"])
    return {"by_tier": {k: {"true": t, "false": f} for k, (t, f) in sorted(tiers.items())},
            "auc_sampling_stability": _auc(samp[True], samp[False]), "auc_transcription_stability": _auc(tran[True], tran[False]),
            "true": len(samp[True]), "false": len(samp[False])}


def recognizer_dependence(runs, family, a="v3", b="baseline"):
    """同じ画像で、2つの方式の転写から見つかる組の集合がどれだけ食い違うか（IMG_4494 で起きた現象の頻度）。"""
    both = only_a = only_b = images_diff = images_any = 0
    true_both = true_one = n_one = 0
    for r in runs:
        A = {tuple(f["gt"]) for f in r["methods"][f"{family}:{a}"] if f["gt"]}
        B = {tuple(f["gt"]) for f in r["methods"][f"{family}:{b}"] if f["gt"]}
        if A or B:
            images_any += 1
            if A != B:
                images_diff += 1
        P = {tuple(p) for p in r["planted"]}
        both += len(A & B)
        only_a += len(A - B)
        only_b += len(B - A)
        true_both += len(A & B & P)
        true_one += len((A ^ B) & P)
        n_one += len(A ^ B)
    return {"pairs_in_both": both, "pairs_only_in_a": only_a, "pairs_only_in_b": only_b,
            "jaccard": round(both / (both + only_a + only_b), 4) if both + only_a + only_b else None,
            "images_with_any": images_any, "images_with_different_sets": images_diff,
            "p_true_if_both": round(true_both / both, 4) if both else None, "p_true_if_only_one": round(true_one / n_one, 4) if n_one else None}


def vision_summary(runs):
    out = {}
    for name in runs[0]["vision"]:
        vs = [r["vision"][name] for r in runs]

        def mean(k):
            xs = [v[k] for v in vs if v.get(k) is not None]
            return round(sum(xs) / len(xs), 4) if xs else None
        out[name] = {"detection_f1": mean("f1"), "ari": mean("ari"), "line_f1": mean("line_f1"),
                     "class_ratio": round(sum(v["classes"] for v in vs) / sum(r["types"] for r in runs), 4),
                     "hapax_ratio": round(sum(v["hapax"] for v in vs) / max(1, sum(r["hapax"] for r in runs)), 4),
                     "row_ratio": round(sum(v["rows"] for v in vs) / (13 * len(runs)), 4)}
    return out


def _spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    if len(x) < 3:
        return None
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    if sx == 0 or sy == 0:
        return None
    return round(sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy), 4)


def agreement_vs_truth(runs):
    """正解を使わない指標（分類方式どうしの一致）が、正解との一致（ARI）を予測するか。"""
    x, y = [], []
    for r in runs:
        a, t = r.get("classifier_agreement_ari"), r["vision"]["v3"].get("ari")
        if a is not None and t is not None:
            x.append(a)
            y.append(t)
    return {"images": len(x), "spearman": _spearman(x, y), "mean_agreement": round(sum(x) / len(x), 4) if x else None,
            "mean_true_ari": round(sum(y) / len(y), 4) if y else None}


def summarize(res):
    if res.get("schema") != "babel-robust-bench-v4":
        raise ValueError("旧評価設計の集計を拒否。bench v4 を再実行する")
    runs = res["runs"]
    main = ["oracle-ngram", "oracle-lcs", "ngram:baseline", "ngram:v2", "ngram:v3", "lcs:baseline", "lcs:v2", "lcs:v3", "soft@0.2", "soft@0.3",
            "soft@0.45"] + [f"vote>={k}" for k in range(2, N_MEMBERS + 1)]
    return {"split": res["split"], "cases": len(runs), "overall": table(runs), "by_family": table(runs, main, "family"),
            "by_condition": table(runs, main, "condition"), "by_vocab": table(runs, main, "vocab"),
            "vote_discrimination": vote_discrimination(runs), "existing_stability": existing_stability_discrimination(runs),
            "dependence_ngram": recognizer_dependence(runs, "ngram"), "dependence_lcs": recognizer_dependence(runs, "lcs"),
            "vision": vision_summary(runs), "agreement_vs_truth": agreement_vs_truth(runs)}
