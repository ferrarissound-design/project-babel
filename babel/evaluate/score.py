"""採点。予測JSONと正解JSONを比較する唯一の場所。解読器からは import されない。

正解にトークン単位の情報（tokens）があれば詳細に採点し、なければ語彙表・構造表だけで採点する
（外部から投入された言語は後者の形式でもよい）。
"""
from __future__ import annotations

from collections import Counter, defaultdict

STRUCT_KEYS = ["verb_position", "so_order", "affix_position", "numeral_base", "adj_noun", "num_noun"]


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def _same_concept(pred, gold, aliases):
    if pred is None or gold is None:
        return False
    if pred == gold or pred in aliases.get(gold, ()):
        return True
    return "+" in gold and set(pred.split("+")) == set(gold.split("+"))


def score(pred: dict, gold: dict, public: dict | None = None) -> dict:
    out = {"language_id": gold.get("language_id"), "level": pred.get("level")}
    aliases = gold.get("aliases", {})
    # ---- 構造 ----
    st, gs = pred.get("structure", {}), gold.get("structure", {})
    struct, calib = {}, []
    for k in STRUCT_KEYS:
        if k not in gs:
            continue
        v = (st.get(k) or {}).get("value")
        conf = (st.get(k) or {}).get("confidence", 0.0) or 0.0
        if v is None:
            struct[k] = "abstain"
        else:
            ok = str(v) == str(gs[k]) or (k == "affix_position" and v == "both" and False)
            struct[k] = "correct" if ok else "wrong"
            calib.append([round(conf, 3), int(ok), "structure"])
    out["structure"] = struct
    # ---- 数詞 ----
    out["null_verdict"] = None
    if gold.get("null"):
        v = pred.get("verdict", {})
        out["null_verdict"] = {"kind": gold["null"], "syntax_detected": v.get("syntax", {}).get("present"),
                               "morphology_detected": v.get("morphology", {}).get("present"),
                               "claims": pred.get("summary", {}).get("claimed", 0),
                               "structure_claims": sum(1 for k in ("verb_position", "numeral_base")
                                                       if (st.get(k) or {}).get("value") is not None)}
        return out
    gtoks, ptoks = gold.get("tokens"), pred.get("tokens", {})
    held = set((public or {}).get("split", {}).get("heldout", []))
    parallel = {(p["doc"], p["sent"]) for p in (public or {}).get("parallel", [])}
    if not gtoks:
        return _score_tables(pred, gold, out, calib)
    # ---- トークン単位 ----
    seg_tp = seg_fp = seg_fn = 0
    seen_types = set()
    pos_ok = pos_n = pos_lab = 0
    lex_votes = defaultdict(Counter)          # 予測語幹 -> 正解概念の分布
    stem_gold_pos = defaultdict(Counter)
    num_votes = defaultdict(Counter)
    gloss = {"train": Counter(), "heldout": Counter()}
    feat = {"train": [0.0, 0], "heldout": [0.0, 0]}
    var_g, var_p = defaultdict(set), defaultdict(set)
    oov = Counter()
    train_stems = set()
    for did in sorted(gtoks):
        part = "heldout" if did in held else "train"
        for si, (gs_, ps_) in enumerate(zip(gtoks[did], ptoks.get(did, []))):
            if len(gs_) != len(ps_):
                out.setdefault("token_mismatch", 0)
                out["token_mismatch"] += 1
                continue
            for g, p in zip(gs_, ps_):
                if g.get("lacuna") or p.get("lacuna"):
                    continue
                var_g[g["canon"]].add(g["s"])
                var_p[p["norm"]].add(g["s"])
                if part == "train":
                    train_stems.add(p["stem"])
                    if g["s"] not in seen_types:
                        seen_types.add(g["s"])
                        a, b = set(p["cuts"]), set(g["cuts"])
                        seg_tp += len(a & b)
                        seg_fp += len(a - b)
                        seg_fn += len(b - a)
                    pos_n += 1
                    if p["pos"] != "X":
                        pos_lab += 1
                        pos_ok += p["pos"] == g["pos"]
                    stem_gold_pos[p["stem"]][g["pos"]] += 1
                    if g["pos"] == "NUM":
                        num_votes[p["norm"]][g["concept"]] += 1
                    else:
                        lex_votes[p["stem"]][g["concept"]] += 1
                # 語釈（対訳が与えられた文は除く）
                if (did, si) in parallel:
                    continue
                c = gloss[part]
                if part == "heldout" and p["stem"] not in train_stems:
                    oov["tokens"] += 1
                    oov["claimed"] += p["concept"] is not None
                if g["pos"] == "NUM":
                    key = "num_"
                else:
                    key = ""
                if p["concept"] is None:
                    c[key + "abstain"] += 1
                elif _same_concept(p["concept"], g["concept"], aliases):
                    c[key + "correct"] += 1
                else:
                    c[key + "wrong"] += 1
                gf, pf = set(g["feats"]), set(p["feats"])
                if gf or pf:
                    feat[part][0] += len(gf & pf) / len(gf | pf)
                    feat[part][1] += 1
    out["segmentation"] = _prf(seg_tp, seg_fp, seg_fn)
    out["pos"] = {"accuracy": round(pos_ok / pos_n, 4) if pos_n else 0.0,
                  "labelled": round(pos_lab / pos_n, 4) if pos_n else 0.0,
                  "accuracy_when_labelled": round(pos_ok / pos_lab, 4) if pos_lab else 0.0}
    gp = {(a, b) for ws in var_g.values() for a in ws for b in ws if a < b}
    pp = {(a, b) for ws in var_p.values() for a in ws for b in ws if a < b}
    out["variants"] = _prf(len(gp & pp), len(pp - gp), len(gp - pp))
    out["variants"]["gold_pairs"] = len(gp)
    # ---- 接辞目録（正解の接辞形のうち、解読器が接辞として切り出したもの）----
    gaff, paff = Counter(), Counter()
    for did in sorted(gtoks):
        if did in held:
            continue
        for gs_, ps_ in zip(gtoks[did], ptoks.get(did, [])):
            if len(gs_) != len(ps_):
                continue
            for g, p in zip(gs_, ps_):
                if g.get("lacuna") or p.get("lacuna"):
                    continue
                cuts = [0] + list(g["cuts"]) + [len(g["s"])]
                for (a, b), m in zip(zip(cuts, cuts[1:]), g["morphs"]):
                    if m[1] == "affix":
                        gaff[g["s"][a:b]] += 1
                pc = [0] + list(p["cuts"]) + [len(p["s"])]
                stem_i = None
                segs = [p["s"][a:b] for a, b in zip(pc, pc[1:])]
                if len(segs) > 1:
                    stem_i = max(range(len(segs)), key=lambda i: len(segs[i]))
                    for i, sg in enumerate(segs):
                        if i != stem_i:
                            paff[sg] += 1
    gset = {a for a, c in gaff.items() if c >= 3}
    pset = {a for a, c in paff.items() if c >= 3}
    out["affix_inventory"] = _prf(len(gset & pset), len(pset - gset), len(gset - pset))
    # ---- 数詞の値（語形タイプ単位）----
    pnum = {x: v for x, v in (pred.get("numerals", {}).get("values") or {}).items()}
    nres = Counter()
    for form, votes in num_votes.items():
        gv = int(votes.most_common(1)[0][0][1:])
        if form not in pnum:
            nres["abstain"] += 1
        elif pnum[form] == gv:
            nres["correct"] += 1
            calib.append([round(pred["numerals"].get("confidence", 0.0), 3), 1, "numeral"])
        else:
            nres["wrong"] += 1
            calib.append([round(pred["numerals"].get("confidence", 0.0), 3), 0, "numeral"])
    nres["false_numeral"] = sum(1 for f in pnum if f not in num_votes)
    tot = max(1, nres["correct"] + nres["wrong"] + nres["abstain"])
    out["numerals"] = {**{k: nres[k] for k in ("correct", "wrong", "abstain", "false_numeral")},
                       "accuracy": round(nres["correct"] / tot, 4), "relative": pred.get("numerals", {}).get("relative")}
    # ---- 語彙（主張した意味の適合率、保留の妥当性）----
    lres, errors = Counter(), []
    homonym_forms = set()
    for h in (gold.get("grammar") or {}).get("homonyms", []):
        homonym_forms.update(h)
    seen_in_parallel = set()
    if public:
        for d_, s_ in parallel:
            for p in ptoks.get(d_, [[]])[s_] if s_ < len(ptoks.get(d_, [])) else []:
                if not p.get("lacuna"):
                    seen_in_parallel.add(p["stem"])
    known_stems = set()
    for e in pred.get("lexicon", []):
        votes = lex_votes.get(e["stem"])
        if not votes:
            continue
        gc, n = votes.most_common(1)[0]
        purity = n / sum(votes.values())
        if e["status"] == "claimed":
            if e.get("reason") == "既知語として与えられた":
                known_stems.add(e["stem"])
                continue
            ok = _same_concept(e["meaning"], gc, aliases)
            lres["correct" if ok else "wrong"] += 1
            calib.append([round(e["confidence"], 3), int(ok), "lexicon"])
            if not ok:
                cause = "同音異義語・多義語" if gc in homonym_forms or purity < 0.8 else \
                    "証拠が1例のみ" if e.get("evidence", 0) <= 1 else \
                    "語幹の切り出し誤り" if purity < 0.95 else "共起語との取り違え"
                if e.get("reason") == "出土地との偏り":
                    cause = "地名と地元人名の混同（出土地の偏り）"
                errors.append({"stem": e["stem"], "predicted": e["meaning"], "gold": gc,
                               "confidence": e["confidence"], "cause": cause})
        else:
            lres["abstain"] += 1
            if e["stem"] not in seen_in_parallel:
                lres["abstain_no_evidence"] += 1
        if e["stem"] not in seen_in_parallel:
            lres["no_evidence_total"] += 1
    claimed = lres["correct"] + lres["wrong"]
    out["lexicon"] = {"claimed": claimed, "correct": lres["correct"], "wrong": lres["wrong"],
                      "abstain": lres["abstain"],
                      "precision": round(lres["correct"] / claimed, 4) if claimed else None,
                      "coverage": round(claimed / max(1, claimed + lres["abstain"]), 4),
                      "abstain_rate_without_evidence": round(lres["abstain_no_evidence"] / lres["no_evidence_total"], 4)
                      if lres["no_evidence_total"] else None}
    out["errors"] = sorted(errors, key=lambda e: -e["confidence"])[:12]
    for part in ("train", "heldout"):
        c = gloss[part]
        n = c["correct"] + c["wrong"] + c["abstain"]
        nn = c["num_correct"] + c["num_wrong"] + c["num_abstain"]
        out["gloss_" + part] = {"tokens": n, "correct": round(c["correct"] / n, 4) if n else 0.0,
                                "wrong": round(c["wrong"] / n, 4) if n else 0.0,
                                "abstain": round(c["abstain"] / n, 4) if n else 0.0,
                                "numeral_correct": round(c["num_correct"] / nn, 4) if nn else None,
                                "feature_jaccard": round(feat[part][0] / feat[part][1], 4) if feat[part][1] else None}
    out["oov"] = {"heldout_tokens_with_unseen_stem": oov["tokens"], "claimed_anyway": oov["claimed"]}
    # ---- 過剰主張 ----
    has_evidence = bool(public and (public.get("parallel") or public.get("known") or
                                    any(d.get("meta") for d in public.get("documents", []))))
    out["overclaim"] = {"semantic_evidence_available": has_evidence,
                        "meaning_claims": claimed + len(known_stems),
                        "overclaims": 0 if has_evidence else claimed}
    # ---- 意味関係（最近傍が同じ意味クラスか）----
    sem_of = defaultdict(Counter)
    for did in sorted(gtoks):
        for gs_, ps_ in zip(gtoks[did], ptoks.get(did, [])):
            if len(gs_) == len(ps_):
                for g, p in zip(gs_, ps_):
                    if not g.get("lacuna") and not p.get("lacuna"):
                        sem_of[p["stem"]][g["sem"]] += 1
    hit = n = 0
    cls_count = Counter()
    rel = pred.get("relatedness", {})
    for s in rel:
        if s in sem_of:
            cls_count[sem_of[s].most_common(1)[0][0]] += 1
    for s, nb in rel.items():
        if nb and s in sem_of and nb[0][0] in sem_of:
            n += 1
            hit += sem_of[s].most_common(1)[0][0] == sem_of[nb[0][0]].most_common(1)[0][0]
    tot = sum(cls_count.values())
    chance = sum(c * (c - 1) for c in cls_count.values()) / (tot * (tot - 1)) if tot > 1 else 0.0
    out["relatedness"] = {"p_at_1": round(hit / n, 4) if n else None, "chance": round(chance, 4), "n": n}
    out["calibration"] = calib
    return out


def _score_tables(pred, gold, out, calib):
    """トークン単位の正解がない場合: 語彙表・数詞表だけで採点する（外部コーパス用）。"""
    by_form = {}
    for e in pred.get("lexicon", []):
        by_form.setdefault(e["stem"], e)
        for f in e.get("forms", []):
            by_form.setdefault(f, e)
    res, errors = Counter(), []
    aliases = gold.get("aliases", {})
    for g in gold.get("lexicon", []):
        e = by_form.get(g["form"])
        if e is None:
            res["not_found"] += 1
        elif e["status"] != "claimed":
            res["abstain"] += 1
        else:
            ok = _same_concept(e["meaning"], g["meaning"], aliases)
            res["correct" if ok else "wrong"] += 1
            calib.append([round(e["confidence"], 3), int(ok), "lexicon"])
            if not ok:
                errors.append({"stem": e["stem"], "predicted": e["meaning"], "gold": g["meaning"],
                               "confidence": e["confidence"], "cause": "未分類"})
    claimed = res["correct"] + res["wrong"]
    out["lexicon"] = {"claimed": claimed, "correct": res["correct"], "wrong": res["wrong"],
                      "abstain": res["abstain"], "not_found": res["not_found"],
                      "precision": round(res["correct"] / claimed, 4) if claimed else None,
                      "coverage": round(claimed / max(1, len(gold.get("lexicon", []))), 4)}
    pnum = pred.get("numerals", {}).get("values") or {}
    nres = Counter()
    for g in gold.get("numerals", []):
        if g["form"] not in pnum:
            nres["abstain"] += 1
        else:
            nres["correct" if pnum[g["form"]] == g["value"] else "wrong"] += 1
    tot = max(1, sum(nres.values()))
    out["numerals"] = {**{k: nres[k] for k in ("correct", "wrong", "abstain")},
                       "accuracy": round(nres["correct"] / tot, 4)}
    all_claims = sum(1 for e in pred.get("lexicon", []) if e["status"] == "claimed")
    out["overclaim"] = {"meaning_claims": all_claims,
                        "overclaims": all_claims if gold.get("no_semantic_evidence") else 0}
    out["errors"] = errors[:12]
    out["calibration"] = calib
    return out
