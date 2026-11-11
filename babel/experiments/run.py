"""実験ランナー。生成 → 解読 → 採点を条件ごと・シードごとに実行して集計する。"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from statistics import mean, pstdev

from babel.decipher.baselines import CONFIGS
from babel.decipher.engine import VERSION, decipher
from babel.decipher.segment import transition_segment, unigram_segment
from babel.evaluate.score import score
from babel.generator.corpus import build_corpus, null_corpus
from babel.generator.languages import FAMILIES

DEV_SEEDS = [0, 1, 2, 3, 4]              # 解読器の調整に使ってよいシード
TEST_SEEDS = list(range(100, 120))       # 最終報告専用。調整に使わない
DEFAULT_GEN = dict(n_docs=130, parallel_frac=0.05, var_rate=0.45, lacuna=0.02, split="random")


def conditions():
    """(実験ID, 条件ラベル, レベル, 設定名, 生成パラメータ)"""
    out = []
    for lv in "ABC":
        out.append(("E1_main", f"level{lv}", lv, "engine", {}))
    for lv in "AC":
        for b in ("baseline_freq", "baseline_ngram"):
            out.append(("E2_baseline", f"level{lv}:{b}", lv, b, {}))
    for lv in "AC":
        for n in (33, 65):
            out.append(("E3_data_size", f"level{lv}:docs={n}", lv, "engine", {"n_docs": n}))
    for f in (0.01, 0.025, 0.10):
        out.append(("E4_parallel", f"levelA:parallel={f}", "A", "engine", {"parallel_frac": f}))
    for v in (0.0, 0.9):
        out.append(("E5_noise", f"levelA:variation={v}", "A", "engine", {"var_rate": v}))
    for la in (0.0, 0.08):
        out.append(("E5_noise", f"levelA:lacuna={la}", "A", "engine", {"lacuna": la}))
    out.append(("E6_unseen", "levelA:heldout=late_era", "A", "engine", {"split": "era"}))
    for a in ("no_variants", "no_morph", "no_em", "no_numeral_solver", "no_abstain"):
        out.append(("E8_ablation", f"levelA:{a}", "A", a, {}))
    out.append(("E8_ablation", "levelB:no_metadata", "B", "no_metadata", {}))
    out.append(("E8_ablation", "levelC:no_numeral_solver", "C", "no_numeral_solver", {}))
    out.append(("E8_ablation", "levelC:no_variants", "C", "no_variants", {}))
    return out


def flatten(r):
    """採点結果 → 数値指標の辞書（None は「該当なし」）。"""
    st = r["structure"]

    def ok(k):
        return None if st.get(k) in (None, "abstain") else float(st[k] == "correct")

    def answered(k):
        return float(st.get(k) not in (None, "abstain"))

    lx, gh, gt = r["lexicon"], r["gloss_heldout"], r["gloss_train"]
    nm = r["numerals"]
    return {
        "seg_f1": r["segmentation"]["f1"], "seg_precision": r["segmentation"]["precision"],
        "seg_recall": r["segmentation"]["recall"], "affix_f1": r["affix_inventory"]["f1"],
        "pos_accuracy": r["pos"]["accuracy"], "variant_f1": r["variants"]["f1"],
        "verb_position_correct": float(st.get("verb_position") == "correct"),
        "affix_position_correct": float(st.get("affix_position") == "correct"),
        "numeral_base_correct": float(st.get("numeral_base") == "correct"),
        "num_noun_correct": float(st.get("num_noun") == "correct"),
        "so_order_accuracy_when_answered": ok("so_order"), "so_order_answered": answered("so_order"),
        "adj_noun_accuracy_when_answered": ok("adj_noun"), "adj_noun_answered": answered("adj_noun"),
        "structure_wrong": float(sum(1 for v in st.values() if v == "wrong")),
        "numeral_accuracy": nm["accuracy"],
        "numeral_wrong_rate": nm["wrong"] / max(1, nm["correct"] + nm["wrong"] + nm["abstain"]),
        "lexicon_precision": lx["precision"], "lexicon_claimed": float(lx["claimed"]),
        "lexicon_correct": float(lx["correct"]), "lexicon_wrong": float(lx["wrong"]),
        "lexicon_coverage": lx["coverage"], "abstain_without_evidence": lx["abstain_rate_without_evidence"],
        "gloss_correct": gh["correct"], "gloss_wrong": gh["wrong"], "gloss_abstain": gh["abstain"],
        "gloss_numeral_correct": gh["numeral_correct"], "feature_jaccard": gh["feature_jaccard"],
        "gloss_train_correct": gt["correct"], "gloss_train_wrong": gt["wrong"],
        "relatedness_p1": r["relatedness"]["p_at_1"], "relatedness_chance": r["relatedness"]["chance"],
        "overclaims": float(r["overclaim"]["overclaims"]),
        "oov_claimed": float(r["oov"]["claimed_anyway"]),
    }


def run_one(task):
    exp, label, level, cfg_name, gen, family, seed = task
    t0 = time.time()
    try:
        kw = dict(DEFAULT_GEN, **gen)
        public, secret = build_corpus(family, seed, level, **kw)
        pred = decipher(public, CONFIGS[cfg_name])
        r = score(pred, secret, public)
        digest = hashlib.sha256(json.dumps({k: v for k, v in pred.items() if k not in ("timing",)},
                                           sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
        return {"experiment": exp, "label": label, "level": level, "config": cfg_name, "family": family,
                "seed": seed, "ok": True, "metrics": flatten(r), "calibration": r["calibration"],
                "errors": r["errors"][:4], "digest": digest, "seconds": round(time.time() - t0, 3),
                "sentences": sum(len(d["sentences"]) for d in public["documents"])}
    except Exception as e:      # 失敗は失敗として記録する（成功扱いにしない）
        return {"experiment": exp, "label": label, "level": level, "config": cfg_name, "family": family,
                "seed": seed, "ok": False, "error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - t0, 3)}


def run_null(task):
    kind, seed = task
    t0 = time.time()
    try:
        public, secret = null_corpus(seed, kind)
        r = score(decipher(public), secret, public)["null_verdict"]
        return {"experiment": "E7_null", "label": f"null:{kind}", "seed": seed, "ok": True, "verdict": r,
                "seconds": round(time.time() - t0, 3)}
    except Exception as e:
        return {"experiment": "E7_null", "label": f"null:{kind}", "seed": seed, "ok": False,
                "error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - t0, 3)}


def _boundary_prf(pred_words, gold_words):
    def cuts(ws):
        out, pos = set(), 0
        for w in ws[:-1]:
            pos += len(w)
            out.add(pos)
        return out
    tp = fp = fn = 0
    for p, g in zip(pred_words, gold_words):
        a, b = cuts(p), cuts(g)
        tp, fp, fn = tp + len(a & b), fp + len(a - b), fn + len(b - a)
    pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return 2 * pr * rc / max(1e-9, pr + rc)


def run_boundary(task):
    family, seed = task
    t0 = time.time()
    try:
        public, secret = build_corpus(family, seed, "C", boundaries=False, lacuna=0.0, n_docs=65)
        lines = [s for d in public["documents"] for s in d["sentences"]]
        gold = [[t["s"] for t in s] for d in public["documents"] for s in secret["tokens"][d["id"]]]
        return {"experiment": "E9_boundary", "label": "no_word_boundaries", "family": family, "seed": seed, "ok": True,
                "metrics": {"word_boundary_f1_engine": _boundary_prf(unigram_segment(lines), gold),
                            "word_boundary_f1_ngram_baseline": _boundary_prf(transition_segment(lines), gold)},
                "seconds": round(time.time() - t0, 3)}
    except Exception as e:
        return {"experiment": "E9_boundary", "label": "no_word_boundaries", "family": family, "seed": seed,
                "ok": False, "error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - t0, 3)}


def _agg(values):
    vs = [v for v in values if v is not None]
    if not vs:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": round(mean(vs), 4), "std": round(pstdev(vs), 4), "n": len(vs)}


def ece(pairs, bins=5):
    """期待較正誤差。pairs: [(信頼度, 正誤)]"""
    if not pairs:
        return None, []
    table, total, err = [], len(pairs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, k) for c, k in pairs if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if sel:
            conf, acc = mean(c for c, _ in sel), mean(k for _, k in sel)
            err += len(sel) / total * abs(conf - acc)
            table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(sel), "confidence": round(conf, 3),
                          "accuracy": round(acc, 3)})
    return round(err, 4), table


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get("GITHUB_SHA", "unknown")


def run_all(mode="quick", seeds=None, jobs=None, experiments=None):
    t0 = time.time()
    seeds = seeds or (DEV_SEEDS[:3] if mode == "quick" else TEST_SEEDS)
    conds = [c for c in conditions() if not experiments or c[0] in experiments]
    tasks = [(*c, fam, s) for c in conds for fam in FAMILIES for s in seeds]
    jobs = jobs or min(8, os.cpu_count() or 1)
    with mp.Pool(jobs) as pool:
        runs = pool.map(run_one, tasks, chunksize=4)
        nulls = pool.map(run_null, [(k, s) for k in ("random", "shuffled") for s in seeds]) \
            if not experiments or "E7_null" in experiments else []
        bounds = pool.map(run_boundary, [(f, s) for f in FAMILIES for s in seeds[: max(3, len(seeds) // 4)]]) \
            if not experiments or "E9_boundary" in experiments else []
    # 再現性: 同じ条件をもう一度実行して出力ハッシュを比べる
    rep_task = ("E10_repro", "levelA", "A", "engine", {}, "KES", seeds[0])
    rep = [run_one(rep_task), run_one(rep_task)]
    first = next((r for r in runs if r["ok"] and r["experiment"] == "E1_main" and r["level"] == "A"
                  and r["family"] == "KES" and r["seed"] == seeds[0]), None)
    digests = [r.get("digest") for r in rep] + ([first["digest"]] if first else [])
    repro = {"digests": digests, "identical": len(set(digests)) == 1 and None not in digests}

    groups = {}
    for r in runs + bounds:
        groups.setdefault((r["experiment"], r["label"]), []).append(r)
    conditions_out = []
    for (exp, label), rs in groups.items():
        good = [r for r in rs if r["ok"]]
        keys = sorted({k for r in good for k in r["metrics"]})
        by_family = {}
        for fam in FAMILIES:
            fr = [r for r in good if r.get("family") == fam]
            by_family[fam] = {k: _agg([r["metrics"].get(k) for r in fr]) for k in keys}
        pairs = [(c, k) for r in good for c, k, kind in r.get("calibration", []) if kind == "lexicon"]
        e, table = ece(pairs)
        conditions_out.append({
            "experiment": exp, "label": label, "level": rs[0].get("level"), "config": rs[0].get("config"),
            "runs": len(rs), "failed": len(rs) - len(good),
            "failures": [{"family": r.get("family"), "seed": r["seed"], "error": r["error"]} for r in rs if not r["ok"]],
            "metrics": {k: _agg([r["metrics"].get(k) for r in good]) for k in keys}, "by_family": by_family,
            "lexicon_ece": e, "lexicon_calibration": table,
            "error_examples": [dict(e_, family=r["family"], seed=r["seed"]) for r in good for e_ in r.get("errors", [])][:15],
            "seconds_mean": round(mean(r["seconds"] for r in rs), 3)})
    null_out = {}
    for kind in ("random", "shuffled"):
        rs = [r for r in nulls if r["label"] == f"null:{kind}" and r["ok"]]
        if rs:
            null_out[kind] = {
                "runs": len(rs),
                "syntax_detected_rate": round(mean(float(bool(r["verdict"]["syntax_detected"])) for r in rs), 4),
                "morphology_detected_rate": round(mean(float(bool(r["verdict"]["morphology_detected"])) for r in rs), 4),
                "meaning_claims_total": sum(r["verdict"]["claims"] for r in rs),
                "structure_claims_mean": round(mean(r["verdict"]["structure_claims"] for r in rs), 3)}
    null_out["failed"] = [r for r in nulls if not r["ok"]]
    all_runs = runs + nulls + bounds
    return {
        "meta": {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": git_commit(),
                 "mode": mode, "seeds": seeds, "seed_role": "development" if set(seeds) <= set(DEV_SEEDS) else "evaluation",
                 "engine_version": VERSION, "python": platform.python_version(),
                 "environment": "github-actions" if os.environ.get("GITHUB_ACTIONS") else "local-container",
                 "run_id": os.environ.get("GITHUB_RUN_ID"), "jobs": jobs, "default_generation": DEFAULT_GEN,
                 "total_runs": len(all_runs), "failed_runs": sum(1 for r in all_runs if not r["ok"]),
                 "seconds": round(time.time() - t0, 1),
                 "sentences_per_language": round(mean(r["sentences"] for r in runs if r["ok"] and r["experiment"] == "E1_main"))
                 if any(r["ok"] and r["experiment"] == "E1_main" for r in runs) else None,
                 "argv": sys.argv[1:]},
        "conditions": conditions_out, "null": null_out, "reproducibility": repro,
        "runs": [{k: v for k, v in r.items() if k not in ("calibration", "errors")} for r in all_runs],
    }
