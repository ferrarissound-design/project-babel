"""構造解析の評価実験: python -m babel structure-benchmark --split eval

- 生成器 × 行数（5, 13, 30, 100）× シード
- 解析器には単語境界を除いた記号列だけを渡す（正解・規則・生成器の設定は渡さない）
- 同じコーパスに既存の BABEL 解読器（v1.2.0）も当て、変更前（既存の構造判定）と変更後（構造解析）を同じ条件で比べる
- シード: dev（0,1,2）は開発中の確認に使う。eval（1000〜1009）は最終評価専用で、閾値の調整に使わない
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DEV_SEEDS = [0, 1, 2]
EVAL_SEEDS = list(range(1000, 1010))
BENCH_CONFIG = {"permutations": 199, "subsamples": 5, "subsample_permutations": 99, "perturbations": 3,
                "perturbation_permutations": 199, "compression_reps": 19, "affix_permutations": 199}
QUICK_CONFIG = {"permutations": 99, "subsamples": 0, "perturbations": 0, "compression_reps": 0, "affix_permutations": 49}


def _old_babel(public):
    """変更前: 既存の BABEL 解読器の構造判定（統語・形態の検定）と単語分割。"""
    from babel.decipher.engine import decipher
    t = time.time()
    try:
        pred = decipher(json.loads(json.dumps(public)))
    except Exception as e:      # 既存の解読器が短いコーパスで落ちる場合も記録する
        return {"error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - t, 2)}
    v = pred.get("verdict") or {}
    syn, mor = v.get("syntax") or {}, v.get("morphology") or {}
    return {"syntax_present": syn.get("present"), "syntax_z": syn.get("z"), "morphology_present": mor.get("present"),
            "decipherable_structure": v.get("decipherable_structure"), "seconds": round(time.time() - t, 2)}


def run_one(args):
    gname, seed, size, cfg = args
    from babel.structure import boundaries as bd
    from babel.structure.analyze import analyze
    from babel.structure.corpus import from_public

    from .generators import GENERATORS, NULL_GENERATORS, gold_cuts, to_corpus
    from .score import Oracle, score_analysis
    gen = GENERATORS[gname](seed)
    rows = gen.sample(size)
    public = to_corpus(rows, f"{gname}-{seed}-{size}")
    corpus = from_public(public)
    t = time.time()
    a = analyze(corpus, dict(cfg, seed=seed))
    secs = time.time() - t
    oracle = Oracle(gen, null=gname in NULL_GENERATORS)
    sc = score_analysis(a, gen, rows, oracle)
    seg = {}
    if gen.has_words:
        gold = gold_cuts(rows)
        for c in a["boundaries"]["candidates"]:
            seg[c["method"]] = round(bd.boundary_f1([set(x) for x in c["cuts"]], gold), 3)
        seg["_top_by_description_length"] = a["boundaries"]["candidates"][0]["method"]
    comp = a["boundaries"].get("compression_test") or {}
    old = _old_babel(public)
    return {"generator": gname, "principle": gen.principle, "seed": seed, "size_requested": size, "lines": len(rows),
            "symbols": corpus.n_symbols, "symbol_types": len(corpus.symbols), "score": sc,
            "global": {k: v["p_holm"] for k, v in a["global_tests"].items()},
            "segmentation_f1": seg, "compression_p": comp.get("p"),
            "old_babel": old, "seconds": round(secs, 2), "description": gen.description}


def _git():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def summarize(runs):
    by = defaultdict(list)
    for r in runs:
        by[(r["generator"], r["size_requested"])].append(r)
    rows = []
    for (g, n), rs in sorted(by.items(), key=lambda x: (x[0][0], x[0][1])):
        k = len(rs)
        gold_total = sum(len(r["score"]["gold_found"]) + len(r["score"]["gold_missed"]) + len(r["score"]["gold_undecided"]) for r in rs)
        found = sum(len(r["score"]["gold_found"]) for r in rs)
        und = sum(len(r["score"]["gold_undecided"]) for r in rs)
        miss = sum(len(r["score"]["gold_missed"]) for r in rs)
        hyp = sum(r["score"]["hypotheses"] for r in rs)
        tru = sum(r["score"]["true"] for r in rs)
        fal = sum(r["score"]["false"] for r in rs)
        unv = sum(r["score"]["unverifiable"] for r in rs)
        detected = sum(r["score"]["verdict"] != "no_structure_detected" for r in rs)
        old_syn = [r["old_babel"].get("decipherable_structure") for r in rs if "error" not in r["old_babel"]]
        old_err = sum("error" in r["old_babel"] for r in rs)
        seg_methods = sorted({m for r in rs for m in r["segmentation_f1"] if not m.startswith("_")})
        seg = {m: round(sum(r["segmentation_f1"].get(m, 0) for r in rs) / k, 3) for m in seg_methods}
        tiers, ftiers = defaultdict(int), defaultdict(int)
        for r in rs:
            for t, v in r["score"]["tiers"].items():
                tiers[t] += v
            for t, v in r["score"]["false_by_tier"].items():
                ftiers[t] += v
        rows.append({
            "generator": g, "size": n, "runs": k, "mean_lines": round(sum(r["lines"] for r in rs) / k, 1),
            "tiers": dict(tiers), "false_by_tier": dict(ftiers),
            "principle": rs[0]["principle"],
            "gold_rules": gold_total, "gold_found": found, "gold_undecided": und, "gold_missed": miss,
            "gold_recall": round(found / gold_total, 3) if gold_total else None,
            "hypotheses": hyp, "true": tru, "false": fal, "unverifiable": unv,
            "false_discovery_rate": round(fal / (tru + fal), 3) if tru + fal else None,
            "corpora_with_any_false_hypothesis": sum(r["score"]["false"] > 0 for r in rs),
            "structure_detected": detected, "old_babel_structure_detected": sum(bool(x) for x in old_syn),
            "old_babel_errors": old_err,
            "segmentation_f1": seg,
            "mean_seconds": round(sum(r["seconds"] for r in rs) / k, 2),
            "max_seconds": max(r["seconds"] for r in rs),
        })
    return rows


def overall(rows):
    """行数ごとのまとめ: 無作為データでの誤検出、設計した規則の発見率、段階ごとの偽の率。"""
    from .generators import NULL_GENERATORS
    out = []
    for n in sorted({r["size"] for r in rows}):
        rs = [r for r in rows if r["size"] == n]
        null = [r for r in rs if r["generator"] in NULL_GENERATORS]
        gold = [r for r in rs if r["gold_rules"]]
        tiers, ft = defaultdict(int), defaultdict(int)
        for r in rs:
            for t, v in r["tiers"].items():
                tiers[t] += v
            for t, v in r["false_by_tier"].items():
                ft[t] += v
        out.append({
            "size": n,
            "null_corpora": sum(r["runs"] for r in null),
            "null_corpora_with_false_hypothesis": sum(r["corpora_with_any_false_hypothesis"] for r in null),
            "null_corpora_judged_structured": sum(r["structure_detected"] for r in null),
            "null_corpora_old_babel_structured": sum(r["old_babel_structure_detected"] for r in null),
            "designed_rules": sum(r["gold_rules"] for r in gold),
            "designed_rules_found": sum(r["gold_found"] for r in gold),
            "designed_rules_undecided": sum(r["gold_undecided"] for r in gold),
            "structured_corpora": sum(r["runs"] for r in rs if r["generator"] not in NULL_GENERATORS),
            "structured_corpora_detected": sum(r["structure_detected"] for r in rs if r["generator"] not in NULL_GENERATORS),
            "structured_corpora_old_babel_detected": sum(r["old_babel_structure_detected"] for r in rs
                                                         if r["generator"] not in NULL_GENERATORS),
            "false_rate_by_tier": {t: round(ft[t] / tiers[t], 3) if tiers[t] else None for t in "ABC"},
            "hypotheses_by_tier": dict(tiers),
            "mean_seconds": round(sum(r["mean_seconds"] * r["runs"] for r in rs) / sum(r["runs"] for r in rs), 2),
        })
    return out


def run(split="dev", seeds=None, sizes=(5, 13, 30, 100), jobs=None, quick=False, out=None, generators=None):
    from .generators import GENERATORS
    seeds = seeds or (DEV_SEEDS if split == "dev" else EVAL_SEEDS)
    if split == "eval" and set(seeds) & set(DEV_SEEDS):
        raise ValueError("eval に開発シードを混ぜない")
    cfg = QUICK_CONFIG if quick else BENCH_CONFIG
    tasks = []
    for g in generators or GENERATORS:
        for n in sizes:
            for s in seeds:
                tasks.append((g, s, n, cfg))
    t0 = time.time()
    jobs = jobs or max(1, (os.cpu_count() or 2))
    if jobs == 1:
        runs = [run_one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(jobs) as ex:
            runs = list(ex.map(run_one, tasks, chunksize=1))
    # 固定コーパスで、行数の上限に達して同じ大きさになった条件は除く（重複を数えない）
    seen, uniq = set(), []
    for r in runs:
        k = (r["generator"], r["seed"], r["lines"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    res = {"meta": {"split": split, "seeds": seeds, "sizes": list(sizes), "config": cfg, "commit": _git(),
                    "python": platform.python_version(), "seconds": round(time.time() - t0, 1), "jobs": jobs,
                    "generators": {g: GENERATORS[g].description for g in (generators or GENERATORS)}},
           "summary": summarize(uniq), "runs": uniq}
    res["overall"] = overall(res["summary"])
    res["summary_line"] = {"split": split, "runs": len(uniq), "seconds": res["meta"]["seconds"]}
    out = Path(out or f"results/structure/benchmark-{split}{'-quick' if quick else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    from .tables import write_markdown
    write_markdown(res, out.with_suffix(".md"))
    return res
