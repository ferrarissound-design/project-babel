"""行認識の評価: python -m babel.vision_eval.lines_bench --split dev|val|holdout [--methods a,b] [--out FILE]

babel/vision_eval/layout.py の人工画像（行の作りを変えたもの）で、行の検出と、記号の検出・分類を別々に測る。
- 行: 行 F1（行の枠）、過剰分割（1つの正解の行が複数の予測の行に分かれた数）、誤結合（1つの予測の行が複数の正解の行にまたがった数）
- 記号: 検出 F1、分割、結合
- クラス: ARI・NMI
- 処理時間と最大メモリ

dev で方式と閾値を選び、val で確かめる。holdout は --final を付けたときだけ走らせる（最終評価に1回だけ使う）。
認識器には画像のパスと、領域の枠（two_column のときだけ）を渡す。正解は渡さない。
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

from . import layout

METRICS = ["line_f1", "line_precision", "line_recall", "row_ratio", "oversplit_extra_rows", "oversplit_lines", "merged_rows",
           "merged_extra_lines", "f1", "split_glyphs", "merged_predictions", "ari", "nmi", "class_ratio", "seconds", "peak_kib"]


def row_of(ev, rep, case):
    d, c, s, ln = ev.get("detection", {}), ev.get("classes", {}), ev.get("segmentation_errors", {}), ev.get("lines", {})
    e = ln.get("errors", {})
    return {"line_f1": ln.get("f1"), "line_precision": ln.get("precision"), "line_recall": ln.get("recall"),
            "rows": len(rep["rows"]), "row_ratio": round(len(rep["rows"]) / case["gt_lines"], 4),
            "oversplit_extra_rows": e.get("oversplit_extra_rows"), "oversplit_lines": e.get("oversplit_lines"),
            "merged_rows": e.get("merged_rows"), "merged_extra_lines": e.get("merged_extra_lines"),
            "f1": d.get("f1"), "split_glyphs": s.get("split_glyphs"), "merged_predictions": s.get("merged_predictions"),
            "ari": c.get("ari"), "nmi": c.get("nmi"), "class_ratio": c.get("class_ratio_predicted_per_true")}


def agg(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not vals:
        return None
    return {"mean": round(statistics.mean(vals), 4), "sd": round(statistics.pstdev(vals), 4), "n": len(vals)}


def run(split="dev", methods=None, out=None, families=None, conditions=None, seeds=None, final=False, variants=None,
        measure_memory=True, workdir=None):
    """variants: {名前: (method, {segment_components への上書き})}。閾値の試行に使う（dev・val だけ）。"""
    from babel.vision.evaluate import evaluate
    from babel.vision.groundtruth import validate
    from babel.vision.recognizers import recognize
    if split == "holdout" and not final:
        raise ValueError("holdout は最終評価にだけ使う（--final を付ける）。方式・閾値の選択には dev と val を使う")
    if split == "holdout" and variants:
        raise ValueError("holdout で閾値を変えた試行はしない")
    if split == "holdout" and workdir:
        raise ValueError("holdout の画像はキャッシュに残さない（別の分割として再利用されないように）")
    spec = {m: (m, {}) for m in (["baseline", "improved", "improved-v2", "improved-v3"] if methods is None else methods)}
    spec.update(variants or {})
    t0 = time.time()
    runs = []
    with tempfile.TemporaryDirectory() as td:
        wd = Path(workdir or td)
        # 同じ画像で閾値の試行を繰り返すとき、画像を作り直さない。キャッシュには生成条件を記録し、要求と違えば使わない
        # （holdout の画像を置いたディレクトリを dev として渡すと、holdout で閾値を選べてしまう: Codex レビュー、PR #12）
        want = {"split": split, "families": list(families or layout.FAMILIES[split]),
                "conditions": list(conditions or layout.conditions_for(split)), "seeds": list(seeds or layout.SPLITS[split])}
        cache = wd / "cases.json"
        if workdir and cache.exists():
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if not isinstance(saved, dict) or saved.get("spec") != want:
                raise ValueError(f"{cache} は別の条件で作った画像のキャッシュ（{saved.get('spec') if isinstance(saved, dict) else '条件の記録なし'}）。"
                                 f"要求は {want}。別のディレクトリを使う")
            cases = saved["cases"]
        else:
            if workdir and any(wd.glob("images/*")):
                raise ValueError(f"{wd} に生成条件の記録のない画像がある。空のディレクトリを使う")
            cases = layout.build(wd, split, families, conditions, seeds)
            if workdir:
                cache.write_text(json.dumps({"spec": want, "cases": cases}), encoding="utf-8")
        if any(c["split"] != split for c in cases):
            raise ValueError("画像の分割が要求と違う")
        for c in cases:
            img = wd / "images" / f"{c['name']}.png"
            gt = validate(json.loads((wd / "gt" / f"{c['name']}.json").read_text(encoding="utf-8")))
            for name, (m, kw) in spec.items():
                if measure_memory:
                    tracemalloc.start()
                t = time.time()
                rep = recognize(img, m, regions=c["regions"], **kw)
                secs = time.time() - t
                peak = None
                if measure_memory:
                    peak = round(tracemalloc.get_traced_memory()[1] / 1024, 1)
                    tracemalloc.stop()
                r = row_of(evaluate(rep, gt), rep, c)
                r.update(seconds=round(secs, 4), peak_kib=peak)
                runs.append({**{k: c[k] for k in ("name", "split", "style", "condition", "seed", "gt_lines")}, "method": name, **r})
    fams = list(dict.fromkeys(c["style"] for c in cases))
    conds = list(dict.fromkeys(c["condition"] for c in cases))
    summary = defaultdict(dict)
    for name in spec:
        rs = [r for r in runs if r["method"] == name]
        summary[name]["all"] = {k: agg([r[k] for r in rs]) for k in METRICS}
        for f in fams:
            summary[name][f"family:{f}"] = {k: agg([r[k] for r in rs if r["style"] == f]) for k in METRICS}
        for cd in conds:
            summary[name][f"condition:{cd}"] = {k: agg([r[k] for r in rs if r["condition"] == cd]) for k in METRICS}
        summary[name]["totals"] = {"gt_lines": sum(r["gt_lines"] for r in rs), "predicted_rows": sum(r["rows"] for r in rs),
                                   "images_with_exact_row_count": sum(r["rows"] == r["gt_lines"] for r in rs),
                                   "images_without_line_error": sum(r["oversplit_extra_rows"] == 0 and r["merged_rows"] == 0 for r in rs)}
    import cv2
    import numpy
    res = {"meta": {"split": split, "seeds": list(seeds or layout.SPLITS[split]), "families": fams, "conditions": conds,
                    "methods": {k: {"method": m, "overrides": kw} for k, (m, kw) in spec.items()}, "images": len(cases),
                    "opencv": cv2.__version__, "numpy": numpy.__version__, "seconds": round(time.time() - t0, 1)},
           "summary": summary, "runs": runs}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return res


def _c(a, d=2):
    return "—" if not a else f"{a['mean']:.{d}f}"


def table(res, key="all"):
    L = ["| 方式 | 行 F1 | 行数比（予測/正解） | 過剰分割（余分な行/枚） | 誤結合（行/枚） | 検出 F1 | 分割 | 結合 | ARI | NMI | 秒/枚 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m, s in res["summary"].items():
        a = s[key]
        L.append(f"| {m} | {_c(a['line_f1'], 3)} | {_c(a['row_ratio'])} | {_c(a['oversplit_extra_rows'])} | {_c(a['merged_rows'])} | "
                 f"{_c(a['f1'], 3)} | {_c(a['split_glyphs'], 1)} | {_c(a['merged_predictions'], 1)} | {_c(a['ari'], 3)} | "
                 f"{_c(a['nmi'], 3)} | {_c(a['seconds'], 3)} |")
    return "\n".join(L)


def breakdown(res, metric, prefix, d=2):
    keys = [k for k in next(iter(res["summary"].values())) if k.startswith(prefix)]
    L = ["| 方式 | " + " | ".join(k.split(":", 1)[1] for k in keys) + " |", "|---|" + "---:|" * len(keys)]
    for m, s in res["summary"].items():
        L.append(f"| {m} | " + " | ".join(_c(s[k][metric], d) for k in keys) + " |")
    return "\n".join(L)


DEFAULT_GRIDS = {
    "hybrid": {"depth": [0.2, 0.3], "join_gap": [0.35, 0.4, 0.45, 0.5, 0.6], "small": [0.3, 0.4, 0.5]},
    "center_scaled": {"tol": [0.4, 0.6, 0.8, 1.0]},
    "projection": {"bridge": [0.0, 0.05, 0.1, 0.2, 0.3]},
    "overlap": {"tau": [0.3, 0.5, 0.7], "attach": [0.6, 1.0]},
    "baseline": {"tol": [0.4, 0.5, 0.7], "seed_ratio": [0.4, 0.5]},
}


def sweep(split="dev", grid=None, out=None, workdir=None, base="improved-v2", line_method="hybrid"):
    """行認識の方式の閾値の試行（dev で選び、val で確かめる。holdout では使わない）。"""
    import itertools
    if split == "holdout":
        raise ValueError("holdout は閾値の選択に使わない")
    grid = grid or DEFAULT_GRIDS[line_method]
    keys = list(grid)
    variants = {}
    for vals in itertools.product(*(grid[k] for k in keys)):
        kw = dict(zip(keys, vals))
        variants[json.dumps(kw, sort_keys=True)] = (base, {"line_method": line_method, "line_kw": kw})
    res = run(split, methods=[], variants=variants, measure_memory=False, workdir=workdir)
    rows = []
    for name in variants:
        a = res["summary"][name]["all"]
        rows.append({"params": json.loads(name), "line_f1": a["line_f1"]["mean"], "oversplit_extra_rows": a["oversplit_extra_rows"]["mean"],
                     "merged_rows": a["merged_rows"]["mean"], "line_errors": round(a["oversplit_extra_rows"]["mean"] + a["merged_rows"]["mean"], 4),
                     "f1": a["f1"]["mean"], "ari": a["ari"]["mean"],
                     "images_without_line_error": res["summary"][name]["totals"]["images_without_line_error"]})
    rows.sort(key=lambda r: (r["line_errors"], -r["line_f1"]))
    out_res = {"split": split, "images": res["meta"]["images"], "base": base, "line_method": line_method, "grid": grid,
               "criterion": "過剰分割（余分な行）と誤結合（行）の合計が最小（同点は行 F1）", "rows": rows}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(out_res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out_res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=list(layout.SPLITS))
    ap.add_argument("--methods")
    ap.add_argument("--out")
    ap.add_argument("--final", action="store_true", help="holdout の最終評価（1回だけ）")
    a = ap.parse_args()
    r = run(a.split, a.methods.split(",") if a.methods else None, out=a.out or f"results/vision/v3/lines-{a.split}.json", final=a.final)
    print(table(r))
    for metric in ("oversplit_extra_rows", "merged_rows", "line_f1"):
        print(f"\n{metric}\n" + breakdown(r, metric, "condition:") + "\n\n" + breakdown(r, metric, "family:"))
