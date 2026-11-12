"""画像認識の方式の比較: python -m babel vision-benchmark --split dev|val|test

人工画像は実行のたびに決定的に生成する（画像と正解を別のディレクトリに書き、認識器には画像のパスだけを渡す）。
dev で方式と閾値を選び、val で確かめ、test は最終評価にだけ使う（test で閾値を変えない）。
"""
from __future__ import annotations

import json
import statistics
import tempfile
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

from . import synth

METRICS = ["f1", "precision", "recall", "ari", "nmi", "split_glyphs", "merged_predictions", "class_ratio",
           "oversplit_extra_clusters", "mismerged_extra_labels", "line_f1", "non_glyph", "seconds", "peak_kib"]


def _row(ev):
    d, c, s, ln = ev.get("detection", {}), ev.get("classes", {}), ev.get("segmentation_errors", {}), ev.get("lines", {})
    return {"f1": d.get("f1"), "precision": d.get("precision"), "recall": d.get("recall"),
            "ari": c.get("ari"), "nmi": c.get("nmi"), "split_glyphs": s.get("split_glyphs"),
            "merged_predictions": s.get("merged_predictions"), "class_ratio": c.get("class_ratio_predicted_per_true"),
            "oversplit_extra_clusters": c.get("oversplit_extra_clusters"),
            "mismerged_extra_labels": c.get("mismerged_extra_labels"), "line_f1": ln.get("f1"),
            "non_glyph": d.get("non_glyph_detected_as_glyph")}


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": round(statistics.mean(vals), 4), "sd": round(statistics.pstdev(vals), 4), "n": len(vals)}


def run(split="dev", methods=None, styles=synth.STYLES, conditions=synth.CONDITIONS, out=None, workdir=None):
    from babel.vision.evaluate import evaluate
    from babel.vision.groundtruth import validate
    from babel.vision.recognizers import METHODS, recognize
    methods = methods or ["baseline"] + list(METHODS)
    methods = list(dict.fromkeys(methods))
    tmp = tempfile.TemporaryDirectory() if workdir is None else None
    wd = Path(workdir or tmp.name)
    t0 = time.time()
    cases = synth.build(wd, split, styles, conditions)
    runs = []
    for c in cases:
        img = wd / "images" / f"{c['name']}.png"
        gt = validate(json.loads((wd / "gt" / f"{c['name']}.json").read_text(encoding="utf-8")))
        for m in methods:
            tracemalloc.start()
            t = time.time()
            rep = recognize(img, m)                       # 認識器には画像のパスだけを渡す
            secs = time.time() - t
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            r = _row(evaluate(rep, gt))
            r.update(seconds=round(secs, 4), peak_kib=round(peak / 1024, 1))
            runs.append({**c, "method": m, **r})
    summary = defaultdict(dict)
    for m in methods:
        rs = [r for r in runs if r["method"] == m]
        summary[m]["all"] = {k: _agg([r[k] for r in rs]) for k in METRICS}
        for st in styles:
            for cond in conditions:
                sub = [r for r in rs if r["style"] == st and r["condition"] == cond]
                summary[m][f"{st}/{cond}"] = {k: _agg([r[k] for r in sub]) for k in METRICS}
    import cv2
    import numpy
    res = {"meta": {"split": split, "seeds": synth.SPLITS[split], "methods": methods, "styles": list(styles),
                    "opencv": cv2.__version__, "numpy": numpy.__version__,
                    "conditions": list(conditions), "images": len(cases), "seconds": round(time.time() - t0, 1)},
           "summary": summary, "runs": runs}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        Path(out).with_suffix(".md").write_text(markdown(res), encoding="utf-8")
    if tmp:
        tmp.cleanup()
    return res


def _cell(a, d=3):
    return "—" if not a else f"{a['mean']:.{d}f} ± {a['sd']:.{d}f}"


def markdown(res):
    m = res["meta"]
    L = [f"# 画像認識の方式の比較（{m['split']}、シード {m['seeds'][0]}〜{m['seeds'][-1]}、画像 {m['images']} 枚）", "",
         "値は画像ごとの指標の平均 ± 標準偏差。人工画像の正解は生成時の座標とクラス（評価専用）。", "",
         "## 全条件", "",
         "| 方式 | 検出 F1 | ARI | NMI | 分割 | 結合 | クラス数比（予測/正解） | 行 F1 | 文字以外の誤検出 | 秒/枚 | 最大メモリ KiB |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for meth, s in res["summary"].items():
        a = s["all"]
        L.append(f"| {meth} | {_cell(a['f1'])} | {_cell(a['ari'])} | {_cell(a['nmi'])} | {_cell(a['split_glyphs'], 1)} | "
                 f"{_cell(a['merged_predictions'], 1)} | {_cell(a['class_ratio'], 2)} | {_cell(a['line_f1'])} | "
                 f"{_cell(a['non_glyph'], 1)} | {_cell(a['seconds'], 3)} | {_cell(a['peak_kib'], 0)} |")
    L += ["", "## 条件ごとの検出 F1 / ARI", ""]
    keys = [k for k in next(iter(res["summary"].values())) if k != "all"]
    L += ["| 方式 | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
    for meth, s in res["summary"].items():
        L.append(f"| {meth} | " + " | ".join(
            f"{s[k]['f1']['mean']:.2f} / {s[k]['ari']['mean']:.2f}" if s[k]["f1"] and s[k]["ari"] else
            (f"{s[k]['f1']['mean']:.2f} / —" if s[k]["f1"] else "—") for k in keys) + " |")
    L += ["", "ARI は、検出で正解と対応した記号だけで計算する（検出できなかった記号はクラスの評価に入らない）。"
          "検出 F1 が低い条件の ARI は少数の記号の値なので、F1 と併せて読むこと。", ""]
    return "\n".join(L) + "\n"


def baseline_threshold_sweep(similarities=(0.6, 0.65, 0.7, 0.75, 0.8, 0.85), out="results/vision/baseline-threshold-dev.json"):
    """既存方式の類似度の閾値を dev だけで変えたときの成績（改善が閾値の違いだけで説明できるかを確かめる）。"""
    import cv2

    from babel.vision.evaluate import evaluate
    from babel.vision.glyphs import analyze_image
    from babel.vision.groundtruth import validate
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        cases = synth.build(wd, "dev")
        rows = []
        for s in similarities:
            f1, ari = [], []
            for c in cases:
                ev = evaluate(analyze_image(wd / "images" / f"{c['name']}.png", similarity=s),
                              validate(json.loads((wd / "gt" / f"{c['name']}.json").read_text(encoding="utf-8"))))
                f1.append(ev["detection"]["f1"])
                if ev["classes"].get("ari") is not None:
                    ari.append(ev["classes"]["ari"])
            rows.append({"similarity": s, "f1": _agg(f1), "ari": _agg(ari)})
    res = {"split": "dev", "opencv": cv2.__version__, "rows": rows}
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return res
