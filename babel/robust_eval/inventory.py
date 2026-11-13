"""「1回しか出ないクラス」は本当に1回しか出ない記号か（正解つきの画像で測る）。

実画像 IMG_4494 の転写は、79 クラス中 58 クラスが1回だけ。これが記号の種類の多さなのか、同じ記号の分けすぎなのかを、
正解なしで見分ける手掛かりとして「2つの方式（baseline・improved-v3）がどちらも1回だけとみなした記号」を使えるかを調べる。

1件 = bench.py と同じ文と描き方（仕込みありの文だけ）。認識器には画像のパス（と領域の枠）だけを渡す。
"""
from __future__ import annotations

import json
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import bench

METHODS = (("v3", "improved-v3"), ("baseline", "baseline"))


def run_case(case):
    import cv2

    from babel.vision.evaluate import match
    from babel.vision.recognizers import recognize
    rows, planted, detail, img, gt, vocab = bench.render_case(case)
    G = [it for it in gt["items"] if it["kind"] == "glyph"]
    true_count = Counter(g["label"] for g in G)
    rec = {**{k: case[k] for k in ("split", "family", "condition", "seed", "vocab")}, "vocab_used": vocab, "tokens": len(G),
           "types": len(true_count), "true_hapax": sum(1 for v in true_count.values() if v == 1), "methods": {}}
    hapax_gt = {}
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "page.png"
        cv2.imwrite(str(path), img)
        for name, method in METHODS:
            rep = recognize(path, method, regions=gt.get("regions"))
            cls = Counter(g["class_id"] for g in rep["glyphs"])
            pairs = match([g["box"] for g in G], [p["box"] for p in rep["glyphs"]])
            pred_hapax = {i for i, j, _ in pairs if cls[rep["glyphs"][j]["class_id"]] == 1}     # 正解の記号の番号で持つ
            hapax_gt[name] = pred_hapax
            matched = {i for i, _, _ in pairs}
            rec["methods"][name] = {
                "glyphs": len(rep["glyphs"]), "classes": len(cls), "hapax_classes": sum(1 for v in cls.values() if v == 1),
                "matched": len(matched), "hapax_matched": len(pred_hapax),
                "hapax_truly_hapax": sum(1 for i in pred_hapax if true_count[G[i]["label"]] == 1),
                "true_hapax_matched": sum(1 for i in matched if true_count[G[i]["label"]] == 1),
                "repeated_called_hapax": sum(1 for i in pred_hapax if true_count[G[i]["label"]] > 1),
                "true_repeated_matched": sum(1 for i in matched if true_count[G[i]["label"]] > 1)}
    both = hapax_gt["v3"] & hapax_gt["baseline"]
    rec["both"] = {"hapax_in_both": len(both), "truly_hapax": sum(1 for i in both if true_count[G[i]["label"]] == 1)}
    return rec


def summarize(runs, by=None):
    groups = {"all": runs}
    if by:
        groups = {}
        for r in runs:
            groups.setdefault(str(r[by]), []).append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        o = {"images": len(rs), "true_types_per_image": round(sum(r["types"] for r in rs) / len(rs), 1),
             "true_hapax_per_image": round(sum(r["true_hapax"] for r in rs) / len(rs), 1)}
        for name, _ in METHODS:
            ms = [r["methods"][name] for r in rs]

            def tot(k):
                return sum(m[k] for m in ms)
            o[name] = {"classes_per_image": round(tot("classes") / len(rs), 1), "hapax_classes_per_image": round(tot("hapax_classes") / len(rs), 1),
                       "class_ratio": round(tot("classes") / sum(r["types"] for r in rs), 3),
                       "hapax_precision": round(tot("hapax_truly_hapax") / tot("hapax_matched"), 4) if tot("hapax_matched") else None,
                       "hapax_recall": round(tot("hapax_truly_hapax") / tot("true_hapax_matched"), 4) if tot("true_hapax_matched") else None,
                       "repeated_called_hapax_rate": round(tot("repeated_called_hapax") / tot("true_repeated_matched"), 4)
                       if tot("true_repeated_matched") else None}
        n = sum(r["both"]["hapax_in_both"] for r in rs)
        o["both"] = {"hapax_in_both_per_image": round(n / len(rs), 1),
                     "precision": round(sum(r["both"]["truly_hapax"] for r in rs) / n, 4) if n else None}
        out[g] = o
    return out


def run(split, out, jobs=4):
    cs = [c for c in bench.cases(split) if c["n_pairs"] == 2]
    t0 = time.time()
    with ProcessPoolExecutor(jobs) as ex:
        runs = list(ex.map(run_case, cs, chunksize=8))
    res = {"schema": "babel-robust-inventory-v1", "split": split, "cases": len(runs), "seconds": round(time.time() - t0, 1),
           "summary": {"overall": summarize(runs), "by_family": summarize(runs, "family"), "by_vocab": summarize(runs, "vocab"),
                       "by_condition": summarize(runs, "condition")}, "runs": runs}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(bench.SPLITS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    r = run(a.split, a.out, a.jobs)
    print(json.dumps(r["summary"]["overall"], ensure_ascii=False))
