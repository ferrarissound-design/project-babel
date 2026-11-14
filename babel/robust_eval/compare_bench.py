"""転写だけの比較（babel/robust/compare.py）の判定が、正解つきの画像でどれだけ当たるかを測る。

IMG_4494 では画像がなく、baseline と improved-v3 の転写2本を突き合わせて「どの転写でも有意」な行の組を出した
（docs/ROBUST_STRUCTURE.md 第2節）。その手順そのもの（記号の位置を使わず、行ごとのクラス番号の列だけを使う）を、
正解つきの画像に同じように使い、判定が仕込んだ組に当たるかを数える。

対象は、2つの方式の行数が同じ画像だけ（実画像と同じ条件。行数が違うと突き合わせられない）。対象にならなかった画像の数も記録する。
"""
from __future__ import annotations

import json
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import bench, summary

CFG = {"permutations": 999}
CLASSES = ("どの転写でも有意", "両方の転写で有意", "一方の転写だけで有意")


def run_case(case):
    import cv2

    from babel.robust import compare
    from babel.vision.recognizers import recognize
    rows, planted, _, img, gt, _ = bench.render_case(case)
    planted = set(planted)
    rec = {k: case[k] for k in ("split", "family", "condition", "seed", "vocab", "n_pairs")}
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "page.png"
        cv2.imwrite(str(path), img)
        ra = recognize(path, "baseline", regions=gt.get("regions"))
        rb = recognize(path, "improved-v3", regions=gt.get("regions"))
    A = [list(r["classes"]) for r in ra["rows"] if r["classes"]]
    B = [list(r["classes"]) for r in rb["rows"] if r["classes"]]
    rec["rows"] = [len(A), len(B)]
    rec["eligible"] = len(A) == len(B) and len(A) >= 2
    if not rec["eligible"]:
        return rec
    res = compare.compare_rows(A, B, cfg=dict(CFG, seed=case["seed"]))
    line = [x for x, r in zip(bench._row_lines(ra, gt), ra["rows"]) if r["classes"]]
    out = []
    for v in res["verdicts"]:
        j = bench._judge([(tuple(v["rows"]), None)], line, planted)[0]
        out.append({"gt": j["gt"], "kind": j["kind"], "class": v["class"], "significant_in": v["significant_in"]})
    rec["verdicts"] = out
    rec["pair_agreement_ari"] = res["pair_agreement"]["all_rows"]["adjusted_rand"]
    return rec


def dedupe(verdicts):
    """同じ正解の行の組に戻る判定は、画像ごとに1件にまとめる（判定の強い方を残す）。正解に戻せない判定はそのまま。

    行の破片など、2つの予測の行の組が同じ正解の組に戻ることがある。別々に数えると、同じ仕込みの発見を重ねて数える
    （Codex レビュー、docs/ISSUES.md #89）。
    """
    best, rest = {}, []
    for v in verdicts:
        if not v["gt"]:
            rest.append(v)
            continue
        key = tuple(v["gt"])
        if key not in best or CLASSES.index(v["class"]) < CLASSES.index(best[key]["class"]):
            best[key] = v
    return list(best.values()) + rest


def _score(runs, classes):
    neg = [r for r in runs if r["n_pairs"] == 0]
    pos = [r for r in runs if r["n_pairs"] > 0]

    def found(r):
        return [v for v in dedupe(r["verdicts"]) if v["class"] in classes]
    neg_any = sum(1 for r in neg if found(r))
    hit = sum(sum(1 for v in found(r) if v["kind"] == "planted") for r in pos)
    false = sum(sum(1 for v in found(r) if v["kind"] != "planted") for r in runs)
    planted = 2 * len(pos)
    return {"negative_images": len(neg), "fwer": round(neg_any / len(neg), 4) if neg else None, "fwer_ci95": summary._wilson(neg_any, len(neg)),
            "positive_images": len(pos), "recall": round(hit / planted, 4) if planted else None, "true": hit, "false": false,
            "precision": round(hit / (hit + false), 4) if hit + false else None}


def summarize(runs):
    el = [r for r in runs if r["eligible"]]
    out = {"images": len(runs), "eligible": len(el)}
    if el:
        out["all_four"] = _score(el, CLASSES[:1])
        out["both_transcriptions"] = _score(el, CLASSES[:2])
        out["any"] = _score(el, CLASSES)
        out["one_only"] = _score(el, CLASSES[2:])
    return out


def run(split, out, jobs=4, reuse=False):
    """reuse: 画像ごとの判定（runs）は保存済みのものを使い、集計だけをやり直す。"""
    cs = bench.cases(split)
    t0 = time.time()
    if reuse:
        old = json.loads(Path(out).read_text(encoding="utf-8"))
        runs, t0 = old["runs"], time.time() - old["seconds"]
    else:
        with ProcessPoolExecutor(jobs) as ex:
            runs = list(ex.map(run_case, cs, chunksize=8))
    fams = sorted({r["family"] for r in runs})
    res = {"schema": "babel-robust-compare-bench-v2", "split": split, "cfg": CFG, "seconds": round(time.time() - t0, 1),
           "summary": {"overall": summarize(runs), "by_family": {f: summarize([r for r in runs if r["family"] == f]) for f in fams}}, "runs": runs}
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(bench.SPLITS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--resummarize", action="store_true", help="保存済みの判定から集計だけをやり直す")
    a = ap.parse_args()
    print(json.dumps(run(a.split, a.out, a.jobs, reuse=a.resummarize)["summary"]["overall"], ensure_ascii=False))
