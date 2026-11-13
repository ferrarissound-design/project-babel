"""旧方式と新方式の行認識の比較画像: python -m babel.vision_eval.lines_figure [--split dev] [--out results/vision/v3/figures]

人工画像を、方式ごとに「同じ行の記号は同じ色の枠」で描いて横に並べる。行の番号を左に書く。
実画像（IMG_4494）はリポジトリにないので、ここで作るのは人工画像の比較だけ。
実画像があるときは python -m babel image-study prepare が同じ描き方の比較画像（rows-comparison.png）を作る。
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from . import layout

EXAMPLES = {"dev": [("stack", "plain", 200), ("stack", "mixed", 202), ("font", "diacritic", 204), ("stroke", "tight", 205)],
            "holdout": [("radical", "plain", 7000), ("radical", "mixed", 7003), ("curve", "diacritic", 7001), ("curve", "photo", 7005),
                        ("radical", "tight", 7002), ("curve", "two_column", 7004)]}


def build(split="dev", out="results/vision/v3/figures", methods=("baseline", "improved-v2", "improved-v3"), final=False):
    import cv2

    from babel.vision.evaluate import evaluate
    from babel.vision.groundtruth import validate
    from babel.vision.recognizers import recognize
    from babel.vision.study import draw_rows, side_by_side

    from .synth import text_for
    if split == "holdout" and not final:
        raise ValueError("holdout の画像は最終評価の後にだけ描く（--final）")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    with tempfile.TemporaryDirectory() as td:
        for fam, cond, seed in EXAMPLES[split]:
            lines, alpha, _ = text_for(seed, layout.n_lines_for(seed))
            img, gt = layout.render(lines, fam, cond, seed, n_classes=len(alpha))
            gt = validate(gt)
            p = Path(td) / "x.png"
            cv2.imwrite(str(p), img)
            panels, rec = [], {"image": f"{split}-{fam}-{cond}-{seed}.png", "gt_lines": len(gt["lines"]), "methods": {}}
            for m in methods:
                rep = recognize(p, m, regions=gt.get("regions"))
                e = evaluate(rep, gt)["lines"]["errors"]
                rec["methods"][m] = {"rows": len(rep["rows"]), "oversplit_extra_rows": e["oversplit_extra_rows"], "merged_rows": e["merged_rows"]}
                panels.append(draw_rows(img, rep, f"{m}: {len(rep['rows'])} rows (gt {len(gt['lines'])})"))
            cv2.imwrite(str(out / rec["image"]), side_by_side(panels))
            index.append(rec)
    (out / f"index-{split}.json").write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return index


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "holdout"])
    ap.add_argument("--out", default="results/vision/v3/figures")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    for r in build(a.split, a.out, final=a.final):
        print(json.dumps(r, ensure_ascii=False))
