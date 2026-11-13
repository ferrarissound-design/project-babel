"""行の過剰分割の原因の切り分け: python -m babel.vision_eval.lines_diagnose [--split dev]

現行方式（improved の行のまとめ方 = 縦の中心）について、dev の人工画像で次を測る。
1. 尺度: 現行方式が使う「成分の高さの中央値」と、正解の字の高さ（中央値）の比
2. 尺度だけを差し替えたとき（正解の字の高さ・B〜E の字の大きさ S）に、過剰分割がどれだけ残るか
3. 余分な行の内訳: 1つの字の部品が別々の行に分かれたもの（字が裂けた）か、字が丸ごと別の行になったものか
正解の字の高さを使う 2. は原因を調べるための診断で、認識器としては使わない。
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

from . import layout
from .lines_bench import agg


def _line_rows(rows, gt):
    """予測の行（成分の枠のリスト）ごとに、含まれる成分 → (正解の行, 正解の字)。"""
    from babel.vision.evaluate import _inter
    G = [it for it in gt["items"] if it["kind"] == "glyph"]
    out = []
    for r in rows:
        m = []
        for c in r:
            best = max(G, key=lambda g: _inter(c[:4], g["box"]))
            if _inter(c[:4], best["box"]) > 0:
                m.append((best["line"], best["id"]))
        out.append(m)
    return out


def diagnose(split="dev", out=None):
    import cv2

    from babel.vision import lines as L
    from babel.vision.groundtruth import validate
    from babel.vision.recognizers import binarize
    if split == "holdout":
        raise ValueError("holdout は診断に使わない")
    acc = defaultdict(lambda: defaultdict(list))
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        for c in layout.build(wd, split):
            gt = validate(json.loads((wd / "gt" / f"{c['name']}.json").read_text(encoding="utf-8")))
            image = cv2.imread(str(wd / "images" / f"{c['name']}.png"), cv2.IMREAD_COLOR)
            mask = binarize(image, "median")
            Gs = [it for it in gt["items"] if it["kind"] == "glyph"]
            h_true = statistics.median(g["box"][3] - g["box"][1] for g in Gs)
            regs = c["regions"] or [{"id": "page", "box": [0, 0, mask.shape[1], mask.shape[0]]}]
            res = defaultdict(int)
            ratios, s_ratios = [], []
            for reg in regs:
                x1, y1, x2, y2 = reg["box"]
                comps, m = L.components(mask[y1:y2, x1:x2])
                if not comps:
                    continue
                comps = [[a + x1, b + y1, cc + x1, d + y1, e] for a, b, cc, d, e in comps]
                ratios.append(L.median_height(comps) / h_true)
                s_ratios.append(L.glyph_scale(comps) / h_true)
                for name, kw in (("A_current", {}), ("A_true_height", {"scale": h_true}), ("A_scale_S", {"scale": L.glyph_scale(comps)})):
                    rows = L.group_center(comps, **kw)
                    lr = _line_rows(rows, gt)
                    per_line = defaultdict(set)
                    for k, mm in enumerate(lr):
                        for ln, _ in mm:
                            per_line[ln].add(k)
                    res[f"{name}/extra_rows"] += sum(len(v) - 1 for v in per_line.values())
                    res[f"{name}/merged_rows"] += sum(1 for mm in lr if len({ln for ln, _ in mm}) > 1)
                    if name == "A_current":
                        # 余分な行の内訳: 正解の行ごとに、成分が最も多い予測の行を「本体」とし、それ以外の行について
                        # 「本体の行にも部品がある字」の部品だけでできた行（字が裂けた）か、そうでない行（字が丸ごと別の行）か
                        for ln, ks in per_line.items():
                            if len(ks) < 2:
                                continue
                            main = max(ks, key=lambda k: sum(1 for a, _ in lr[k] if a == ln))
                            main_glyphs = {g for a, g in lr[main] if a == ln}
                            for k in ks - {main}:
                                gl = {g for a, g in lr[k] if a == ln}
                                res["torn_rows" if gl <= main_glyphs else "partly_torn_rows" if gl & main_glyphs else "whole_glyph_rows"] += 1
            for key in (("all",), ("family", c["style"]), ("condition", c["condition"]), ("cell", c["style"], c["condition"])):
                k = ":".join(key)
                acc[k]["median_height_over_true"].append(statistics.mean(ratios))
                acc[k]["scale_S_over_true"].append(statistics.mean(s_ratios))
                for name in ("A_current", "A_true_height", "A_scale_S"):
                    acc[k][f"{name}/extra_rows"].append(res[f"{name}/extra_rows"])
                    acc[k][f"{name}/merged_rows"].append(res[f"{name}/merged_rows"])
                for name in ("torn_rows", "partly_torn_rows", "whole_glyph_rows"):
                    acc[k][name].append(res[name])
    res = {"split": split, "note": "A_true_height は正解の字の高さを尺度に入れた診断（認識器としては使わない）",
           "groups": {k: {m: agg(v) for m, v in d.items()} for k, d in acc.items()}}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "val"])
    ap.add_argument("--out")
    a = ap.parse_args()
    r = diagnose(a.split, a.out or f"results/vision/v3/oversplit-diagnosis-{a.split}.json")
    cols = ["median_height_over_true", "scale_S_over_true", "A_current/extra_rows", "A_true_height/extra_rows", "A_scale_S/extra_rows",
            "A_current/merged_rows", "A_true_height/merged_rows", "torn_rows", "partly_torn_rows", "whole_glyph_rows"]
    print("| 群 | " + " | ".join(cols) + " |")
    for k, d in r["groups"].items():
        if not k.startswith("cell"):
            print(f"| {k} | " + " | ".join(f"{d[c]['mean']:.2f}" for c in cols) + " |")
