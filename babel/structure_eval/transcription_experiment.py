"""転写の誤差が構造解析に与える影響の実験（画像解析の誤差と構造解析の誤差を分けて測る）。

実験 T1（画像なし、記号列で模擬）: 規則のある人工言語（regular）の 13行・30行で
  clean      … 正しい転写
  substitute … 記号の10%を似た別の記号に誤分類
  oversplit  … 各文字を2つの視覚クラスに分ける（同じ文字が別クラスになる誤り、出現ごとに半々）
  merged     … oversplit を、人が「同じ文字」と訂正した後（apply_corrections の merge）
  を比べる。oversplit では「置き換え可能な記号」の仮説が、同じ文字の2クラスを指すか（＝人に統合を提案できるか）も数える。

実験 T2（合成画像）: 同じ言語を画像に描き、babel vision の形状の類似度 0.6 / 0.7 / 0.8 で転写して解析する。
  描いた文字の位置は分かっているので、転写の誤り（1文字が分割・複数文字の結合・クラスの分けすぎ/まとめすぎ）を
  解析とは別に数える。仮説の記号は、評価のときだけ多数決で元の文字に読み替えて正解の規則と照合する。
"""
from __future__ import annotations

import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from babel.structure import corpus as sc
from babel.structure.analyze import analyze

from .generators import Regular
from .score import match_gold

CFG = {"permutations": 199, "subsamples": 5, "subsample_permutations": 99, "perturbations": 3,
       "perturbation_permutations": 199, "compression_reps": 19, "affix_permutations": 199}


def _rows(seed, n):
    g = Regular(seed)
    return g, g.sample(n)


def _gold_found(analysis, rules, rename=None):
    rename = rename or {}
    found = []
    for r in rules:
        for h in analysis["hypotheses"]:
            k = dict(h["key"])
            if "symbols" in k:
                k["symbols"] = [rename.get(x, x) for x in k["symbols"]]
            if match_gold(r, k):
                found.append(r["label"])
                break
    return found


def t1(seeds=(1000, 1001, 1002), sizes=(13, 30)):
    out = []
    for seed in seeds:
        for n in sizes:
            g, rows = _rows(seed, n)
            rules = g.gold_rules()
            lines = [list("".join(ws)) for ws, _ in rows]
            rng = random.Random(seed)
            alpha = sorted({c for ln in lines for c in ln})
            variants = {
                "clean": lines,
                "substitute": [[rng.choice(alpha) if rng.random() < 0.1 else c for c in ln] for ln in lines],
                "oversplit": [[c + ("'" if rng.random() < 0.5 else "") for c in ln] for ln in lines],
            }
            res = {}
            for name, ls in variants.items():
                a = analyze(sc.from_lines(ls), dict(CFG, seed=seed))
                rename = {c + "'": c for c in alpha} if name == "oversplit" else {}
                subst = [h for h in a["hypotheses"] if h["family"] == "substitution"]
                same_char = sum(1 for h in subst if len({x.rstrip("'") for x in h["symbols"]}) == 1)
                res[name] = {"hypotheses": len(a["hypotheses"]), "gold_found": _gold_found(a, rules, rename),
                             "symbol_types": a["statistics"]["symbol_types"],
                             "substitution_hypotheses": len(subst), "substitution_same_character": same_char}
                if name == "oversplit":
                    merged = sc.apply_corrections(sc.from_lines(ls), {"merge": [[c, c + "'"] for c in alpha]})
                    am = analyze(merged, dict(CFG, seed=seed))
                    res["merged"] = {"hypotheses": len(am["hypotheses"]), "gold_found": _gold_found(am, rules),
                                     "symbol_types": am["statistics"]["symbol_types"],
                                     "identical_to_clean_lines": merged.names(merged.lines[0]) == lines[0]}
            out.append({"seed": seed, "lines": n, "gold_rules": len(rules), "conditions": res})
    return out


# ------------------------------------------------------------------------------------------------ 合成画像

def render(rows, path, scale=0.55, advance=15, line_h=26, margin=8, degraded=False, seed=0):
    """degraded: 文字ごとに大きさ（±20%）・太さ（1〜2）・上下位置（±2px）を揺らし、ぼかしと雑音を加える。"""
    import cv2
    import numpy as np
    rng = random.Random(seed)
    lines = ["".join(ws) for ws, _ in rows]
    width = margin * 2 + advance * max(len(s) for s in lines)
    height = margin * 2 + line_h * len(lines)
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    truth = []
    for li, s in enumerate(lines):
        y = margin + line_h * (li + 1) - 8
        for k, ch in enumerate(s):
            x = margin + advance * k
            if degraded:
                cv2.putText(img, ch, (x, y + rng.randint(-2, 2)), cv2.FONT_HERSHEY_SIMPLEX, scale * rng.uniform(0.8, 1.2),
                            (0, 0, 0), rng.choice((1, 1, 2)), cv2.LINE_8)
            else:
                cv2.putText(img, ch, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_8)
            truth.append({"line": li, "pos": k, "char": ch, "x1": x - 1, "x2": x + advance - 2,
                          "y1": y - line_h + 6, "y2": y + 8})
    if degraded:
        img = cv2.GaussianBlur(img, (3, 3), 0)
        noise = np.random.default_rng(seed).normal(0, 18, img.shape)
        img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img)
    return truth


def _assign(report, truth):
    """検出した記号候補を、描いた文字（位置で対応）に割り当てる。転写の誤りを数える。"""
    by_line = defaultdict(list)
    for t in truth:
        by_line[t["line"]].append(t)
    owner = []                       # 候補ごとの、中心が入る描画文字
    for g in report["glyphs"]:
        x1, y1, x2, y2 = g["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        hit = None
        for t in truth:
            if t["x1"] <= cx <= t["x2"] and t["y1"] <= cy <= t["y2"]:
                hit = t
                break
        owner.append(hit)
    per_char = Counter((t["line"], t["pos"]) for t in owner if t)
    split = sum(1 for v in per_char.values() if v > 1)
    missing = len(truth) - len(per_char)
    merged = 0
    for g in report["glyphs"]:
        x1, y1, x2, y2 = g["box"]
        cover = [t for t in truth if t["x1"] < x2 and x1 < t["x2"] and t["y1"] < y2 and y1 < t["y2"]
                 and min(x2, t["x2"]) - max(x1, t["x1"]) > 4]
        merged += len(cover) > 1
    cls_chars = defaultdict(Counter)
    for g, t in zip(report["glyphs"], owner):
        if t:
            cls_chars[g["class_id"]][t["char"]] += 1
    rename = {f"g{c}": cnt.most_common(1)[0][0] for c, cnt in cls_chars.items()}
    char_classes = defaultdict(set)
    for c, cnt in cls_chars.items():
        for ch in cnt:
            char_classes[ch].add(c)
    impure = sum(1 for cnt in cls_chars.values() if len(cnt) > 1)
    return {"glyph_candidates": len(report["glyphs"]), "drawn_characters": len(truth), "split_characters": split,
            "merged_candidates": merged, "missing_characters": missing, "visual_classes": len(cls_chars),
            "true_characters": len({t["char"] for t in truth}),
            "characters_split_into_several_classes": sum(1 for v in char_classes.values() if len(v) > 1),
            "classes_mixing_characters": impure}, rename


def t2(out_dir, seeds=(1000, 1001), n=13, similarities=(0.7, 0.6, 0.8), degraded_options=(False, True)):
    from babel.structure.transcription import compare
    from babel.vision.glyphs import analyze_image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for seed, degraded in [(s, d) for s in seeds for d in degraded_options]:
        g, rows = _rows(seed, n)
        rules = g.gold_rules()
        img = out_dir / f"regular-{seed}-{n}{'-degraded' if degraded else ''}.png"
        truth = render(rows, img, degraded=degraded, seed=seed)
        text = analyze(sc.from_lines([list("".join(ws)) for ws, _ in rows]), dict(CFG, seed=seed))
        res = {"seed": seed, "lines": n, "image": img.name, "degraded": degraded,
               "text_transcription": {"hypotheses": len(text["hypotheses"]), "gold_found": _gold_found(text, rules)},
               "vision": {}}
        analyses = []
        for s in similarities:
            rep = analyze_image(img, similarity=s)
            err, rename = _assign(rep, truth)
            a = analyze(sc.from_glyph_report(rep), dict(CFG, seed=seed))
            analyses.append((f"similarity={s}", a))
            res["vision"][str(s)] = {"transcription_errors": err, "hypotheses": len(a["hypotheses"]),
                                     "tiers": dict(Counter(h["tier"] for h in a["hypotheses"])),
                                     "gold_found": _gold_found(a, rules, rename),
                                     "mean_transcription_stability": round(
                                         sum(h["stability"]["transcription"] or 0 for h in a["hypotheses"]) /
                                         max(1, len(a["hypotheses"])), 3)}
        cmp = compare(analyses[0][1], analyses[1:])
        res["reproduced_across_settings"] = {
            "hypotheses_in_main": len(cmp["by_hypothesis"]),
            "reproduced_in_all": sum(1 for x in cmp["by_hypothesis"] if all(x["reproduced_in"].values())),
            "reproduced_in_none": sum(1 for x in cmp["by_hypothesis"] if not any(x["reproduced_in"].values())),
            "only_in_alternatives": len(cmp["only_in_alternatives"])}
        out.append(res)
    return out


def run(out="results/structure/transcription-experiment.json", image_dir="results/structure/images"):
    t0 = time.time()
    res = {"T1_simulated_transcription_errors": t1()}
    try:
        res["T2_synthetic_image"] = t2(image_dir)
    except ImportError:
        res["T2_synthetic_image"] = "OpenCV がないので未実施"
    res["seconds"] = round(time.time() - t0, 1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return res


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=1)[:4000])
