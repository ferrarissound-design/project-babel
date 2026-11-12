"""認識の誤差が構造解析に与える影響: python -m babel.vision_eval.impact

同じ人工言語の文章（structure_eval の regular）を画像に描き、次の転写を同じ設定の babel analyze にかける。
  A  正解の転写（描いた記号そのもの）
  B  既存の認識器（baseline）の転写
  C  改善した認識器（improved）の転写
  D  正解の転写に、意図的な誤りを加えたもの（誤りの種類を分けて影響を切り分ける）
     D-sub    記号の10%を別の記号に誤分類
     D-split  クラスの分けすぎ（各記号を出現ごとに半々で2つのクラスに分ける）
     D-seg    記号の10%で分割（1記号→2記号）または結合（隣の記号と1つに）
採点は評価側だけで行う: B/C の予測クラスは、画像上の位置で対応した正解の記号の多数決で読み替える。
対応しない記号（余分な検出）を含む仮説は、存在しない記号についての仮説なので偽と数える。
"""
from __future__ import annotations

import json
import random
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SEEDS = [1000, 1001, 1002, 1003, 1004, 1005]
SIZES = [13, 30]
CONDITIONS = ["clean", "noise", "size_thickness", "spacing"]
CFG = {"permutations": 199, "subsamples": 5, "subsample_permutations": 99, "perturbations": 3,
       "perturbation_permutations": 199, "compression_reps": 19, "affix_permutations": 99}


def _score(analysis, rename, gen, rows_true, cuts_pred=None, gold_cuts=None):
    from babel.structure import boundaries as bd
    from babel.structure_eval.score import Oracle, match_gold
    oracle = Oracle(gen)
    rules = gen.gold_rules()
    found = []
    for r in rules:
        for h in analysis["hypotheses"]:
            k = dict(h["key"])
            if "symbols" in k:
                k["symbols"] = [rename.get(x, "?") for x in k["symbols"]]
            if match_gold(r, k):
                found.append(r["label"])
                break
    tru = fal = unv = unmapped = 0
    for h in analysis["hypotheses"]:
        k = dict(h["key"])
        if "symbols" in k:
            if any(rename.get(x) is None for x in k["symbols"]):
                fal += 1
                unmapped += 1             # 正解のどの記号とも対応しない記号（分割の破片・余分な検出）を含む
                continue
            k["symbols"] = [rename[x] for x in k["symbols"]]
        if k["family"] == "linepair":
            unv += 1                       # 行の番号が転写ごとに変わるので、ここでは検証しない
            continue
        v = oracle.verify(k, rows_true)
        if v is None:
            unv += 1
        elif v:
            tru += 1
        else:
            fal += 1
    seg = {}
    if gold_cuts is not None and cuts_pred is not None:
        for c in analysis["boundaries"]["candidates"]:
            if c["method"] in ("consensus", "mdl", "branching", "unigram"):
                pred = cuts_pred(c["cuts"])
                seg[c["method"]] = round(bd.boundary_f1(pred, gold_cuts), 3)
    stab = [h["stability"]["sampling"] for h in analysis["hypotheses"] if h["stability"].get("sampling") is not None]
    tstab = [h["stability"]["transcription"] for h in analysis["hypotheses"] if h["stability"].get("transcription") is not None]
    return {"hypotheses": len(analysis["hypotheses"]), "gold_rules": len(rules), "gold_found": len(found), "found": found,
            "true": tru, "false": fal, "false_unmapped_symbols": unmapped, "unverifiable": unv,
            "tiers": dict(Counter(h["tier"] for h in analysis["hypotheses"])),
            "mean_sampling_stability": round(statistics.mean(stab), 3) if stab else None,
            "mean_transcription_stability": round(statistics.mean(tstab), 3) if tstab else None,
            "symbol_types": analysis["statistics"]["symbol_types"], "symbols": analysis["statistics"]["symbols_total"],
            "verdict": analysis["verdict"]["code"], "segmentation_f1": seg}


def _gold_cuts(rows):
    out = []
    for ws, _ in rows:
        k, cs = 0, set()
        for w in ws[:-1]:
            k += len(w)
            cs.add(k)
        out.append(cs)
    return out


def _corrupt(lines, kind, rng):
    """lines: 記号名のリストのリスト。戻り値: (誤りを加えた行, 元の記号名への読み替え, 元の位置への対応)"""
    alpha = sorted({s for ln in lines for s in ln})
    out, origin = [], []
    for ln in lines:
        new, org = [], []
        i = 0
        while i < len(ln):
            s = ln[i]
            if kind == "sub" and rng.random() < 0.10:
                new.append(rng.choice(alpha))
                org.append(i)
            elif kind == "split":
                new.append(s + ("'" if rng.random() < 0.5 else ""))
                org.append(i)
            elif kind == "seg" and rng.random() < 0.10:
                if rng.random() < 0.5 or i + 1 >= len(ln):
                    new += [s + "~a", s + "~b"]
                    org += [i, None]
                else:
                    new.append(s + "+" + ln[i + 1])
                    org.append(i)
                    i += 1
            else:
                new.append(s)
                org.append(i)
            i += 1
        out.append(new)
        origin.append(org)
    rename = {}
    for ln in out:
        for s in ln:
            if s.endswith("'"):
                rename[s] = s[:-1]
            elif "~" in s or "+" in s:
                rename[s] = None
            else:
                rename[s] = s
    return out, rename, origin


def _cuts_via(origin_per_line):
    def f(cuts):
        res = []
        for cs, org in zip(cuts, origin_per_line):
            m = set()
            for c in cs:
                if 0 < c < len(org) and org[c] is not None:
                    m.add(org[c])
            res.append({x for x in m if x > 0})
        return res
    return f


def run_text(args):
    """A と D（正解の転写に基づく入力）。"""
    seed, n = args
    from babel.structure import corpus as sc
    from babel.structure.analyze import analyze
    from babel.structure_eval.generators import Regular

    from .synth import text_for
    lines_idx, alpha, rows = text_for(seed, n)
    gen = Regular(seed)
    letters = [[alpha[s] for s in ln] for ln in lines_idx]
    gold = _gold_cuts(rows)
    out = []
    a = analyze(sc.from_lines(letters), dict(CFG, seed=seed))
    ident = {c: c for c in alpha}
    out.append({"seed": seed, "lines": n, "input": "A", "style": None, "condition": None,
                **_score(a, ident, gen, rows, lambda cuts: [set(c) for c in cuts], gold)})
    for kind in ("sub", "split", "seg"):
        rng = random.Random(f"impact-{kind}-{seed}-{n}")
        bad, rename, origin = _corrupt(letters, kind, rng)
        a = analyze(sc.from_lines(bad), dict(CFG, seed=seed))
        out.append({"seed": seed, "lines": n, "input": f"D-{kind}", "style": None, "condition": None,
                    **_score(a, rename, gen, rows, _cuts_via(origin), gold)})
    return out


def run_image(args):
    """B と C（画像を認識した転写）。"""
    seed, n, style, cond = args
    import cv2

    from babel.structure import corpus as sc
    from babel.structure.analyze import analyze
    from babel.structure_eval.generators import Regular
    from babel.vision.evaluate import evaluate, match
    from babel.vision.groundtruth import validate
    from babel.vision.recognizers import recognize

    from .synth import render, text_for
    lines_idx, alpha, rows = text_for(seed, n)
    gen = Regular(seed)
    gold = _gold_cuts(rows)
    img, gt = render(lines_idx, style, cond, seed, n_classes=len(alpha))
    gt = validate(gt)
    out = []
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "page.png"
        cv2.imwrite(str(path), img)
        for inp, method in (("B", "baseline"), ("C", "improved")):
            rep = recognize(path, method)                     # 認識器には画像のパスだけ
            ev = evaluate(rep, gt)
            corpus = sc.from_glyph_report(rep)
            try:
                a = analyze(corpus, dict(CFG, seed=seed))
            except ValueError as e:
                # 認識が崩れて記号が足りない（例: 雑音で画像全体が1つの塊になる）。構造解析はできない、という結果として記録する
                out.append({"seed": seed, "lines": n, "input": inp, "style": style, "condition": cond,
                            "hypotheses": 0, "gold_rules": len(gen.gold_rules()), "gold_found": 0, "found": [], "true": 0,
                            "false": 0, "false_unmapped_symbols": 0, "unverifiable": 0, "tiers": {},
                            "mean_sampling_stability": None, "mean_transcription_stability": None,
                            "symbol_types": len(corpus.symbols), "symbols": corpus.n_symbols, "verdict": "analysis_impossible",
                            "analysis_error": str(e), "segmentation_f1": {},
                            "vision": {"f1": ev.get("detection", {}).get("f1"), "ari": ev.get("classes", {}).get("ari")}})
                continue
            # 評価専用: 予測クラス → 正解の記号（位置で対応した正解の多数決）
            G = [it for it in gt["items"] if it["kind"] == "glyph"]
            P = rep["glyphs"]
            pairs = match([g["box"] for g in G], [p["box"] for p in P])
            votes = defaultdict(Counter)
            gt_pos = {}
            line_index = {ln["id"]: k for k, ln in enumerate(gt["lines"])}
            pos_in_line = {}
            for ln in gt["lines"]:
                its = [g for g in G if g["line"] == ln["id"]]
                for k, g in enumerate(sorted(its, key=lambda g: g["box"][0])):
                    pos_in_line[g["id"]] = k
            for i, j, _ in pairs:
                votes[f"g{P[j]['class_id']}"][alpha[int(G[i]['label'][1:])]] += 1
                gt_pos[j] = (line_index[G[i]["line"]], pos_in_line[G[i]["id"]])
            rename = {k: v.most_common(1)[0][0] for k, v in votes.items()}
            # 予測の行と正解の行の対応（予測の行の記号の多数が属する正解の行）
            origin = []
            row_gt = []
            for row in rep["rows"]:
                idx = row.get("glyph_indices")
                tgt = Counter(gt_pos[j][0] for j in idx if j in gt_pos)
                L = tgt.most_common(1)[0][0] if tgt else None
                row_gt.append(L)
                origin.append([gt_pos[j][1] if j in gt_pos and gt_pos[j][0] == L else None for j in idx])
            def cuts_pred(cuts, origin=origin, row_gt=row_gt):
                res = [set() for _ in gold]
                for cs, org, L in zip(cuts, origin, row_gt):
                    if L is None:
                        continue
                    for c in cs:
                        if 0 < c < len(org) and org[c] is not None and org[c] > 0:
                            res[L].add(org[c])
                return res
            # 正解の行の順に並べた gold（予測の行の数が違っても、正解の行ごとに比べる）
            sc_ = _score(a, rename, gen, rows, cuts_pred, gold)
            out.append({"seed": seed, "lines": n, "input": inp, "style": style, "condition": cond, **sc_,
                        "vision": {"f1": ev.get("detection", {}).get("f1"), "ari": ev.get("classes", {}).get("ari"),
                                   "class_ratio": ev.get("classes", {}).get("class_ratio_predicted_per_true"),
                                   "split": ev.get("segmentation_errors", {}).get("split_glyphs"),
                                   "merge": ev.get("segmentation_errors", {}).get("merged_predictions")}})
    return out


def _agg(vals):
    vals = [v for v in vals if v is not None]
    return {"mean": round(statistics.mean(vals), 3), "sd": round(statistics.pstdev(vals), 3), "n": len(vals)} if vals else None


def summarize(runs):
    out = []
    keys = sorted({(r["lines"], r["input"], r["style"], r["condition"]) for r in runs},
                  key=lambda k: (k[0], k[1], str(k[2]), str(k[3])))
    for n, inp, st, cond in keys:
        rs = [r for r in runs if (r["lines"], r["input"], r["style"], r["condition"]) == (n, inp, st, cond)]
        out.append({"lines": n, "input": inp, "style": st, "condition": cond, "runs": len(rs),
                    "gold_found": _agg([r["gold_found"] for r in rs]), "gold_rules": rs[0]["gold_rules"],
                    "hypotheses": _agg([r["hypotheses"] for r in rs]), "false": _agg([r["false"] for r in rs]),
                    "false_unmapped": _agg([r["false_unmapped_symbols"] for r in rs]),
                    "false_rate": _agg([r["false"] / (r["true"] + r["false"]) if r["true"] + r["false"] else None for r in rs]),
                    "symbol_types": _agg([r["symbol_types"] for r in rs]),
                    "analysis_impossible": sum(r["verdict"] == "analysis_impossible" for r in rs),
                    "sampling_stability": _agg([r["mean_sampling_stability"] for r in rs]),
                    "transcription_stability": _agg([r["mean_transcription_stability"] for r in rs]),
                    "seg_f1_consensus": _agg([r["segmentation_f1"].get("consensus") for r in rs]),
                    "vision_f1": _agg([(r.get("vision") or {}).get("f1") for r in rs]),
                    "vision_ari": _agg([(r.get("vision") or {}).get("ari") for r in rs])})
    return out


def run(out="results/vision/impact.json", jobs=4, seeds=SEEDS, sizes=SIZES, conditions=CONDITIONS, styles=("font", "stroke")):
    t0 = time.time()
    text_tasks = [(s, n) for s in seeds for n in sizes]
    img_tasks = [(s, n, st, c) for s in seeds for n in sizes for st in styles for c in conditions]
    with ProcessPoolExecutor(jobs) as ex:
        runs = [r for rs in ex.map(run_text, text_tasks) for r in rs]
        runs += [r for rs in ex.map(run_image, img_tasks) for r in rs]
    import cv2
    res = {"meta": {"seeds": seeds, "sizes": sizes, "conditions": conditions, "styles": list(styles), "config": CFG,
                    "opencv": cv2.__version__,
                    "seconds": round(time.time() - t0, 1)}, "summary": summarize(runs), "runs": runs}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return res


if __name__ == "__main__":
    r = run()
    print(json.dumps(r["meta"], ensure_ascii=False))
