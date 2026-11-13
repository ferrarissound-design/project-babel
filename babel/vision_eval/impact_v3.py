"""行認識の改善が構造解析に与える影響: python -m babel.vision_eval.impact_v3

同じ人工言語の文章（structure_eval の regular）を、行の作りを変えた画像（layout.py）に描き、次の転写を同じ設定の
babel analyze にかける。
  A   正解の転写
  B   baseline の転写
  C2  improved-v2（行 = 縦の中心）の転写
  D   improved-v3（新しい行認識）の転写
  L-split  正解の転写の各行を中央で2行に分けたもの（記号は正しいまま、行の過剰分割だけを加える）
  L-merge  正解の転写の隣り合う2行を1行にまとめたもの（行の誤結合だけを加える）
L-split・L-merge は、行境界の誤りだけの影響を、記号の検出・分類の誤りから切り分けるための入力。

画像は、行認識の方式と閾値の選択（layout の dev・val）に使っていない字の作り（radical・curve）とシード（8000〜8005）で作る。
採点は babel/vision_eval/impact.py（v1・v2 と同じ）: 予測クラスは画像上の位置で対応した正解の記号の多数決で読み替える。
設計した8規則のうち、行境界に依存する規則（文頭標識・文末標識・各行に1回の2件）と、それ以外を分けて数える。
"""
from __future__ import annotations

import json
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .impact import CFG, _gold_cuts, _score

SEEDS = [8000, 8001, 8002, 8003, 8004, 8005]
SIZES = [13, 30]
FAMILIES = ("radical", "curve")
CONDITIONS = ("plain", "varheight", "diacritic", "tight", "mixed", "photo")
IMAGE_INPUTS = (("B", "baseline"), ("C2", "improved-v2"), ("D", "improved-v3"))
LINE_RULES = ("文頭標識 S", "文末標識 E", "x は各行に1回（必須）", "v は各行に1回（必須）")


def _split_found(found):
    return {"gold_found_line_rules": sum(1 for f in found if f in LINE_RULES),
            "gold_found_other_rules": sum(1 for f in found if f not in LINE_RULES)}


def run_text(args):
    seed, n = args
    from babel.structure import corpus as sc
    from babel.structure.analyze import analyze
    from babel.structure_eval.generators import Regular

    from .synth import text_for
    lines_idx, alpha, rows = text_for(seed, n)
    gen = Regular(seed)
    letters = [[alpha[s] for s in ln] for ln in lines_idx]
    gold = _gold_cuts(rows)
    ident = {c: c for c in alpha}
    out = []

    def add(name, lines, cuts_pred):
        a = analyze(sc.from_lines(lines), dict(CFG, seed=seed))
        s = _score(a, ident, gen, rows, cuts_pred, gold)
        out.append({"seed": seed, "lines": n, "input": name, "style": None, "condition": None, **s, **_split_found(s["found"]),
                    "rows": len(lines)})
    add("A", letters, lambda cuts: [set(c) for c in cuts])
    # 各行を中央で2行に分ける。境界の位置は元の行の位置に戻して比べる
    halves, back = [], []
    for li, ln in enumerate(letters):
        k = len(ln) // 2
        for part, off in ((ln[:k], 0), (ln[k:], k)):
            if part:
                halves.append(part)
                back.append((li, off))

    def cuts_split(cuts):
        res = [set() for _ in letters]
        for cs, (li, off) in zip(cuts, back):
            res[li] |= {c + off for c in cs if c + off > 0}
        return res
    add("L-split", halves, cuts_split)
    pairs, back2 = [], []
    for i in range(0, len(letters), 2):
        pairs.append(letters[i] + (letters[i + 1] if i + 1 < len(letters) else []))
        back2.append((i, len(letters[i])))

    def cuts_merge(cuts):
        res = [set() for _ in letters]
        for cs, (li, k) in zip(cuts, back2):
            for c in cs:
                if 0 < c < k:
                    res[li].add(c)
                elif c > k and li + 1 < len(letters):
                    res[li + 1].add(c - k)
        return res
    add("L-merge", pairs, cuts_merge)
    return out


def run_image(args):
    seed, n, family, cond, inputs = args
    import cv2

    from babel.structure import corpus as sc
    from babel.structure.analyze import analyze
    from babel.structure_eval.generators import Regular
    from babel.vision.evaluate import evaluate, match
    from babel.vision.groundtruth import validate
    from babel.vision.recognizers import recognize

    from .layout import render
    from .synth import text_for
    lines_idx, alpha, rows = text_for(seed, n)
    gen = Regular(seed)
    gold = _gold_cuts(rows)
    img, gt = render(lines_idx, family, cond, seed, n_classes=len(alpha))
    gt = validate(gt)
    out = []
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "page.png"
        cv2.imwrite(str(path), img)
        for inp, method in inputs:
            rep = recognize(path, method, regions=gt.get("regions"))
            ev = evaluate(rep, gt)
            le = ev.get("lines", {}).get("errors", {})
            vis = {"f1": ev.get("detection", {}).get("f1"), "ari": ev.get("classes", {}).get("ari"),
                   "line_f1": ev.get("lines", {}).get("f1"), "rows": len(rep["rows"]), "gt_lines": len(gt["lines"]),
                   "oversplit_extra_rows": le.get("oversplit_extra_rows"), "merged_rows": le.get("merged_rows"),
                   "split": ev.get("segmentation_errors", {}).get("split_glyphs"),
                   "merge": ev.get("segmentation_errors", {}).get("merged_predictions")}
            base = {"seed": seed, "lines": n, "input": inp, "style": family, "condition": cond, "rows": len(rep["rows"])}
            corpus = sc.from_glyph_report(rep)
            try:
                a = analyze(corpus, dict(CFG, seed=seed))
            except ValueError as e:
                out.append({**base, "hypotheses": 0, "gold_rules": len(gen.gold_rules()), "gold_found": 0, "found": [], "true": 0,
                            "false": 0, "false_unmapped_symbols": 0, "unverifiable": 0, "tiers": {},
                            "mean_sampling_stability": None, "mean_transcription_stability": None,
                            "symbol_types": len(corpus.symbols), "symbols": corpus.n_symbols, "verdict": "analysis_impossible",
                            "analysis_error": str(e), "segmentation_f1": {}, "gold_found_line_rules": 0, "gold_found_other_rules": 0,
                            "vision": vis})
                continue
            G = [it for it in gt["items"] if it["kind"] == "glyph"]
            P = rep["glyphs"]
            pairs = match([g["box"] for g in G], [p["box"] for p in P])
            votes, gt_pos, pos_in_line = defaultdict(Counter), {}, {}
            line_index = {ln["id"]: k for k, ln in enumerate(gt["lines"])}
            for ln in gt["lines"]:
                for k, g in enumerate(sorted((g for g in G if g["line"] == ln["id"]), key=lambda g: g["box"][0])):
                    pos_in_line[g["id"]] = k
            for i, j, _ in pairs:
                votes[f"g{P[j]['class_id']}"][alpha[int(G[i]['label'][1:])]] += 1
                gt_pos[j] = (line_index[G[i]["line"]], pos_in_line[G[i]["id"]])
            rename = {k: v.most_common(1)[0][0] for k, v in votes.items()}
            origin, row_gt = [], []
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
            s = _score(a, rename, gen, rows, cuts_pred, gold)
            out.append({**base, **s, **_split_found(s["found"]), "vision": vis})
    return out


def _agg(vals):
    vals = [v for v in vals if v is not None]
    return {"mean": round(statistics.mean(vals), 3), "sd": round(statistics.pstdev(vals), 3), "n": len(vals)} if vals else None


def summarize(runs, by=("lines", "input")):
    out = []
    keys = sorted({tuple(r[k] for k in by) for r in runs}, key=lambda k: tuple(str(x) for x in k))
    for key in keys:
        rs = [r for r in runs if tuple(r[k] for k in by) == key]
        v = [r.get("vision") or {} for r in rs]
        out.append({**dict(zip(by, key)), "runs": len(rs),
                    "gold_found": _agg([r["gold_found"] for r in rs]),
                    "gold_found_line_rules": _agg([r["gold_found_line_rules"] for r in rs]),
                    "gold_found_other_rules": _agg([r["gold_found_other_rules"] for r in rs]),
                    "hypotheses": _agg([r["hypotheses"] for r in rs]), "true": _agg([r["true"] for r in rs]),
                    "false": _agg([r["false"] for r in rs]), "false_unmapped": _agg([r["false_unmapped_symbols"] for r in rs]),
                    "analysis_impossible": sum(r["verdict"] == "analysis_impossible" for r in rs),
                    "sampling_stability": _agg([r["mean_sampling_stability"] for r in rs]),
                    "transcription_stability": _agg([r["mean_transcription_stability"] for r in rs]),
                    "seg_f1_consensus": _agg([r["segmentation_f1"].get("consensus") for r in rs]),
                    "rows": _agg([r["rows"] for r in rs]),
                    "vision_f1": _agg([x.get("f1") for x in v]), "vision_ari": _agg([x.get("ari") for x in v]),
                    "vision_line_f1": _agg([x.get("line_f1") for x in v]),
                    "oversplit_extra_rows": _agg([x.get("oversplit_extra_rows") for x in v]),
                    "merged_rows": _agg([x.get("merged_rows") for x in v])})
    return out


def paired(runs, a, b, key="gold_found"):
    """同じ画像での差（a − b）。"""
    ix = {(r["seed"], r["lines"], r["style"], r["condition"], r["input"]): r for r in runs}
    d = [ix[k[:4] + (a,)][key] - ix[k[:4] + (b,)][key] for k in ix if k[4] == a and k[:4] + (b,) in ix]
    if not d:
        return None
    return {"n": len(d), "mean": round(statistics.mean(d), 3), "sd": round(statistics.pstdev(d), 3),
            "better": sum(x > 0 for x in d), "worse": sum(x < 0 for x in d), "same": sum(x == 0 for x in d)}


def run(out="results/vision/v3/impact-v3.json", jobs=4, seeds=SEEDS, sizes=SIZES, families=FAMILIES, conditions=CONDITIONS,
        inputs=IMAGE_INPUTS):
    t0 = time.time()
    text_tasks = [(s, n) for s in seeds for n in sizes]
    img_tasks = [(s, n, f, c, tuple(inputs)) for s in seeds for n in sizes for f in families for c in conditions]
    with ProcessPoolExecutor(jobs) as ex:
        runs = [r for rs in ex.map(run_text, text_tasks) for r in rs]
        runs += [r for rs in ex.map(run_image, img_tasks) for r in rs]
    import cv2
    img_runs = [r for r in runs if r["style"]]
    res = {"meta": {"seeds": list(seeds), "sizes": list(sizes), "families": list(families), "conditions": list(conditions),
                    "config": CFG, "inputs": [list(x) for x in inputs], "line_rules": list(LINE_RULES), "opencv": cv2.__version__,
                    "seconds": round(time.time() - t0, 1)},
           "summary": summarize(runs), "by_condition": summarize(img_runs, ("lines", "condition", "input")),
           "by_family": summarize(img_runs, ("lines", "style", "input")),
           "paired": {f"{a}-{b}/{k}": paired(img_runs, a, b, k) for a, b in (("D", "C2"), ("D", "B"), ("C2", "B"))
                      for k in ("gold_found", "gold_found_line_rules", "gold_found_other_rules", "false", "hypotheses")},
           "runs": runs}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return res


if __name__ == "__main__":
    r = run()
    print(json.dumps(r["meta"], ensure_ascii=False))
