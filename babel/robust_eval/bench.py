"""行の組の仮説が「本当にある構造」か「認識の誤りが作ったもの」かを、正解つきの画像で測る。

1件 = 文（texts.planted_text）1つ × 描き方1つ。画像を方式の集まり（babel/robust/ensemble.py）で転写し、方式ごとに
行の組の検定を行い、見つかった組を正解の行に戻して、仕込んだ組（真）か、それ以外（偽）かを数える。

比べる判定:
  ngram:<方式>  既存の babel analyze の linepair 族（共有する 2〜4 記号列の数）。現状
  lcs:<方式>    順序を保った共通部分列の検定（babel/robust/rowsim.py）を、1つの転写に使う
  soft          分類を経ない検定（字形の類似度を重みにする。切り出しは improved-v3）
  vote>=k       lcs を方式ごとに行い、k 個以上の方式で有意な組だけを採る
  oracle-*      正解の転写に同じ検定を使う（認識の誤りがないときの上限）

認識器には画像のパス（と領域の枠）だけを渡す。正解は採点にだけ使う。
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ANALYZE_CFG = {"permutations": 199, "subsamples": 5, "subsample_permutations": 99, "perturbations": 3,
               "perturbation_permutations": 199, "compression_reps": 19, "affix_permutations": 99}   # impact.py と同じ
ROW_CFG = {"permutations": 999}
NGRAM_MEMBERS = ("v3", "v2", "baseline")
SOFT_TAUS = (0.2, 0.3, 0.45)

SPLITS = {
    # 字の作り・行の作りは babel/vision_eval/layout.py。dev は方式の設計と閾値の選択に使う
    "dev": {"renderer": "layout", "families": ("font", "stroke", "stack"), "conditions": ("plain", "diacritic", "tight", "mixed"),
            "seeds": range(100, 108), "vocabs": (30, 80, 250), "n_pairs": (0, 2)},
    # holdout-synth: dev に出てこない字の作り（radical・curve）と劣化（photo）。閾値を固定してから1回だけ
    "holdout-synth": {"renderer": "layout", "families": ("radical", "curve"), "conditions": ("plain", "diacritic", "tight", "mixed", "photo"),
                      "seeds": range(9100, 9110), "vocabs": (30, 80, 250), "n_pairs": (0, 2)},
    # holdout-font: 実在の文字体系・実在のフォント（babel/robust_eval/realfont.py）。閾値を固定してから1回だけ
    "holdout-font": {"renderer": "realfont", "families": ("georgian", "armenian", "thai", "katakana", "hiragana", "kanji", "hangul"),
                     "conditions": ("clean", "degraded"), "seeds": range(9200, 9208), "vocabs": (30, 80, 250), "n_pairs": (0, 2)},
}


def cases(split, limit=None):
    sp = SPLITS[split]
    out = []
    for seed in sp["seeds"]:
        for fam in sp["families"]:
            for cond in sp["conditions"]:
                for vocab in sp["vocabs"]:
                    for npairs in sp["n_pairs"]:
                        out.append({"split": split, "renderer": sp["renderer"], "family": fam, "condition": cond, "seed": seed,
                                    "vocab": vocab, "n_pairs": npairs})
    return out[:limit] if limit else out


def text_seed(case):
    """Give each evaluated image its own text, including negative/positive counterparts.

    The renderer seed stays fixed for paired rendering comparisons; it must not
    also make all conditions/families share an identical text realization.
    """
    key = [case[k] for k in ("split", "renderer", "family", "condition", "seed", "vocab", "n_pairs")]
    return int.from_bytes(hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).digest()[:8], "big")


def _vocab_for(case):
    """字の種類が足りない描き方では語彙を詰める（記録に残す）。"""
    if case["renderer"] == "layout":
        from babel.vision_eval import synth
        return min(case["vocab"], len(synth.FONT_CHARS)) if case["family"] == "font" else case["vocab"]
    from . import realfont
    return min(case["vocab"], realfont.max_vocab(case["family"]))


def render_case(case):
    from babel.vision.groundtruth import validate

    from . import texts
    vocab = _vocab_for(case)
    rows, planted, detail = texts.planted_text(text_seed(case), vocab=vocab, n_pairs=case["n_pairs"])
    if case["renderer"] == "layout":
        from babel.vision_eval import layout
        img, gt = layout.render(rows, case["family"], case["condition"], case["seed"], n_classes=vocab)
    else:
        from . import realfont
        img, gt = realfont.render(rows, case["family"], case["seed"], vocab, degrade=case["condition"] == "degraded")
    return rows, planted, detail, img, validate(gt), vocab


def _row_lines(rep, gt):
    """予測の行 → 正解の行の番号（なければ None）。

    予測の記号ごとに、重なる面積が最大の正解の記号の行を取る。複数の正解行が混ざる予測行は対応不能とする。
    多数決で一方に割り当てると、結合された行を仕込み組として誤採点してしまう。IoU の閾値は使わない:
    字の破片だけでできた行（行の過剰分割）も、元の正解の行に戻す。破片の行どうしの類似が仕込んだ組に当たれば真、
    同じ正解の行の破片どうしなら same_line（偽）と数える。
    """
    G = [it for it in gt["items"] if it["kind"] == "glyph"]
    line_index = {ln["id"]: k for k, ln in enumerate(gt["lines"])}
    out = []
    for row in rep["rows"]:
        lines = set()
        for j in row["glyph_indices"]:
            p = rep["glyphs"][j]["box"]
            best, best_a = None, 0
            for g in G:
                q = g["box"]
                a = max(0, min(p[2], q[2]) - max(p[0], q[0])) * max(0, min(p[3], q[3]) - max(p[1], q[1]))
                if a > best_a:
                    best, best_a = g, a
                # 予測の記号が正解の記号の面積の半分以上を覆うなら、その記号の行にも属する（上下の行の字が1つの記号に
                # 結合された場合。最大の重なりだけを見ると、結合された記号が一方の行にだけ対応して混合行の検査をすり抜ける。
                # Codex レビュー、docs/ISSUES.md #84。babel/vision/evaluate.py の line_errors と同じ基準）
                if 2 * a >= (q[2] - q[0]) * (q[3] - q[1]) and a > 0:
                    lines.add(line_index[g["line"]])
            if best is not None:
                lines.add(line_index[best["line"]])
        out.append(next(iter(lines)) if len(lines) == 1 else None)
    return out


def _judge(pairs, row_line, planted):
    """有意な行の組を、正解の行の組に戻して分類する。"""
    out, seen = [], set()
    for (a, b), p_adj in pairs:
        la, lb = row_line[a], row_line[b]
        if la is None or lb is None:
            kind, key = "unmapped", None
        elif la == lb:
            kind, key = "same_line", None
        else:
            key = (min(la, lb), max(la, lb))
            kind = "planted" if key in planted else "unplanted"
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        out.append({"gt": list(key) if key else None, "kind": kind, "p_adj": p_adj})
    return out


def judge_votes(votes, ref_line, planted):
    """票の一覧（基準の行の組）を正解の行の組に戻して分類する。

    同じ正解の組に戻る基準の行の組（行の破片など）は1件にまとめ、票の多い方を残す。正解に戻せない組は、基準の行の組ごとに1件。
    正解の行の組と、正解に戻せない基準の行の組は、番号が同じでも別物なので、キーに印を付けて区別する
    （印がないと、たとえば正解の組 (1, 2) と基準の行の組 (1, 2) が衝突して、戻せない方の報告が消える。Codex レビュー、docs/ISSUES.md #83）。
    """
    judged = {}
    for v in votes:
        j = _judge([(tuple(v["rows"]), None)], ref_line, planted)[0]
        key = ("gt", *j["gt"]) if j["gt"] else ("rows", *v["rows"])
        old = judged.get(key)
        if old is None or v["votes"] > old["votes"]:
            judged[key] = {"gt": j["gt"], "kind": j["kind"], "votes": v["votes"],
                           "raw_votes": v["raw_votes"], "groups": v["groups"], "members": v["members"]}
    return list(judged.values())


def _ngram_pairs(analysis):
    out = []
    for h in analysis.get("hypotheses", []):
        if h["family"] == "linepair":
            for p in h["key"]["pairs"]:
                out.append(((p[0], p[1]), h["statistics"]["p_family_corrected"], h["tier"], h["stability"]))
    return out


def _partition_ari(a, b):
    from babel.vision.evaluate import adjusted_rand
    return adjusted_rand(a, b)


def run_case(case):
    import cv2

    from babel.robust import ensemble, rowsim
    from babel.structure import corpus as sc
    from babel.structure.analyze import analyze
    from babel.vision.evaluate import evaluate
    t0 = time.time()
    rows, planted, detail, img, gt, vocab = render_case(case)
    planted = set(planted)
    rec = {**case, "text_seed": text_seed(case), "vocab_used": vocab, "planted": [list(p) for p in sorted(planted)], "planted_detail": detail["planted"],
           "tokens": sum(len(r) for r in rows), "types": len({s for r in rows for s in r}),
           "hapax": sum(1 for v in Counter(s for r in rows for s in r).values() if v == 1), "methods": {}, "vision": {}}
    ident = list(range(len(rows)))
    # 正解の転写（上限）
    res = rowsim.row_similarity(rows, cfg=dict(ROW_CFG, seed=case["seed"]))
    rec["methods"]["oracle-lcs"] = _judge([(tuple(p["rows"]), p["p_adj"]) for p in res["pairs"] if p["significant"]], ident, planted)
    a = analyze(sc.from_lines([[f"s{s}" for s in r] for r in rows]), dict(ANALYZE_CFG, seed=case["seed"]))
    rec["methods"]["oracle-ngram"] = _judge([(k, p) for k, p, _, _ in _ngram_pairs(a)], ident, planted)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "page.png"
        cv2.imwrite(str(path), img)
        regions = gt.get("regions")
        reports = ensemble.transcribe_all(path, regions)
        results, row_lines = {}, {}
        for name, rep in reports.items():
            results[name] = ensemble.member_pairs(rep, dict(ROW_CFG, seed=case["seed"]))
            row_lines[name] = _row_lines(rep, gt)
            rec["methods"][f"lcs:{name}"] = _judge([(tuple(p["rows"]), p["p_adj"]) for p in results[name]["pairs"] if p["significant"]],
                                                   row_lines[name], planted)
            ev = evaluate(rep, gt)
            cls = Counter(g["class_id"] for g in rep["glyphs"])
            le = ev.get("lines", {}).get("errors", {})
            rec["vision"][name] = {"f1": ev.get("detection", {}).get("f1"), "ari": ev.get("classes", {}).get("ari"),
                                   "line_f1": ev.get("lines", {}).get("f1"), "rows": len(rep["rows"]), "glyphs": len(rep["glyphs"]),
                                   "classes": len(cls), "hapax": sum(1 for v in cls.values() if v == 1),
                                   "extra_rows": le.get("oversplit_extra_rows"), "merged_rows": le.get("merged_rows")}
        for name in NGRAM_MEMBERS:
            try:
                a = analyze(sc.from_glyph_report(reports[name]), dict(ANALYZE_CFG, seed=case["seed"]))
                found = _ngram_pairs(a)
            except ValueError:
                found = []
            rec["methods"][f"ngram:{name}"] = _judge([(k, p) for k, p, _, _ in found], row_lines[name], planted)
            if name == "v3":      # 既存の安定性の指標が真偽を分けるかを見るための記録
                judged = _judge([(k, p) for k, p, _, _ in found], row_lines[name], planted)
                tiers = {}
                for (k, p, tier, stab) in found:
                    la, lb = row_lines[name][k[0]], row_lines[name][k[1]]
                    if la is not None and lb is not None and la != lb:
                        tiers[(min(la, lb), max(la, lb))] = (tier, stab)
                rec["ngram_v3_detail"] = [{**j, "tier": tiers.get(tuple(j["gt"]), (None, None))[0] if j["gt"] else None,
                                           "stability": tiers.get(tuple(j["gt"]), (None, None))[1] if j["gt"] else None} for j in judged]
        ref = reports[ensemble.REFERENCE]
        for tau in SOFT_TAUS:
            res = ensemble.soft_pairs(path, ref, regions, dict(ROW_CFG, seed=case["seed"]), tau=tau)
            rec["methods"][f"soft@{tau}"] = _judge([(tuple(p["rows"]), p["p_adj"]) for p in res["pairs"] if p["significant"]],
                                                   row_lines[ensemble.REFERENCE], planted)
        votes = ensemble.vote(reports, results)
        ref_line = row_lines[ensemble.REFERENCE]
        rec["votes"] = judge_votes(votes, ref_line, planted)
        # 同じ切り出しを使う方式どうしの、分類の一致（正解を使わない）
        same_seg = [m["name"] for m in ensemble.MEMBERS if m["base"] == "improved-v3" and not m.get("scale")]
        agree = []
        for i in range(len(same_seg)):
            for j in range(i + 1, len(same_seg)):
                x = [g["class_id"] for g in reports[same_seg[i]]["glyphs"]]
                y = [g["class_id"] for g in reports[same_seg[j]]["glyphs"]]
                if len(x) > 1:
                    agree.append(_partition_ari(x, y))
        rec["classifier_agreement_ari"] = round(sum(agree) / len(agree), 4) if agree else None
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def run(split, out, jobs=4, limit=None):
    cs = cases(split, limit)
    t0 = time.time()
    with ProcessPoolExecutor(jobs) as ex:
        runs = list(ex.map(run_case, cs, chunksize=4))
    import cv2
    res = {"schema": "babel-robust-bench-v4", "split": split, "config": {k: (list(v) if not isinstance(v, str) else v) for k, v in SPLITS[split].items()},
           "analyze_cfg": ANALYZE_CFG, "row_cfg": ROW_CFG, "opencv": cv2.__version__, "cases": len(runs), "seconds": round(time.time() - t0, 1),
           "runs": runs}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(SPLITS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    r = run(a.split, a.out, a.jobs, a.limit)
    print(f"{a.split}: {r['cases']} 件、{r['seconds']} 秒")
