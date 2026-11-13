"""画像認識の評価: 認識器の出力（glyph-report）と正解（babel-glyph-gt-v1）を比べる。

- 検出: IoU ≥ 0.5 の組を IoU の高い順に1対1で対応付ける（1つの正解に複数の予測を対応させて検出率を水増ししない）
- 分割の誤り: 1つの正解の記号の中に、面積の半分以上が入る予測が2つ以上ある
- 結合の誤り: 1つの予測が、面積の半分以上を覆う正解の記号を2つ以上含む
- クラス: クラス番号は任意なので、番号の一致ではなく分類の構造を比べる（ARI・NMI・過剰分割・誤統合・純度）
- 行: 行の枠を縦方向の IoU で1対1に対応付ける。改行が文の区切りでない行（continues_sentence）の数も数える
- 行の分け方の誤り: 予測の各記号を、最も重なる正解の記号の行（と、面積の半分以上を覆う正解の記号の行）に対応させ、1つの正解の行が複数の予測の行に分かれた数
  （過剰分割）と、1つの予測の行が複数の正解の行にまたがった数（誤結合）を数える。行の枠の一致（F1）とは別に測る
- 位置: 対応した組の IoU・中心のずれ
- 正解のうち人が確認していない項目（verified=false）は使わない。使える正解がなければ「測定不能」
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from .groundtruth import aliases, canonical_label, evaluation_view

UNMEASURABLE = "測定不能"


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u else 0.0


def _inter(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _area(a):
    return (a[2] - a[0]) * (a[3] - a[1])


def _on_non_glyph(box, item):
    """予測が文字以外の要素の上にあるか。枠のような中空の要素（outline=true）は、内部ではなく縁の帯だけを見る
    （内部全体を文字以外とみなすと、枠の中の文字の切り出しの誤りまで装飾の誤検出に数えてしまう: Codex レビュー、PR #10）。"""
    b = item["box"]
    if not item.get("outline"):
        return _inter(box, b) >= 0.5 * _area(box)
    w = max(2, round(0.02 * min(b[2] - b[0], b[3] - b[1])))
    bands = [[b[0], b[1], b[2], b[1] + w], [b[0], b[3] - w, b[2], b[3]], [b[0], b[1], b[0] + w, b[3]], [b[2] - w, b[1], b[2], b[3]]]
    return sum(_inter(box, band) for band in bands) >= 0.5 * _area(box)


def match(gt_boxes, pred_boxes, thr=0.5):
    """IoU の高い順に1対1で対応付ける。戻り値: [(gt_index, pred_index, iou)]"""
    cand = []
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            if g[2] <= p[0] or p[2] <= g[0] or g[3] <= p[1] or p[3] <= g[1]:
                continue
            v = iou(g, p)
            if v >= thr:
                cand.append((v, i, j))
    cand.sort(reverse=True)
    used_g, used_p, out = set(), set(), []
    for v, i, j in cand:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        out.append((i, j, v))
    return out


# ------------------------------------------------------------------------------------------------ クラスタリングの指標

def _comb2(n):
    return n * (n - 1) / 2


def adjusted_rand(a, b):
    n = len(a)
    if n < 2:
        return None
    cont = Counter(zip(a, b))
    sa, sb = Counter(a), Counter(b)
    index = sum(_comb2(v) for v in cont.values())
    ea, eb = sum(_comb2(v) for v in sa.values()), sum(_comb2(v) for v in sb.values())
    expected = ea * eb / _comb2(n)
    mx = (ea + eb) / 2
    if mx == expected:
        return 1.0 if index == expected else 0.0
    return (index - expected) / (mx - expected)


def _entropy(c, n):
    return -sum(v / n * math.log(v / n) for v in c.values() if v)


def nmi(a, b):
    """正規化相互情報量（算術平均で正規化）。"""
    n = len(a)
    if n == 0:
        return None
    cont = Counter(zip(a, b))
    sa, sb = Counter(a), Counter(b)
    mi = sum(v / n * math.log(v * n / (sa[x] * sb[y])) for (x, y), v in cont.items())
    ha, hb = _entropy(sa, n), _entropy(sb, n)
    if ha + hb == 0:
        return 1.0
    return 2 * mi / (ha + hb)


def class_structure(true, pred):
    """true: 正解の字形名、pred: 予測のクラス番号（同じ長さ、対応した記号だけ）。"""
    if not true:
        return {"status": UNMEASURABLE, "reason": "クラスを比べられる記号（対応し、人が label を付けた正解）がない"}
    by_true, by_pred = defaultdict(set), defaultdict(set)
    for t, p in zip(true, pred):
        by_true[t].add(p)
        by_pred[p].add(t)
    over = {t: len(ps) for t, ps in by_true.items() if len(ps) > 1}
    mis = {p: len(ts) for p, ts in by_pred.items() if len(ts) > 1}
    cont = Counter(zip(true, pred))
    purity = sum(max(v for (t, p2), v in cont.items() if p2 == p) for p in by_pred) / len(true)
    inv = sum(max(v for (t2, p), v in cont.items() if t2 == t) for t in by_true) / len(true)
    return {"n": len(true), "true_classes": len(by_true), "predicted_classes": len(by_pred),
            "ari": round(adjusted_rand(true, pred), 4) if len(true) > 1 else None, "nmi": round(nmi(true, pred), 4),
            "purity": round(purity, 4), "inverse_purity": round(inv, 4),
            "oversplit_true_classes": len(over), "oversplit_extra_clusters": sum(k - 1 for k in over.values()),
            "mismerged_predicted_classes": len(mis), "mismerged_extra_labels": sum(k - 1 for k in mis.values()),
            "class_ratio_predicted_per_true": round(len(by_pred) / len(by_true), 3)}


# ------------------------------------------------------------------------------------------------ 行の分け方

def line_errors(report: dict, gt: dict) -> dict:
    """行の過剰分割と誤結合（確認済みの行と、その行に属する確認済みの記号だけを使う）。

    予測の記号は、枠が最も大きく重なる正解の記号の行に属するとみなす（どの正解の記号にも重ならない予測は数えない）。
    加えて、予測の記号が正解の記号の面積の半分以上を覆うなら、その記号の行にも属するとみなす。行をまとめた後に
    上下の行の字が1つの記号に結合されると、予測の枠が複数の正解の行を覆う。最も重なる1つだけに対応させると、
    その誤結合が数えられない（Codex レビュー、PR #12）。
    行の枠の IoU による対応（行 F1）と違い、行の高さが少し違うだけでは誤りにならず、行のまとめ方だけを測る。
    """
    vlines = [ln["id"] for ln in gt["lines"] if ln["verified"]]
    G = [it for it in gt["items"] if it["kind"] == "glyph" and it["verified"] and it["line"] in set(vlines)]
    P = _pred_glyphs(report)
    rows_of = defaultdict(set)                      # 正解の行 → その行の記号を含む予測の行
    lines_of = defaultdict(set)                     # 予測の行 → 含まれる正解の行
    for p in P:
        if p["row"] is None:
            continue
        best, area, covered = None, 0, set()
        for g in G:
            a = _inter(p["box"], g["box"])
            if a > area:
                best, area = g, a
            if a > 0 and a >= 0.5 * _area(g["box"]):
                covered.add(g["line"])
        if best is not None:
            for ln in covered | {best["line"]}:
                rows_of[ln].add(p["row"])
                lines_of[p["row"]].add(ln)
    return {"oversplit_lines": sum(1 for v in rows_of.values() if len(v) > 1),
            "oversplit_extra_rows": sum(len(v) - 1 for v in rows_of.values()),
            "merged_rows": sum(1 for v in lines_of.values() if len(v) > 1),
            "merged_extra_lines": sum(len(v) - 1 for v in lines_of.values()),
            "rows_without_gt_glyph": len(report.get("rows") or []) - len(lines_of),
            "lines_without_prediction": len(vlines) - len(rows_of)}


# ------------------------------------------------------------------------------------------------ 本体

def _pred_glyphs(report):
    glyphs = report.get("glyphs") or []
    row_of = {}
    offset = 0
    for r, row in enumerate(report.get("rows") or []):
        idx = row.get("glyph_indices") or list(range(offset, offset + len(row.get("classes") or [])))
        offset += len(row.get("classes") or [])
        for gi in idx:
            row_of[gi] = r
    return [{"box": g["box"], "class": g.get("class_id"), "row": row_of.get(i)} for i, g in enumerate(glyphs)]


def evaluate(report: dict, gt: dict, iou_thr=0.5) -> dict:
    view = evaluation_view(gt)
    al = view["aliases"]
    G = [it for it in gt["items"] if it["kind"] == "glyph" and it["verified"]]
    N = [it for it in gt["items"] if it["kind"] != "glyph" and it["verified"]]
    unverified = sum(1 for it in gt["items"] if not it["verified"])
    P = _pred_glyphs(report)
    out = {"gt_verified_glyphs": len(G), "gt_unverified_items": unverified, "predicted_glyphs": len(P),
           "iou_threshold": iou_thr, "evaluation_scope": view["scope"]}
    if not G:
        out["status"] = UNMEASURABLE
        out["reason"] = "人が確認した正解の記号がない（verified=true の glyph が0）"
        return out
    gb, pb = [g["box"] for g in G], [p["box"] for p in P]
    pairs = match(gb, pb, iou_thr)
    tp = len(pairs)
    matched_p = {j for _, j, _ in pairs}
    # 未確認の正解の位置にある予測は、正しいか判断できないので偽陽性にも precision の分母にも入れない
    # （確認の進み具合で精度が変わらないように。Codex レビュー、PR #10）
    U = [it for it in gt["items"] if not it["verified"]]
    um = {j for _, j, _ in match([u["box"] for u in U], [pb[j] for j in range(len(P))], iou_thr)} if U else set()
    ignored = {j for j in range(len(P)) if j not in matched_p and
               (j in um or any(_on_non_glyph(pb[j], u) for u in U))}      # 中空の要素（outline）は縁だけ
    # 文字以外（罫線・装飾・雑音）を記号として検出したもの
    nonglyph_hits = sum(1 for j, p in enumerate(P) if j not in matched_p and j not in ignored and
                        any(_on_non_glyph(p["box"], n) for n in N))
    n_pred = len(P) - len(ignored)
    prec, rec = tp / n_pred if n_pred else 0.0, tp / len(G)
    out["detection"] = {"tp": tp, "fp": n_pred - tp, "fn": len(G) - tp, "precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
                        "non_glyph_detected_as_glyph": nonglyph_hits,
                        "ignored_predictions_on_unverified": len(ignored)}
    # 分割・結合
    splits = merges = fragments = 0
    for g in gb:
        inside = [p for p in pb if _inter(p, g) >= 0.5 * _area(p)]
        if len(inside) >= 2:
            splits += 1
            fragments += len(inside)
    for p in pb:
        covered = [g for g in gb if _inter(p, g) >= 0.5 * _area(g)]
        if len(covered) >= 2:
            merges += 1
    out["segmentation_errors"] = {"split_glyphs": splits, "split_fragments": fragments, "merged_predictions": merges}
    # クラス
    true, pred = [], []
    for i, j, _ in pairs:
        lab = canonical_label(G[i], al)
        if lab and not G[i]["uncertain"]:
            true.append(lab)
            pred.append(P[j]["class"])
    out["classes"] = class_structure(true, pred)
    # 位置
    if pairs:
        cen = [math.hypot((gb[i][0] + gb[i][2] - pb[j][0] - pb[j][2]) / 2, (gb[i][1] + gb[i][3] - pb[j][1] - pb[j][3]) / 2)
               for i, j, _ in pairs]
        out["localization"] = {"mean_iou": round(sum(v for _, _, v in pairs) / tp, 4),
                               "mean_center_offset_px": round(sum(cen) / tp, 3), "max_center_offset_px": round(max(cen), 3)}
    # 行: 予測の行を、確認済みの行と未確認の行の両方に1対1で対応付ける。未確認の行に対応した予測は
    # 正しいか判断できないので precision の分母に入れない（どの行とも対応しない余分な予測は偽陽性のまま）。
    # 行割り当ての精度は、確認済みの行に属する記号だけで計算する（Codex レビュー、PR #10）
    L = [ln for ln in gt["lines"] if ln["verified"]]
    UL = [ln for ln in gt["lines"] if not ln["verified"]]
    rows = report.get("rows") or []
    if L:
        def yiou(a, b):
            inter = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            u = (a[3] - a[1]) + (b[3] - b[1]) - inter
            return inter / u if u else 0.0
        allL = L + UL
        cand = sorted(((yiou(ln["box"], r["box"]), i, j) for i, ln in enumerate(allL) for j, r in enumerate(rows)
                       if min(ln["box"][2], r["box"][2]) > max(ln["box"][0], r["box"][0])), reverse=True)
        used_l, used_r, lm, ignored_rows = set(), set(), {}, set()
        for v, i, j in cand:
            if v >= 0.5 and i not in used_l and j not in used_r:
                used_l.add(i)
                used_r.add(j)
                if i < len(L):
                    lm[j] = L[i]["id"]
                else:
                    ignored_rows.add(j)
        n_rows = len(rows) - len(ignored_rows)
        verified_ids = {ln["id"] for ln in L}
        assign = [lm.get(P[j]["row"]) == G[i]["line"] for i, j, _ in pairs if G[i]["line"] in verified_ids]
        res = {"gt_lines": len(L), "unverified_gt_lines": len(UL), "predicted_rows": len(rows),
               "ignored_rows_on_unverified_lines": len(ignored_rows), "matched": len(lm),
               "glyph_line_assignment_accuracy": round(sum(assign) / len(assign), 4) if assign else None,
               "line_breaks_not_sentence_ends": sum(1 for ln in L if ln["continues_sentence"]),
               "note": "babel vision は1行を1文として出力するので、continues_sentence の行はすべて文の区切りの誤認になる"}
        lr = len(lm) / len(L)
        # n_rows == 0 なら確認済みの行はどれも検出されていない（recall 0）。F1 ≤ 2R/(1+R) なので precision によらず F1 は0。
        # 「測定不能」にすると集計（_agg）の分母から消え、行の検出に失敗した画像が平均に入らない（Codex レビュー、PR #11）
        if not rows:
            res.update(precision=0.0, recall=round(lr, 4), f1=0.0, reason="予測の行が1つもない")
        elif n_rows == 0:
            res.update(recall=round(lr, 4), precision=UNMEASURABLE, f1=0.0,
                       reason="確認済みの行に対応しうる予測の行がない（予測の行はすべて未確認の行に対応した）。"
                              "precision は測定不能、確認済みの行の再現率が0なので F1 は0")
        else:
            lp = len(lm) / n_rows
            res.update(precision=round(lp, 4), recall=round(lr, 4), f1=round(2 * lp * lr / (lp + lr), 4) if lp + lr else 0.0)
        res["errors"] = line_errors(report, gt)
        out["lines"] = res
    else:
        out["lines"] = {"status": UNMEASURABLE, "reason": "人が確認した行がない"}
    out["status"] = "measured"
    return out


def transcription_from(report: dict, gt: dict, iou_thr=0.5):
    """評価専用: 予測の各記号に、対応した正解の字形名を付ける（構造解析の結果を正解の記号で採点するため）。"""
    al = aliases(gt)
    G = [it for it in gt["items"] if it["kind"] == "glyph" and it["verified"]]
    P = _pred_glyphs(report)
    pairs = match([g["box"] for g in G], [p["box"] for p in P], iou_thr)
    return {j: canonical_label(G[i], al) for i, j, _ in pairs}
