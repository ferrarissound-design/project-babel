"""1枚の画像を複数の認識方式で転写し、行の組の仮説が方式をまたいで再現するかを数える。

考え方: 転写は画像についての1つの推定で、方式を変えると変わる。ある転写でだけ有意になる規則は、その転写の誤りが作った
可能性がある。画像に本当にある規則なら、切り出し・分類・閾値・画像の大きさを変えても残るはず（残らなければ、少なくとも
「認識に依存する」と分かる）。逆は成り立たない: どの方式にも共通する誤り（同じ二値化、同じ連結成分）が作る規則は、
方式をまたいでも残る。その限界は docs/ROBUST_STRUCTURE.md に書いた。

仮説の対応付けはクラス番号を使わない（番号は転写ごとに別）。行の枠の上下の重なりで、各方式の行を基準の方式の行に対応させる。

既存の認識方式（babel/vision/recognizers.py の METHODS）は変えていない。ここでは同じ部品を別の組み合わせで呼ぶだけ。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from ..vision.glyphs import _dependencies, _rectangles, _without_rules
from ..vision.recognizers import _aspect_fit, _crop, binarize, classify, recognize
from . import rowsim

REFERENCE = "v3"

# 方式の一覧。base: 切り出しに使う既存の方式。classify / threshold: 記号の分類だけを差し替える。scale: 画像を拡大してから認識する。
# 選び方: 切り出し（投影 / 連結成分・行のまとめ方2種）、分類（template の閾値3種 / shape / iou）、画像の大きさを1つずつ変えた。
# 構成は dev の結果を見る前に決め、その後は変えていない（docs/ROBUST_STRUCTURE.md 第3節）。
MEMBERS = [
    {"name": "v3", "base": "improved-v3"},
    {"name": "v3-fine", "base": "improved-v3", "classify": "template", "threshold": 0.12},
    {"name": "v3-coarse", "base": "improved-v3", "classify": "template", "threshold": 0.30},
    {"name": "v3-shape", "base": "improved-v3", "classify": "shape"},
    {"name": "v3-iou", "base": "improved-v3", "classify": "iou"},
    {"name": "v3-scaled", "base": "improved-v3", "scale": 1.3},
    {"name": "v2", "base": "improved-v2"},
    {"name": "baseline", "base": "baseline"},
]

# 字形の類似度の重み: max(0, 1 - 距離 / tau)。dev では tau = 0.2・0.3・0.45 のどれも、クラス番号の一致による検定より良くならず、
# 選択の基準（仕込みのない画像での誤発見が 0.05 以下）を満たさなかった。判定には使わず、参考として出すだけ（docs/ROBUST_STRUCTURE.md）
SOFT_TAU = 0.3


def _clean_mask(path, regions, preprocess="median"):
    cv2, np = _dependencies()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"画像を読めない: {path}")
    mask = binarize(image, preprocess)
    clean = np.zeros_like(mask)
    for reg in _rectangles(image.shape, regions):
        x1, y1, x2, y2 = reg["box"]
        clean[y1:y2, x1:x2] = _without_rules(mask[y1:y2, x1:x2])
    return clean


def reclassify(report, clean, method, threshold=None):
    """切り出し（記号の枠と行）はそのままで、分類だけをやり直した glyph-report を返す。"""
    boxes = [g["box"] for g in report["glyphs"]]
    labels = classify(clean, boxes, method, threshold) if boxes else []
    glyphs = [{**g, "class_id": int(c)} for g, c in zip(report["glyphs"], labels)]
    rows = [{**r, "classes": [glyphs[i]["class_id"] for i in r["glyph_indices"]]} for r in report["rows"]]
    return {**report, "glyphs": glyphs, "rows": rows,
            "settings": {**report.get("settings", {}), "reclassified": {"classify": method, "threshold": threshold}},
            "stats": {**report.get("stats", {}), "visual_classes": len(set(labels))}}


def _scaled(path, regions, method, scale):
    """画像を scale 倍してから認識し、枠を元の座標に戻す。"""
    cv2, _ = _dependencies()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    big = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    regs = None
    if regions:
        regs = [{**r, "box": [int(round(v * scale)) for v in r["box"]]} for r in regions]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "scaled.png"
        cv2.imwrite(str(p), big)
        rep = recognize(p, method, regions=regs)

    def back(b):
        return [int(b[0] / scale), int(b[1] / scale), max(int(b[0] / scale) + 1, int(round(b[2] / scale))),
                max(int(b[1] / scale) + 1, int(round(b[3] / scale)))]
    glyphs = [{**g, "box": back(g["box"])} for g in rep["glyphs"]]
    rows = [{**r, "box": back(r["box"])} for r in rep["rows"]]
    return {**rep, "glyphs": glyphs, "rows": rows, "source_image": Path(path).name,
            "image_size": [int(image.shape[1]), int(image.shape[0])], "regions": _rectangles(image.shape, regions),
            "settings": {**rep["settings"], "scale": scale}}


def transcribe_all(path, regions=None, members=None):
    """方式の名前 → glyph-report。同じ切り出しを使う方式は、切り出しを1回だけ行う。"""
    members = members or MEMBERS
    base_cache, clean, out = {}, None, {}
    for m in members:
        if m.get("scale"):
            out[m["name"]] = _scaled(path, regions, m["base"], m["scale"])
            continue
        if m["base"] not in base_cache:
            base_cache[m["base"]] = recognize(path, m["base"], regions=regions)
        rep = base_cache[m["base"]]
        if m.get("classify"):
            if clean is None:
                clean = _clean_mask(path, regions)
            rep = reclassify(rep, clean, m["classify"], m.get("threshold"))
        out[m["name"]] = rep
    return out


def soft_weights(path, report, regions=None, tau=SOFT_TAU):
    """記号の出現どうしの字形の類似度（0〜1）の行列。クラスへの分類を経ない。

    距離は improved の分類と同じもの（縦横比を保って 32×32 に収め、ぼかした形の 1 − 正規化相互相関）。
    分類は距離を閾値で切ってまとめるので、閾値の近くの組は「同じ」と「違う」のどちらかに倒れる。重みはそこを連続にする。
    """
    _, np = _dependencies()
    clean = _clean_mask(path, regions)
    boxes = [g["box"] for g in report["glyphs"]]
    if not boxes:
        return np.zeros((0, 0))
    V = np.array([_aspect_fit(_crop(clean, b)) for b in boxes])
    D = 1.0 - V @ V.T
    W = np.clip(1.0 - D / tau, 0.0, 1.0)
    np.fill_diagonal(W, 1.0)
    return W


def _y_overlap(a, b):
    return max(0, min(a[3], b[3]) - max(a[1], b[1]))


def map_rows(ref, rep, min_overlap=0.5):
    """rep の各行 → ref の行の番号（対応がなければ None）。行の枠の上下の重なりが rep の行の高さの min_overlap 以上で最大のもの。"""
    out = []
    for r in rep["rows"]:
        best, best_v = None, 0.0
        h = max(1, r["box"][3] - r["box"][1])
        for k, q in enumerate(ref["rows"]):
            if r.get("region") != q.get("region"):
                continue
            if min(r["box"][2], q["box"][2]) <= max(r["box"][0], q["box"][0]):
                continue
            v = _y_overlap(r["box"], q["box"]) / h
            if v > best_v:
                best, best_v = k, v
        out.append(best if best_v >= min_overlap else None)
    return out


def class_rows(report):
    return [list(r["classes"]) for r in report["rows"] if r["classes"]], [k for k, r in enumerate(report["rows"]) if r["classes"]]


def member_pairs(report, cfg=None):
    """1つの転写での行の組の検定。行の番号は report["rows"] の番号。"""
    rows, index = class_rows(report)
    res = rowsim.row_similarity(rows, cfg=cfg)
    for p in res["pairs"]:
        p["rows"] = [index[p["rows"][0]], index[p["rows"][1]]]
    return res


def soft_pairs(path, report, regions=None, cfg=None, tau=SOFT_TAU):
    """分類を経ない検定: 記号の出現を項目にし、字形の類似度を重みにする。"""
    W = soft_weights(path, report, regions, tau)
    rows = [list(r["glyph_indices"]) for r in report["rows"] if r["classes"]]
    index = [k for k, r in enumerate(report["rows"]) if r["classes"]]
    res = rowsim.row_similarity(rows, weight_matrix=W, cfg=cfg)
    for p in res["pairs"]:
        p["rows"] = [index[p["rows"][0]], index[p["rows"][1]]]
    res["statistic"] = "soft-lcs"
    res["tau"] = tau
    return res


def vote(reports, results, reference=REFERENCE):
    """方式ごとの有意な行の組を、基準の方式の行の組に対応させて数える。

    A reclassification of the same segmentation is one evidence group.  Keep
    raw_votes for diagnostics, but thresholds use distinct segmentation bases.
    """
    ref = reports[reference]
    tally = {}
    for name, res in results.items():
        mp = map_rows(ref, reports[name])
        seen = set()
        for p in res["pairs"]:
            if not p["significant"]:
                continue
            a, b = mp[p["rows"][0]], mp[p["rows"][1]]
            if a is None or b is None or a == b:
                continue
            seen.add((min(a, b), max(a, b)))
        for k in seen:
            tally.setdefault(k, []).append(name)
    bases = {m["name"]: m["base"] for m in MEMBERS}
    groups = {bases.get(name, name) for name in results}
    n = len(groups)
    out = [{"rows": list(k), "votes": len({bases.get(name, name) for name in v}),
            "raw_votes": len(v), "groups": sorted({bases.get(name, name) for name in v}),
            "members": sorted(v), "fraction": round(len({bases.get(name, name) for name in v}) / n, 4)}
           for k, v in tally.items()]
    out.sort(key=lambda r: (-r["votes"], r["rows"]))
    return out


def analyze_image(path, regions=None, cfg=None, members=None, with_soft=True):
    """画像1枚の解析。戻り値は JSON にできる dict（転写そのものは "transcriptions" に行のクラス番号だけを入れる）。"""
    reports = transcribe_all(path, regions, members)
    results = {name: member_pairs(rep, cfg) for name, rep in reports.items()}
    votes = vote(reports, results)
    out = {"schema": "babel-robust-structure-v1", "image": Path(path).name, "reference": REFERENCE,
           "members": [m["name"] for m in (members or MEMBERS)],
           "transcriptions": {n: {"rows": [{"region": r.get("region"), "box": r["box"], "classes": r["classes"]} for r in rep["rows"]],
                                  "stats": rep.get("stats"), "row_map": map_rows(reports[REFERENCE], rep)} for n, rep in reports.items()},
           "tests": results, "votes": votes}
    if with_soft:
        out["soft"] = soft_pairs(path, reports[REFERENCE], regions, cfg)
    return out
