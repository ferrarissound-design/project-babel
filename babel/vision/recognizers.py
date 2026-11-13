"""画像認識の方式（比較用）。どれも glyph-report 互換の出力（rows, glyphs）を返す。外部API不要。

部品
- 前処理: otsu（既存と同じ） / median（3×3 の中央値フィルタの後に otsu。点状の雑音を減らす）
- 切り出し:
    projection  既存の babel vision（行の投影 → 列の投影）
    components  連結成分。行は成分の縦の中心でまとめ、点などの小さな成分は近い記号に付け、
                横方向に少しでも重なる成分は1つの記号にまとめる（開発用データで選択）
                行のまとめ方は line_method で選べる（babel/vision/lines.py）。improved・improved-v2 は従来の "center"、
                improved-v3 は "hybrid"
- 分類:
    iou         既存と同じ（32×32 に引き伸ばした形の IoU と大きさの近さ、先着の代表と比較）
    template    縦横比を保って 32×32 に収め、ぼかした画像の正規化相互相関（平均連結の階層クラスタリング）
    improved = median + components + template（開発用データで選んだ改善候補）
    shape       縦横比・充填率・Hu モーメント・部品数・穴の数の特徴量（標準化ユークリッド距離、平均連結）
    combined    template の距離に縦横比・部品数の差を加えたもの（平均連結）

既存の babel vision（analyze_image）は "baseline" としてそのまま残す。
"""
from __future__ import annotations

import math
import time
from pathlib import Path

from .glyphs import _cluster, _dependencies, _feature, _mask, _without_rules, analyze_image

# ------------------------------------------------------------------------------------------------ 前処理

def binarize(image, method="otsu"):
    cv2, np = _dependencies()
    if method == "median":
        image = cv2.medianBlur(image, 3)
    return _mask(image)


# ------------------------------------------------------------------------------------------------ 切り出し

def _width_join(glyphs, H, alpha, beta):
    """隣り合う候補を、まとめた幅が字1つ分（alpha × 字の高さ H）以下で、隙間が beta × H 以下ならまとめる。
    離れた部品からなる字は、まとめても1字の幅に収まるが、隣り合う別の字を2つ合わせると1字の幅を超える、という仮定。"""
    out = []
    for g in glyphs:
        if out:
            p = out[-1]
            gap = g[0] - p[2]
            if gap <= beta * H and (max(p[2], g[2]) - min(p[0], g[0])) <= alpha * H:
                out[-1] = [min(p[0], g[0]), min(p[1], g[1]), max(p[2], g[2]), max(p[3], g[3])] + p[4:]
                continue
        out.append(list(g))
    return out


def segment_components(mask, remove_rules=True, min_area=3, small_ratio=0.35, join_gap_ratio=0.35, join_overlap=0.0,
                       join_width=None, join_width_gap=0.5, line_method="center", line_kw=None):
    """戻り値: rows = [[box, ...], ...]（行ごとに左から）

    line_method: 成分を行にまとめる方式（babel/vision/lines.py）。"center" は従来の処理（improved・improved-v2）で、
    この関数の中の元のコードをそのまま通る。それ以外は行のまとめ方だけを差し替え、行の中の部品の結合は同じ。"""
    cv2, np = _dependencies()
    m = _without_rules(mask) if remove_rules else mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i])
        if a >= min_area:
            comps.append([x, y, x + w, y + h, a])
    if not comps:
        return []
    hs = sorted(c[3] - c[1] for c in comps)
    med_h = hs[len(hs) // 2]
    big = [c for c in comps if (c[3] - c[1]) >= small_ratio * med_h]
    small = [c for c in comps if (c[3] - c[1]) < small_ratio * med_h]
    if line_method != "center":
        from .lines import group_rows
        rows = group_rows(comps, m, line_method, **(line_kw or {}))
    else:
        # 行: 大きな成分を縦の中心の順に並べ、前の行の中心から離れたら新しい行
        big.sort(key=lambda c: (c[1] + c[3]) / 2)
        rows = []
        for c in big:
            cy = (c[1] + c[3]) / 2
            if rows:
                r = rows[-1]
                rcy = sum((d[1] + d[3]) / 2 for d in r) / len(r)
                if abs(cy - rcy) <= 0.6 * med_h:
                    r.append(c)
                    continue
            rows.append([c])
        # 小さな成分（点など）: 縦に最も近い行の、横に最も近い成分の記号に付ける（近すぎなければ単独の記号）
        for c in small:
            cy = (c[1] + c[3]) / 2
            best = min(range(len(rows)), key=lambda k: min(abs(cy - (d[1] + d[3]) / 2) for d in rows[k])) if rows else None
            if best is None:
                rows.append([c])
                continue
            rows[best].append(c)
    out = []
    for r in rows:
        r.sort(key=lambda c: c[0])
        glyphs = []
        for c in r:
            if glyphs:
                g = glyphs[-1]
                ov = min(g[2], c[2]) - max(g[0], c[0])
                narrow = min(g[2] - g[0], c[2] - c[0])
                gap = c[0] - g[2]
                small_c = (c[3] - c[1]) < small_ratio * med_h or (g[3] - g[1]) < small_ratio * med_h
                if (ov > 0 and ov >= join_overlap * narrow) or (small_c and gap <= join_gap_ratio * med_h):
                    glyphs[-1] = [min(g[0], c[0]), min(g[1], c[1]), max(g[2], c[2]), max(g[3], c[3]), g[4] + c[4]]
                    continue
            glyphs.append(list(c))
        out.append([g[:4] for g in glyphs])
    if join_width:
        # 字の高さ: 候補の高さの上位半分の中央値（分割された破片は字より低いので除く）
        hs = sorted((b[3] - b[1] for r in out for b in r), reverse=True)
        H = hs[len(hs) // 4] if hs else med_h
        out = [[g[:4] for g in _width_join(r, H, join_width, join_width_gap)] for r in out]
    return out


def segment_projection(mask, remove_rules=True):
    """既存 babel vision の切り出し（行の投影→列の投影）を、領域1つ（画像全体）で使う。"""
    from .glyphs import _extract_regions
    h, w = mask.shape[:2]
    rows, shapes = _extract_regions(mask, [{"id": "page", "box": [0, 0, w, h]}], remove_rules=remove_rules)
    return [[shapes[i]["box"] for i in r["glyph_indices"]] for r in rows]


# ------------------------------------------------------------------------------------------------ 特徴量と分類

def _crop(mask, box):
    x1, y1, x2, y2 = box
    return mask[y1:y2, x1:x2]


def _aspect_fit(crop, size=32):
    cv2, np = _dependencies()
    h, w = crop.shape[:2]
    s = (size - 2) / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    r = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), np.float32)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = r
    out = cv2.GaussianBlur(out, (3, 3), 0)
    v = out - out.mean()
    nrm = float(np.sqrt((v * v).sum())) or 1.0
    return (v / nrm).ravel()


def _shape_vec(crop):
    cv2, np = _dependencies()
    b = (crop > 0).astype(np.uint8)
    h, w = b.shape
    hu = cv2.HuMoments(cv2.moments(b, binaryImage=True)).ravel()
    hu = [-math.copysign(1, v) * math.log10(abs(v)) if v else 0.0 for v in hu[:4]]
    n_comp = cv2.connectedComponents(b, connectivity=8)[0] - 1
    inv = (1 - b)
    inv_p = np.pad(inv, 1, constant_values=1)
    holes = cv2.connectedComponents(inv_p, connectivity=4)[0] - 2
    return [math.log(w / h), b.mean(), *hu, n_comp, max(0, holes)]


def _agglomerate(D, threshold):
    """平均連結の階層クラスタリング（距離行列 D、閾値で打ち切り）。戻り値: クラス番号のリスト。"""
    _, np = _dependencies()
    n = D.shape[0]
    if n == 0:
        return []
    D = D.astype(np.float64).copy()
    np.fill_diagonal(D, np.inf)
    size = np.ones(n)
    alive = np.ones(n, bool)
    members = {i: [i] for i in range(n)}
    while alive.sum() > 1:
        idx = np.argmin(D)
        i, j = divmod(int(idx), n)
        if D[i, j] > threshold:
            break
        # j を i にまとめる（Lance–Williams 更新）
        D[i, :] = (size[i] * D[i, :] + size[j] * D[j, :]) / (size[i] + size[j])
        D[:, i] = D[i, :]
        D[i, i] = np.inf
        D[j, :] = np.inf
        D[:, j] = np.inf
        size[i] += size[j]
        alive[j] = False
        members[i] += members.pop(j)
    labels = [0] * n
    for k, (_, mem) in enumerate(sorted(members.items())):
        for m in mem:
            labels[m] = k
    return labels


def classify(mask, boxes, method="iou", threshold=None):
    _, np = _dependencies()
    crops = [_crop(mask, b) for b in boxes]
    if method == "iou":
        shapes = [{"feature": _feature(c), "width": b[2] - b[0], "height": b[3] - b[1]} for c, b in zip(crops, boxes)]
        _cluster(shapes, threshold or 0.7)
        return [s["class_id"] for s in shapes]
    if method in ("template", "combined"):
        V = np.array([_aspect_fit(c) for c in crops])
        D = 1.0 - V @ V.T                                    # 1 - 正規化相互相関
        if method == "combined":
            A = np.array([math.log((b[2] - b[0]) / (b[3] - b[1])) for b in boxes])
            K = np.array([_shape_vec(c)[6] for c in crops])
            D = D + 0.5 * np.abs(A[:, None] - A[None, :]) + 0.15 * (K[:, None] != K[None, :])
        return _agglomerate(D, threshold if threshold is not None else DEV_THRESHOLDS[method])
    if method == "shape":
        F = np.array([_shape_vec(c) for c in crops], dtype=np.float64)
        sd = F.std(axis=0)
        sd[sd == 0] = 1.0
        Z = (F - F.mean(axis=0)) / sd
        D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)) / math.sqrt(Z.shape[1])
        return _agglomerate(D, threshold if threshold is not None else DEV_THRESHOLDS["shape"])
    raise ValueError(f"分類の方式 {method}")


# ------------------------------------------------------------------------------------------------ 本体

# 分類の閾値は開発用の人工画像（シード 0〜9）だけで選んだ（平均 ARI が最大の値）。val・test では変えない
DEV_THRESHOLDS = {"template": 0.20, "combined": 0.25, "shape": 0.15}

METHODS = {
    "improved":          {"preprocess": "median", "segment": "components", "classify": "template"},
    # improved-v2: improved に、まとめた幅が字1つ分に収まる隣の部品の結合を加えたもの（離れた部品を持つ字の分割を減らす）。
    # join_width・join_width_gap は dev だけで選び（0.95, 0.5）、val で確かめた。test では変えない（docs/ISSUES.md #68）
    "improved-v2":       {"preprocess": "median", "segment": "components", "classify": "template",
                          "seg": {"join_width": 0.95, "join_width_gap": 0.5}},
    # improved-v3: improved-v2 の行のまとめ方だけを替えたもの（babel/vision/lines.py の hybrid）。前処理・部品の結合・分類は同じ。
    # 行の方式と閾値は、行の作りを変えた人工画像（babel/vision_eval/layout.py）の dev で選び、val で確かめた。
    # holdout（別の字の作り・別の劣化）は最終評価に1回だけ使った（docs/VISION_EVAL_V3.md、docs/ISSUES.md #72）
    "improved-v3":       {"preprocess": "median", "segment": "components", "classify": "template",
                          "seg": {"join_width": 0.95, "join_width_gap": 0.5, "line_method": "hybrid",
                                  "line_kw": {"small": 0.5, "depth": 0.3, "join_gap": 0.4}}},
    "baseline":          {"preprocess": "otsu",   "segment": "projection", "classify": "iou"},
    "proj+combined":     {"preprocess": "otsu",   "segment": "projection", "classify": "combined"},
    "cc+iou":            {"preprocess": "otsu",   "segment": "components", "classify": "iou"},
    "cc+template":       {"preprocess": "otsu",   "segment": "components", "classify": "template"},
    "cc+shape":          {"preprocess": "otsu",   "segment": "components", "classify": "shape"},
    "cc+combined":       {"preprocess": "otsu",   "segment": "components", "classify": "combined"},
    "median+cc+combined": {"preprocess": "median", "segment": "components", "classify": "combined"},
}


def recognize(path, method="baseline", threshold=None, regions=None, **seg_kwargs):
    """画像のパスだけを受け取り、glyph-report 互換の dict を返す。

    regions: [{"id": "left", "box": [x1, y1, x2, y2]}]（右下は含まない）。指定すると、領域ごとに切り出し
    （罫線の除去も領域ごと）、記号の座標は画像全体の座標で返す。分類は全領域の記号をまとめて行う。
    """
    from .glyphs import _rectangles
    cv2, np = _dependencies()
    t0 = time.time()
    if method == "baseline":
        rep = analyze_image(path, regions=regions)
        rep["settings"]["method"] = "baseline"
        rep["seconds"] = round(time.time() - t0, 4)
        return rep
    cfg = METHODS[method]
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"画像を読めない: {path}")
    mask = binarize(image, cfg["preprocess"])
    selected = _rectangles(image.shape, regions)
    clean = np.zeros_like(mask)
    region_rows = []                                   # [(領域 ID, [記号の枠, ...]), ...]
    for reg in selected:
        x1, y1, x2, y2 = reg["box"]
        m = mask[y1:y2, x1:x2]
        clean[y1:y2, x1:x2] = _without_rules(m)
        kw = {**cfg.get("seg", {}), **seg_kwargs}
        rb = segment_components(m, **kw) if cfg["segment"] == "components" else segment_projection(m)
        for r in rb:
            region_rows.append((reg["id"], [[b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1] for b in r]))
    boxes = [b for _, r in region_rows for b in r]
    labels = classify(clean, boxes, cfg["classify"], threshold) if boxes else []
    glyphs, rows, k = [], [], 0
    for rid, r in region_rows:
        idx = []
        for b in r:
            glyphs.append({"box": [int(v) for v in b], "class_id": int(labels[k]), "match_iou": None})
            idx.append(len(glyphs) - 1)
            k += 1
        if idx:
            rb = [min(glyphs[i]["box"][0] for i in idx), min(glyphs[i]["box"][1] for i in idx),
                  max(glyphs[i]["box"][2] for i in idx), max(glyphs[i]["box"][3] for i in idx)]
            rows.append({"region": rid, "box": rb, "classes": [glyphs[i]["class_id"] for i in idx], "glyph_indices": idx})
    return {"schema": "babel-glyph-pilot-v1", "source_image": Path(path).name,
            "image_size": [int(image.shape[1]), int(image.shape[0])], "regions": selected, "rows": rows, "glyphs": glyphs,
            "settings": {"method": method, **cfg, "threshold": threshold, "seg_overrides": seg_kwargs},
            "stats": {"rows": len(rows), "glyph_candidates": len(glyphs), "visual_classes": len(set(labels))},
            "seconds": round(time.time() - t0, 4)}
