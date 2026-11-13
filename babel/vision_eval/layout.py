"""行認識の評価用の人工画像（レイアウトと字の作りを変える）。正解は babel-glyph-gt-v1（source=synthetic）。

babel/vision_eval/synth.py（v1・v2 の評価）は、行の間隔が一定で、字の高さがほぼそろった画像しか作らない。
ここでは「同じ行の部品を同じ行にまとめられるか」を試すために、行の作りを変える。

字の作り（family）
- font     Hershey フォントの英字（下に伸びる字 g・p・q・y を含む）
- stroke   synth.py と同じ、線・弧・点の字形
- stack    上の部品と下の部品が離れている字形（字ごとの隙間は画像の中で一定）。全部の字が上下に分かれる画像もある
- radical  （holdout 専用）箱・横棒の重なり・縦棒・十字などの部品を、左右・上下・囲みで組み合わせた字形
- curve    （holdout 専用）点を滑らかにつないだ曲線の本体に、上や下の点・短い線を付けた字形（アンチエイリアスで描く）

行の作り（condition）
- plain       行の間隔・字の高さが一定（対照）
- varheight   字ごとに高さが違う（字形ごとに 0.5〜1.3 倍、下端をそろえる）
- diacritic   一部の字形の上または下に、小さな記号（点・短い線・山形）が付く（行の間隔は字の高さの 0.9 倍以上）
- jitter      字ごとに上下にずれる（字の高さの ±0.2 倍）
- tight       下に伸びる字・上に伸びる字があり、隣り合う行の上下の範囲が重なる
- uneven      行の間隔が不均一（行の間の空きが字の高さの 0.45〜2.5 倍）
- rule_touch  横の罫線が字の下端に接し、左に縦の罫線がある
- two_column  2つの文字領域（左右）。行の間隔と上端が違う。領域の枠は正解の regions に入れ、認識器に領域として渡す
- mixed       varheight + diacritic + 小さな jitter + uneven
- photo       （holdout 専用）mixed に、ぼかし・傾き・明るさのむら・雑音・JPEG 圧縮を加える（dev・val にない劣化）

分割: dev（シード 200〜207）で方式と閾値を選び、val（300〜307）で確かめ、holdout（7000〜7009）は最終評価に1回だけ使う。
holdout は字の作り（radical・curve）も劣化（photo）も dev・val にないもの。行数は画像ごとに 5〜14 で変える。
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from . import synth

SPLITS = {"dev": list(range(200, 208)), "val": list(range(300, 308)), "holdout": list(range(7000, 7010))}
FAMILIES = {"dev": ("font", "stroke", "stack"), "val": ("font", "stroke", "stack"), "holdout": ("radical", "curve")}
CONDITIONS = ("plain", "varheight", "diacritic", "jitter", "tight", "uneven", "rule_touch", "two_column", "mixed")
HOLDOUT_CONDITIONS = CONDITIONS + ("photo",)


def conditions_for(split):
    return HOLDOUT_CONDITIONS if split == "holdout" else CONDITIONS


def _deps():
    import cv2
    import numpy as np
    return cv2, np


# ------------------------------------------------------------------------------------------------ 字形

def _simple_shape(rng):
    """つながった小さな形（線1〜2本または弧）。stack の上下の部品に使う。"""
    grid = [(x / 2, y / 2) for x in range(3) for y in range(3)]
    if rng.random() < 0.35:
        start = rng.choice((0, 90, 180, 270))
        return [("arc", (0.5, 0.5), (0.45, 0.45), start, start + rng.choice((180, 270)))]
    a, b = rng.sample(grid, 2)
    parts = [("line", a, b)]
    if rng.random() < 0.6:
        parts.append(("line", b, rng.choice([g for g in grid if g not in (a, b)])))
    return parts


def stack_prototypes(n, seed):
    rng = random.Random(f"stack-{seed}")
    share = rng.choice((0.5, 0.8, 1.0))                    # 上下に分かれる字形の割合（1.0 = 全部）
    protos, seen = [], set()
    while len(protos) < n:
        if rng.random() < share:
            p = ("stack", _simple_shape(rng), _simple_shape(rng))
        else:
            p = ("full", _simple_shape(rng) + _simple_shape(rng))
        if repr(p) not in seen:
            seen.add(repr(p))
            protos.append(p)
    return protos, share


RADICALS = ("box", "bars_h2", "bars_h3", "bars_v2", "bars_v3", "cross", "corner", "tee", "diag", "dots")


def radical_prototypes(n, seed):
    rng = random.Random(f"radical-{seed}")
    protos, seen = [], set()
    while len(protos) < n:
        layout = rng.choice(("single", "lr", "tb", "tb", "tbb", "enclose"))
        k = {"single": 1, "lr": 2, "tb": 2, "tbb": 3, "enclose": 1}[layout]
        p = (layout, tuple(rng.choice(RADICALS) for _ in range(k)))
        if p not in seen:
            seen.add(p)
            protos.append(p)
    return protos


def curve_prototypes(n, seed):
    rng = random.Random(f"curve-{seed}")
    protos = []
    for _ in range(n):
        pts = [(rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)) for _ in range(rng.randint(4, 6))]
        body_top = rng.choice((0.0, 0.3, 0.3, 0.45))       # 本体が字の枠の上から始まる位置（低い本体の字がある）
        mark = rng.choice((None, None, ("above", rng.randint(1, 3)), ("below", rng.randint(1, 2)), ("bar", 1)))
        protos.append((tuple(pts), body_top, mark))
    return protos


def _radical(img, kind, x, y, w, h, t):
    cv2, _ = _deps()
    x2, y2 = x + w, y + h
    cx, cy = x + w // 2, y + h // 2
    if kind == "box":
        cv2.rectangle(img, (x, y), (x2, y2), 255, t)
    elif kind.startswith("bars_h"):
        k = int(kind[-1])
        for i in range(k):
            yy = y + round(i * h / (k - 1))
            cv2.line(img, (x, yy), (x2, yy), 255, t)
    elif kind.startswith("bars_v"):
        k = int(kind[-1])
        for i in range(k):
            xx = x + round(i * w / (k - 1))
            cv2.line(img, (xx, y), (xx, y2), 255, t)
    elif kind == "cross":
        cv2.line(img, (x, cy), (x2, cy), 255, t)
        cv2.line(img, (cx, y), (cx, y2), 255, t)
    elif kind == "corner":
        cv2.line(img, (x, y), (x, y2), 255, t)
        cv2.line(img, (x, y2), (x2, y2), 255, t)
    elif kind == "tee":
        cv2.line(img, (x, y), (x2, y), 255, t)
        cv2.line(img, (cx, y), (cx, y2), 255, t)
    elif kind == "diag":
        cv2.line(img, (x, y), (x2, y2), 255, t)
        cv2.line(img, (x2, y), (x, y2), 255, t)
    else:                                                  # dots: 横に並ぶ2点
        r = max(1, t)
        cv2.circle(img, (x + r, cy), r, 255, -1)
        cv2.circle(img, (x2 - r, cy), r, 255, -1)


def _draw(img, family, proto, x, y, w, h, t, gap):
    """字形を枠 [x, x+w] × [y, y+h] に描く。gap: 部品の間の隙間（画素、インクの間）。"""
    cv2, np = _deps()
    if family == "font":
        cv2.putText(img, proto, (int(x), int(y + h)), cv2.FONT_HERSHEY_SIMPLEX, h / 22.0, 255, t, cv2.LINE_8)
    elif family == "stroke":
        synth._draw_stroke(img, proto, x, y, w, h, t)
    elif family == "stack":
        if proto[0] == "full":
            synth._draw_stroke(img, proto[1], x, y, w, h, t)
        else:
            g = gap + t                                    # 線の太さの分だけ離して、インクの間に gap 画素を空ける
            hu = max(3, (h - g) // 2)
            synth._draw_stroke(img, proto[1], x, y, w, hu, t)
            synth._draw_stroke(img, proto[2], x, y + hu + g, w, max(3, h - hu - g), t)
    elif family == "radical":
        layout, rs = proto
        g = gap + t
        if layout == "single":
            _radical(img, rs[0], x, y, w, h, t)
        elif layout == "lr":
            wl = max(3, (w - g) // 2)
            _radical(img, rs[0], x, y, wl, h, t)
            _radical(img, rs[1], x + wl + g, y, max(3, w - wl - g), h, t)
        elif layout == "tb":
            hu = max(3, (h - g) // 2)
            _radical(img, rs[0], x, y, w, hu, t)
            _radical(img, rs[1], x, y + hu + g, w, max(3, h - hu - g), t)
        elif layout == "tbb":
            hu = max(3, (h - g) // 2)
            wl = max(3, (w - g) // 2)
            _radical(img, rs[0], x, y, w, hu, t)
            _radical(img, rs[1], x, y + hu + g, wl, max(3, h - hu - g), t)
            _radical(img, rs[2], x + wl + g, y + hu + g, max(3, w - wl - g), max(3, h - hu - g), t)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), 255, t)
            m = g + 1
            if w > 2 * m + 2 and h > 2 * m + 2:
                _radical(img, rs[0], x + m, y + m, w - 2 * m, h - 2 * m, t)
    elif family == "curve":
        pts, body_top, mark = proto
        by, bh = y + body_top * h, (1 - body_top) * h
        P = [(x + px * w, by + py * bh) for px, py in pts]
        Q = [P[0]] + P + [P[-1]]
        poly = []
        for i in range(1, len(Q) - 2):                     # Catmull–Rom
            p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
            for k in range(8):
                u = k / 8
                poly.append(tuple(0.5 * ((2 * p1[d]) + (-p0[d] + p2[d]) * u + (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * u * u
                                         + (-p0[d] + 3 * p1[d] - 3 * p2[d] + p3[d]) * u ** 3) for d in (0, 1)))
        poly.append(P[-1])
        arr = np.array([[min(max(px, x), x + w), min(max(py, by), by + bh)] for px, py in poly])
        cv2.polylines(img, [np.round(arr * 16).astype(np.int32)], False, 255, t, cv2.LINE_AA, 4)
        if mark:
            r = max(1, t)
            if mark[0] == "bar":
                yy = int(by - gap - t - 1)
                cv2.line(img, (int(x + 0.2 * w), yy), (int(x + 0.8 * w), yy), 255, t, cv2.LINE_AA)
            else:
                yy = int(by - gap - r - 1) if mark[0] == "above" else int(y + h + gap + r + 1)
                for k in range(mark[1]):
                    cv2.circle(img, (int(x + w * (k + 1) / (mark[1] + 1)), yy), r, 255, -1, cv2.LINE_AA)


def _mark(img, kind, x1, x2, y, t, size):
    """字の上または下の小さな記号（中心の高さ y）。"""
    cv2, _ = _deps()
    cx = (x1 + x2) // 2
    if kind == "dot":
        cv2.circle(img, (cx, y), max(1, t), 255, -1)
    elif kind == "bar":
        cv2.line(img, (cx - size, y), (cx + size, y), 255, t)
    else:
        cv2.line(img, (cx - size, y + size // 2), (cx, y - size // 2), 255, t)
        cv2.line(img, (cx, y - size // 2), (cx + size, y + size // 2), 255, t)


# ------------------------------------------------------------------------------------------------ 画像

def render(lines_symbols, family, condition, seed, n_classes):
    """lines_symbols: 行（記号番号のリスト）のリスト。戻り値: (画像, 正解)。"""
    cv2, np = _deps()
    rng = random.Random(f"layout-{family}-{condition}-{seed}")
    H0 = rng.choice((16, 20, 26, 34))                      # 字の基準の高さ（画素の絶対値に頼る方式を見分ける）
    t = 2 if H0 < 30 else rng.choice((2, 3))
    part_gap = max(3, round(rng.uniform(0.08, 0.2) * H0))  # 1つの字の中の部品の隙間（インクの間、画素）
    flags = {"varheight": condition in ("varheight", "mixed", "photo"), "diacritic": condition in ("diacritic", "mixed", "photo"),
             "jitter": 0.2 if condition == "jitter" else 0.08 if condition in ("mixed", "photo") else 0.0,
             "uneven": condition in ("uneven", "mixed", "photo"), "tight": condition == "tight"}
    info = {}
    if family == "font":
        if n_classes > len(synth.FONT_CHARS):
            raise ValueError("font の字形が足りない")
        protos = list(synth.FONT_CHARS[:n_classes])
    elif family == "stroke":
        protos = synth.stroke_prototypes(n_classes, seed)
    elif family == "stack":
        protos, info["stacked_share"] = stack_prototypes(n_classes, seed)
    elif family == "radical":
        protos = radical_prototypes(n_classes, seed)
    elif family == "curve":
        protos = curve_prototypes(n_classes, seed)
    else:
        raise ValueError(f"family {family}")
    aspect = [rng.uniform(0.6, 1.0) for _ in range(n_classes)]
    hfac = [rng.uniform(0.5, 1.3) if flags["varheight"] else 1.0 for _ in range(n_classes)]
    marks = [None] * n_classes
    if flags["diacritic"]:
        for s in range(n_classes):
            u = rng.random()
            if u < 0.5:
                marks[s] = ("above" if u < 0.35 else "below", rng.choice(("dot", "bar", "caret")))
    drop = [0.0] * n_classes                                # 下に伸びる字（基準線より下がる量、H0 に対する比）
    if flags["tight"]:
        for s in range(n_classes):
            u = rng.random()
            if u < 0.25:
                drop[s] = 0.3
            elif u < 0.5:
                hfac[s] = 1.3
    # 行の配置（two_column は左右の領域に分ける）
    n = len(lines_symbols)
    cols = [list(range(n))]
    if condition == "two_column" and n >= 4:
        k = math.ceil(n * 0.6)
        cols = [list(range(k)), list(range(k, n))]
    margin = 14
    reach = 1.4 * H0 + 3 * part_gap + 12                    # 1行が基準線の上下に伸びうる量
    col_w = [max(len(lines_symbols[i]) for i in c) * int(1.7 * H0 + 8) + 20 for c in cols]
    line_gaps = []
    for ci, c in enumerate(cols):
        gs = []
        for _ in c:
            if flags["uneven"]:                             # 上下に記号が付く字があるときは、その分だけ行を離す
                gs.append(rng.choice((0.9, 1.2, 1.8, 2.5) if flags["diacritic"] else (0.45, 0.6, 1.0, 1.8, 2.5)))
            elif flags["tight"]:                            # 上に 0.3・下に 0.3 伸びるので、行の範囲が 0.1 重なる
                gs.append(0.5)
            elif flags["diacritic"]:
                gs.append(0.9)
            else:
                gs.append(0.7 if ci == 0 else 1.3)
        line_gaps.append(gs)
    col_h = [sum(H0 * (1 + g) for g in gs) + 2 * reach + (0.6 * H0 if ci else 0) for ci, gs in enumerate(line_gaps)]
    W = int(2 * margin + sum(col_w) + 40 * (len(cols) - 1))
    Hh = int(2 * margin + max(col_h))
    ink = np.zeros((Hh, W), np.uint8)
    items, lines, rules, regions = [], [], [], []
    x_col = margin
    for ci, c in enumerate(cols):
        base = margin + reach + H0 + (0.6 * H0 if ci else 0)
        col_x2 = x_col
        for k, li in enumerate(c):
            x = x_col
            lid = f"L{li + 1}"
            row_items = []
            for s in lines_symbols[li]:
                h = max(6, int(round(H0 * hfac[s])))
                w = max(5, int(round(h * aspect[s])))
                dy = int(round(H0 * (drop[s] + rng.uniform(-flags["jitter"], flags["jitter"]))))
                layer = None
                for attempt in range(8):
                    layer = np.zeros_like(ink)
                    yt = int(base - h + dy)
                    _draw(layer, family, protos[s], x, yt, w, h, t, part_gap)
                    ys, xs = np.nonzero(layer >= 128)
                    if len(xs) == 0:
                        break
                    if marks[s]:
                        size = max(2, round(0.18 * H0))
                        my = int(ys.min()) - part_gap - size // 2 - t if marks[s][0] == "above" else \
                            int(ys.max()) + 1 + part_gap + size // 2 + t
                        _mark(layer, marks[s][1], int(xs.min()), int(xs.max()), my, t, size)
                        ys, xs = np.nonzero(layer >= 128)
                    if not flags["tight"]:                  # 行の範囲が重なる条件だけ、上の行の字との接触を調べる
                        break
                    near = cv2.dilate(ink, np.ones((5, 5), np.uint8))
                    if not np.any((near > 0) & (layer > 0)):
                        break
                    x += 3                                  # 他の字（上の行の下に伸びる字など）に触れるなら右にずらす
                ys, xs = np.nonzero(layer >= 128)
                if len(xs) == 0:
                    continue
                box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                ink = np.maximum(ink, layer)
                row_items.append({"box": box, "symbol": s})
                x = box[2] + max(4, round(0.3 * H0))
            if row_items:
                lb = [min(i["box"][0] for i in row_items), min(i["box"][1] for i in row_items),
                      max(i["box"][2] for i in row_items), max(i["box"][3] for i in row_items)]
                lines.append({"id": lid, "box": lb, "verified": True, "continues_sentence": False})
                col_x2 = max(col_x2, lb[2])
                for it in row_items:
                    items.append({"id": f"g{len(items) + 1}", "box": it["box"], "line": lid, "kind": "glyph",
                                  "label": f"c{it['symbol']}", "verified": True, "uncertain": False, "same_as": None,
                                  "symbol": it["symbol"]})
                if condition == "rule_touch" and k % 2 == 1:
                    rules.append(int(base))
            base += H0 * (1 + line_gaps[ci][k])
        regions.append([x_col - margin // 2, col_x2 + margin // 2])
        x_col = col_x2 + 40
    # 余白を切り落とす
    gx2 = max(i["box"][2] for i in items) + margin
    gy1 = max(0, min(i["box"][1] for i in items) - margin)
    gy2 = min(Hh, max(i["box"][3] for i in items) + margin)
    ink = ink[gy1:gy2, :gx2]
    Hh, W = ink.shape
    for o in items + lines:
        o["box"] = [o["box"][0], o["box"][1] - gy1, o["box"][2], o["box"][3] - gy1]
    if condition == "rule_touch":
        for yb in rules:
            yy = yb - gy1
            cv2.line(ink, (3, yy), (W - 4, yy), 255, 1)
            items.append({"id": f"r{len(items)}", "box": [3, yy, W - 3, yy + 1], "line": None, "kind": "rule", "label": None,
                          "verified": True, "uncertain": False, "same_as": None})
        cv2.line(ink, (5, 2), (5, Hh - 3), 255, 1)
        items.append({"id": "rv", "box": [5, 2, 6, Hh - 2], "line": None, "kind": "rule", "label": None,
                      "verified": True, "uncertain": False, "same_as": None})
    gt_regions = None
    if len(cols) > 1:
        cut = (regions[0][1] + regions[1][0]) // 2
        gt_regions = [{"id": "left", "box": [0, 0, cut, Hh]}, {"id": "right", "box": [cut, 0, W, Hh]}]
    img = 255 - ink
    if condition == "photo":
        img, shift = _photo(img, rng, seed)
        for o in items + lines:
            b = o["box"]
            ya, yb = shift(b[0], b[1]), shift(b[2], b[1])
            yc, yd = shift(b[0], b[3]), shift(b[2], b[3])
            o["box"] = [b[0], int(math.floor(min(ya, yb))), b[2], int(math.ceil(max(yc, yd)))]
        Hh, W = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gt = {"schema": "babel-glyph-gt-v1", "image": None, "image_size": [W, Hh], "source": "synthetic",
          "generator": {"module": "layout", "family": family, "condition": condition, "seed": seed, "n_classes": n_classes,
                        "glyph_height": H0, "part_gap": part_gap, **info},
          "lines": lines, "items": items, "regions": gt_regions, "annotators": ["synthetic-generator"],
          "notes": "人工画像の生成時の座標とクラス（評価専用）。認識器に渡すのは画像と regions（領域の枠）だけ"}
    return img, gt


def _photo(img, rng, seed):
    """写真・スクリーンショットに近づける劣化（holdout だけ）: 傾き・ぼかし・明るさのむら・雑音・JPEG。"""
    cv2, np = _deps()
    h, w = img.shape
    k = rng.choice((-1, 1)) * rng.uniform(0.006, 0.014)    # 傾き（約 0.3〜0.8 度）。縦方向のずれで近似する
    pad = int(math.ceil(abs(k) * w)) + 2
    off = pad if k < 0 else 2
    M = np.float32([[1, 0, 0], [k, 1, off]])
    out = cv2.warpAffine(img, M, (w, h + pad + 2), flags=cv2.INTER_LINEAR, borderValue=255)
    out = cv2.GaussianBlur(out, (0, 0), 0.8).astype(np.float32)
    grad = np.linspace(0, rng.uniform(15, 40), out.shape[1], dtype=np.float32)[None, :]
    out = out * 0.85 + 20 - grad                           # 紙の地を少し暗くし、左右で明るさを変える
    out += np.random.default_rng(seed).normal(0, 6, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 40])
    out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return out, (lambda x, y: y + k * x + off)


def n_lines_for(seed):
    return random.Random(f"layout-lines-{seed}").randint(5, 14)


def build(out_dir, split, families=None, conditions=None, seeds=None):
    """画像は out_dir/images、正解は out_dir/gt に分けて書く。"""
    cv2, _ = _deps()
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    cases = []
    for seed in seeds or SPLITS[split]:
        lines, alpha, _ = synth.text_for(seed, n_lines_for(seed))
        for fam in families or FAMILIES[split]:
            for cond in conditions or conditions_for(split):
                name = f"layout-{split}-{fam}-{cond}-{seed}"
                img, gt = render(lines, fam, cond, seed, n_classes=len(alpha))
                gt["image"] = f"{name}.png"
                gt["text_alphabet"] = alpha
                cv2.imwrite(str(out / "images" / f"{name}.png"), img)
                (out / "gt" / f"{name}.json").write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")
                cases.append({"name": name, "split": split, "style": fam, "condition": cond, "seed": seed,
                              "regions": gt["regions"], "gt_lines": len(gt["lines"])})
    return cases
