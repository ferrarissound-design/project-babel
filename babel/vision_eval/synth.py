"""評価用の人工文字画像と、その正解（babel-glyph-gt-v1, source=synthetic）。

描画方法（生成原理の違う2系統）
- font:   OpenCV の Hershey フォントの英字（既存の転写実験 T2 と同じ系統）
- stroke: フォントを使わず、線・弧・点を組み合わせて作った字形（シードごとに別の字形の集合）。
          離れた部品を持つ字形（点と線など）を含むので、1文字が分割される誤りを試せる

条件（condition）
- clean          明瞭
- noise          ガウス雑音と点状の汚れ
- broken         文字の一部を消す（欠け）
- similar        1本の短い線・点だけが違う字形の対（stroke のみ意味がある）
- size_thickness 大きさ ±25%・太さ 1〜3
- rules          罫線（横線・縦線）が文字と交差し、枠と装飾の点がある（正解に rule / decoration として記録）
- spacing        文字の間隔が不均一（接触する組を含む）
- wrapped        文を行幅で折り返す（改行が文の区切りでない行を正解に記録）

正解の枠は、雑音・欠けを加える前の、その記号だけを描いたインクの外接矩形。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

CONDITIONS = ("clean", "noise", "broken", "similar", "size_thickness", "rules", "spacing", "wrapped")
STYLES = ("font", "stroke")
SPLITS = {"dev": list(range(0, 10)), "val": list(range(100, 110)), "test": list(range(1000, 1020))}


def _deps():
    import cv2
    import numpy as np
    return cv2, np


# ------------------------------------------------------------------------------------------------ 字形

def stroke_prototypes(n, seed, similar=False):
    """n 種の字形。各字形は 0〜1 の正方形の中の部品（線・弧・点）のリスト。"""
    rng = random.Random(f"stroke-{seed}")
    grid = [(x / 2, y / 2) for x in range(3) for y in range(3)]
    protos, seen = [], set()

    def new_shape():
        parts = []
        for _ in range(rng.randint(2, 3)):
            kind = rng.choice(("line", "line", "arc", "dot")) if parts else rng.choice(("line", "arc"))
            if kind == "line":
                a, b = rng.sample(grid, 2)
                parts.append(("line", a, b))
            elif kind == "arc":
                c = rng.choice([(0.5, 0.5), (0.5, 0.3), (0.5, 0.7)])
                start = rng.choice((0, 90, 180, 270))
                parts.append(("arc", c, (0.35, 0.3), start, start + rng.choice((180, 270))))
            else:
                parts.append(("dot", rng.choice(grid)))
        return parts
    while len(protos) < n:
        if similar and len(protos) % 2 == 1:
            base = protos[-1]
            extra = ("dot", rng.choice([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)])) if rng.random() < 0.5 else \
                ("line", (0.8, 0.0), (1.0, 0.2))
            shape = base + [extra]
        else:
            shape = new_shape()
        key = repr(shape)
        if key in seen:
            continue
        seen.add(key)
        protos.append(shape)
    return protos


FONT_CHARS = "abcdefghkmnopqrstuvwxyzABDEGHKMNPRTW"


def _draw_stroke(img, shape, x, y, w, h, t):
    cv2, _ = _deps()
    def P(p):
        return (int(round(x + p[0] * w)), int(round(y + p[1] * h)))
    for part in shape:
        if part[0] == "line":
            cv2.line(img, P(part[1]), P(part[2]), 255, t, cv2.LINE_8)
        elif part[0] == "arc":
            c, ax = part[1], part[2]
            cv2.ellipse(img, P(c), (max(1, int(ax[0] * w)), max(1, int(ax[1] * h))), 0, part[3], part[4], 255, t, cv2.LINE_8)
        else:
            cv2.circle(img, P(part[1]), max(1, t), 255, -1, cv2.LINE_8)


def _draw_font(img, ch, x, y, h, t):
    cv2, _ = _deps()
    scale = h / 22.0
    cv2.putText(img, ch, (int(x), int(y + h)), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, t, cv2.LINE_8)


# ------------------------------------------------------------------------------------------------ 画像

def render(lines_symbols, style, condition, seed, n_classes=None):
    """lines_symbols: 文（記号番号のリスト）のリスト。戻り値: (画像, 正解)。"""
    cv2, np = _deps()
    rng = random.Random(f"render-{style}-{condition}-{seed}")
    syms = sorted({s for ln in lines_symbols for s in ln})
    n_classes = n_classes or (max(syms) + 1)
    if style == "stroke":
        protos = stroke_prototypes(n_classes, seed, similar=(condition == "similar"))
        aspect = [rng.uniform(0.6, 1.0) for _ in range(n_classes)]
    else:
        if n_classes > len(FONT_CHARS):
            raise ValueError("font の字形が足りない")
        chars = FONT_CHARS[:n_classes]
    # 行の間隔: 34 では font の下に伸びる字形で行の隙間がちょうど6画素になり、OpenCV の版によって
    # 既存方式の行の橋渡し（6画素まで）に入ったり外れたりした（docs/ISSUES.md #61）。版に依存しない余裕をとる
    H0, margin, line_h = 20, 12, 40
    # 文を行に配置する（wrapped は行幅で折り返し、それ以外は1文1行）
    width_limit = 520 if condition == "wrapped" else 10 ** 9
    rows, cont = [], []
    for sent in lines_symbols:
        cur = []
        for s in sent:
            cur.append(s)
            if condition == "wrapped" and len(cur) * 22 > width_limit - 2 * margin:
                rows.append(cur)
                cont.append(True)
                cur = []
        if cur:
            rows.append(cur)
            cont.append(False)
    max_len = max(len(r) for r in rows)
    W = margin * 2 + max_len * 30 + 20
    Hh = margin * 2 + line_h * len(rows)
    ink = np.zeros((Hh, W), np.uint8)
    items, lines = [], []
    for li, row in enumerate(rows):
        x = margin
        y0 = margin + li * line_h + 6
        lid = f"L{li + 1}"
        row_items = []
        for s in row:
            h = H0
            t = 2
            if condition == "size_thickness":
                h = int(round(H0 * rng.uniform(0.75, 1.25)))
                t = rng.choice((1, 2, 3))
            layer = np.zeros_like(ink)
            if style == "stroke":
                w = int(round(h * aspect[s]))
                _draw_stroke(layer, protos[s], x, y0 + (H0 - h), w, h, t)
            else:
                _draw_font(layer, chars[s], x, y0 + (H0 - h) - 4, h, t)
            ys, xs = np.nonzero(layer)
            if len(xs) == 0:
                continue
            box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            ink |= layer
            row_items.append({"box": box, "label": f"c{s}", "symbol": s})
            gap = 6
            if condition == "spacing":
                gap = rng.choice((0, 1, 2, 4, 8, 12, 16))
            x = box[2] + gap
        if not row_items:
            continue
        lb = [min(i["box"][0] for i in row_items), min(i["box"][1] for i in row_items),
              max(i["box"][2] for i in row_items), max(i["box"][3] for i in row_items)]
        lines.append({"id": lid, "box": lb, "verified": True, "continues_sentence": cont[li]})
        for it in row_items:
            items.append({"id": f"g{len(items) + 1}", "box": it["box"], "line": lid, "kind": "glyph", "label": it["label"],
                          "verified": True, "uncertain": False, "same_as": None, "symbol": it["symbol"]})
    if condition == "broken":
        for it in items:
            if rng.random() < 0.5:
                x1, y1, x2, y2 = it["box"]
                cx, cy = rng.randint(x1, max(x1, x2 - 4)), rng.randint(y1, max(y1, y2 - 4))
                ink[cy:cy + 4, cx:cx + 4] = 0
    if condition == "rules":
        for li, ln in enumerate(lines):
            if li % 3 == 1:
                yy = (ln["box"][1] + ln["box"][3]) // 2
                cv2.line(ink, (margin // 2, yy), (W - margin // 2, yy), 255, 1)
                items.append({"id": f"r{li}", "box": [margin // 2, yy, W - margin // 2 + 1, yy + 1], "line": None,
                              "kind": "rule", "label": None, "verified": True, "uncertain": False, "same_as": None})
        cv2.rectangle(ink, (1, 1), (W - 2, Hh - 2), 255, 1)
        items.append({"id": "frame", "box": [1, 1, W - 1, Hh - 1], "line": None, "kind": "decoration", "label": None,
                      "verified": True, "uncertain": False, "same_as": None, "outline": True})
        xx = W - margin + 2
        for k in range(3):
            yy = margin + k * 7
            cv2.circle(ink, (xx, yy), 1, 255, -1)
            items.append({"id": f"dot{k}", "box": [xx - 1, yy - 1, xx + 2, yy + 2], "line": None, "kind": "decoration",
                          "label": None, "verified": True, "uncertain": False, "same_as": None})
    img = 255 - ink
    if condition == "noise":
        nrng = np.random.default_rng(seed)
        img = np.clip(img.astype(float) + nrng.normal(0, 28, img.shape), 0, 255)
        speck = nrng.random(img.shape) < 0.002
        img[speck] = 0
        img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gt = {"schema": "babel-glyph-gt-v1", "image": None, "image_size": [W, Hh], "source": "synthetic",
          "generator": {"style": style, "condition": condition, "seed": seed, "n_classes": n_classes},
          "lines": lines, "items": items, "annotators": ["synthetic-generator"],
          "notes": "人工画像の生成時の座標とクラス（評価専用）。認識器に渡さない"}
    return img, gt


def text_for(seed, n_lines, n_classes=None):
    """画像に描く文。構造解析の評価と同じ規則のある人工言語（regular）を使い、記号を番号にする。"""
    from babel.structure_eval.generators import Regular
    rows = Regular(seed).sample(n_lines, which="vision")
    alpha = sorted({c for ws, _ in rows for w in ws for c in w})
    idx = {c: i for i, c in enumerate(alpha)}
    return [[idx[c] for w in ws for c in w] for ws, _ in rows], alpha, rows


def build(out_dir, split, styles=STYLES, conditions=CONDITIONS, n_lines=13):
    """画像は out_dir/images、正解は out_dir/gt に分けて書く（認識器には images だけを渡す）。"""
    cv2, _ = _deps()
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    cases = []
    for seed in SPLITS[split]:
        lines, alpha, _ = text_for(seed, n_lines)
        for style in styles:
            for cond in conditions:
                name = f"{split}-{style}-{cond}-{seed}"
                img, gt = render(lines, style, cond, seed, n_classes=len(alpha))
                gt["image"] = f"{name}.png"
                gt["text_alphabet"] = alpha
                cv2.imwrite(str(out / "images" / f"{name}.png"), img)
                (out / "gt" / f"{name}.json").write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")
                cases.append({"name": name, "split": split, "style": style, "condition": cond, "seed": seed})
    return cases
