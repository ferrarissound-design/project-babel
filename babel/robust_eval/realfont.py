"""実在の文字体系を、実在のフォントで描いた評価画像（正解つき）。

人工の字形（babel/vision_eval/synth.py・layout.py）は、このリポジトリの認識器と同じ作者が設計した。ここでは、字形の設計に
作者が関わっていないデータとして、OS に入っているフォントの字（グルジア文字・アルメニア文字・タイ文字・片仮名・平仮名・
漢字・ハングル）を Pillow で描く（アンチエイリアスあり）。文の内容（記号の並び）は texts.py が作り、字は番号で割り当てる。

- 評価専用。Pillow とフォントがない環境では使えない（available() が False を返す）
- 正解の枠は、字を1つずつ描いたときのインクの範囲
- 実画像（写真・手書き・劣化した資料）の代わりにはならない: 同じ字は画素までほぼ同じに描かれる
"""
from __future__ import annotations

import random
from pathlib import Path

FONT_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts"]

# 文字体系 → (フォントのファイル名の候補, 字の候補)。字の候補は Unicode の範囲から機械的に取り、フォントにない字は除く
SCRIPTS = {
    "georgian": (["DejaVuSans.ttf"], [chr(c) for c in range(0x10D0, 0x10F1)]),
    "armenian": (["DejaVuSans.ttf"], [chr(c) for c in range(0x0561, 0x0587)]),
    "thai": (["Loma.ttf", "Loma.otf"], [chr(c) for c in range(0x0E01, 0x0E2F)]),
    "katakana": (["ipag.ttf", "ipagp.ttf"], [chr(c) for c in range(0x30A1, 0x30F7)]),
    "hiragana": (["ipag.ttf", "ipagp.ttf"], [chr(c) for c in range(0x3041, 0x3094)]),
    "kanji": (["ipag.ttf", "wqy-zenhei.ttc"], [chr(c) for c in range(0x4E00, 0x4E00 + 900)]),
    "hangul": (["wqy-zenhei.ttc", "unifont.otf", "unifont.ttf"], [chr(0xAC00 + 37 * k) for k in range(300)]),
}


def _find_font(names):
    for d in FONT_DIRS:
        root = Path(d)
        if not root.exists():
            continue
        for n in names:
            hits = sorted(root.rglob(n))
            if hits:
                return hits[0]
    return None


def available(script=None):
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    names = [script] if script else list(SCRIPTS)
    return all(_find_font(SCRIPTS[s][0]) is not None for s in names)


def _glyph(font, ch):
    """字1つのインクの画像（NumPy、0〜255）と、描画の原点からのずれ。"""
    import numpy as np
    from PIL import Image, ImageDraw
    size = font.size * 3
    im = Image.new("L", (size, size), 0)
    ImageDraw.Draw(im).text((font.size, font.size), ch, fill=255, font=font)
    a = np.array(im)
    ys, xs = np.nonzero(a > 40)
    if len(xs) == 0:
        return None
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return a[y1:y2, x1:x2], (int(x1) - font.size, int(y1) - font.size)


def inventory(script, px, n):
    """フォントに字形があり、互いに画像が違う字を n 個。戻り値: (フォント, [(字, インク, ずれ)])"""
    from PIL import ImageFont
    path = _find_font(SCRIPTS[script][0])
    if path is None:
        raise RuntimeError(f"フォントがない: {SCRIPTS[script][0]}")
    font = ImageFont.truetype(str(path), px)
    missing = _glyph(font, "￿")
    seen, out = set(), []
    if missing is not None:
        seen.add(missing[0].tobytes())
    for ch in SCRIPTS[script][1]:
        g = _glyph(font, ch)
        if g is None or g[0].tobytes() in seen:
            continue
        seen.add(g[0].tobytes())
        out.append((ch, g[0], g[1]))
        if len(out) == n:
            break
    if len(out) < n:
        raise ValueError(f"{script}: 字が足りない（{len(out)} < {n}）")
    return font, out


def max_vocab(script, px=28):
    try:
        return len(inventory(script, px, 10 ** 6)[1])
    except ValueError as e:
        return int(str(e).split("（")[1].split(" ")[0])


def render(lines_symbols, script, seed, n_classes, degrade=False):
    """lines_symbols: 行（記号番号のリスト）のリスト。戻り値: (BGR 画像, 正解)。layout.render と同じ形の正解を返す。"""
    import cv2
    import numpy as np
    rng = random.Random(f"realfont-{script}-{seed}")
    px = rng.choice((22, 28, 36))
    font, inv = inventory(script, px, n_classes)
    order = list(range(len(inv)))
    rng.shuffle(order)                                       # 記号番号 → 字 の割り当てを画像ごとに変える
    track = max(2, round(px * rng.uniform(0.12, 0.3)))       # 字の間の空き（インクの間、画素）
    pitch = round(px * rng.uniform(1.5, 2.0))
    margin = px
    top = min(g[2][1] for g in inv)
    width = margin * 2 + max(sum(inv[order[s]][1].shape[1] + track for s in ln) for ln in lines_symbols)
    height = margin * 2 + pitch * len(lines_symbols)
    ink = np.zeros((height, width), dtype=np.uint8)
    items, lines = [], []
    for li, ln in enumerate(lines_symbols):
        x = margin
        y0 = margin + li * pitch - top
        boxes = []
        for s in ln:
            _, g, (dx, dy) = inv[order[s]]
            h, w = g.shape
            yy = y0 + dy
            ink[yy:yy + h, x:x + w] = np.maximum(ink[yy:yy + h, x:x + w], g)
            box = [int(x), int(yy), int(x + w), int(yy + h)]
            boxes.append(box)
            items.append({"id": f"g{len(items)}", "box": box, "line": f"L{li}", "kind": "glyph", "label": f"c{s}",
                          "verified": True, "uncertain": False, "same_as": None})
            x += w + track
        lines.append({"id": f"L{li}", "box": [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes),
                                               max(b[3] for b in boxes)], "region": None, "verified": True})
    img = 255 - ink
    if degrade:                                              # 画面の写しに近い劣化: 少しのぼかしと JPEG
        img = cv2.GaussianBlur(img, (0, 0), 0.6)
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 55])
        img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gt = {"schema": "babel-glyph-gt-v1", "image": None, "image_size": [int(width), int(height)], "source": "synthetic",
          "generator": {"module": "realfont", "script": script, "seed": seed, "n_classes": n_classes, "font_px": px,
                        "font": Path(font.path).name, "degrade": degrade},
          "lines": lines, "items": items, "regions": None, "annotators": ["synthetic-generator"],
          "notes": "実在のフォントで描いた評価画像（評価専用）。認識器に渡すのは画像だけ"}
    return img, gt
