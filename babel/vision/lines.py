"""行認識（連結成分を行にまとめる）の方式。比較用に5つ実装する。どれも正解を使わない。

  center      A  現行（improved・improved-v2）: 大きな成分を縦の中心でまとめる。recognizers.segment_components の写し
  center_scaled A' 現行の尺度（成分の高さの中央値）だけを字の大きさ S に替えたもの（閾値・尺度の調整だけの候補）
  projection  B  行の投影: インクのある高さの連なりを行とし、成分を最も重なる行に入れる
  overlap     C  上下の範囲の重なり: 成分の縦の範囲が行の範囲と十分に重なれば同じ行
  baseline    D  下端（基準線）と字の高さ: 大きな成分を下端でまとめ、行の上下の範囲を推定し、残りを最も近い行に入れる
  hybrid      E  複合: 成分の位置から作った投影の谷で帯に分け、隙間の小さい帯どうしをまとめ、
                 小さな部品は縦の近さと横のそろい方で行を決める

A は成分の高さの中央値（点・破片を含む）を尺度に使う。B〜E は「字の大きさ」S を、成分の max(幅, 高さ) の
インク量で重みを付けた中央値で推定する（小さな部品や雑音が多くても字の大きさから離れにくい）。

成分は [x1, y1, x2, y2, 面積]。戻り値の行は成分のリスト（上から順）。
"""
from __future__ import annotations

from .glyphs import _dependencies, _runs, _without_rules

LINE_METHODS = ("center", "center_scaled", "projection", "overlap", "baseline", "hybrid")


def components(mask, remove_rules=True, min_area=3):
    """罫線を除いた二値画像と、その連結成分。"""
    cv2, np = _dependencies()
    m = _without_rules(mask) if remove_rules else mask
    n, _, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i])
        if a >= min_area:
            comps.append([x, y, x + w, y + h, a])
    return comps, m


def median_height(comps):
    """現行方式の尺度: すべての成分の高さの中央値。"""
    hs = sorted(c[3] - c[1] for c in comps)
    return hs[len(hs) // 2]


def glyph_scale(comps):
    """字の大きさ S: 成分の max(幅, 高さ) の、インク量で重みを付けた中央値。"""
    v = sorted((max(c[2] - c[0], c[3] - c[1]), c[4]) for c in comps)
    half, acc = sum(a for _, a in v) / 2, 0
    for s, a in v:
        acc += a
        if acc >= half:
            return s
    return v[-1][0]


def _cy(c):
    return (c[1] + c[3]) / 2


def _h(c):
    return c[3] - c[1]


def _quantile(vals, q):
    v = sorted(vals)
    return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]


def _order(rows):
    rows = [r for r in rows if r]
    rows.sort(key=lambda r: (sorted(_cy(c) for c in r)[len(r) // 2], min(c[0] for c in r)))
    return rows


# ------------------------------------------------------------------------------------------------ A 現行

def group_center(comps, scale=None, small_ratio=0.35, tol=0.6):
    """現行方式の写し。scale を渡すと、尺度（成分の高さの中央値）だけを差し替える（原因の切り分け用）。"""
    med_h = median_height(comps) if scale is None else scale
    big = [c for c in comps if _h(c) >= small_ratio * med_h]
    small = [c for c in comps if _h(c) < small_ratio * med_h]
    big.sort(key=_cy)
    rows = []
    for c in big:
        if rows:
            r = rows[-1]
            rcy = sum(_cy(d) for d in r) / len(r)
            if abs(_cy(c) - rcy) <= tol * med_h:
                r.append(c)
                continue
        rows.append([c])
    for c in small:
        if not rows:
            rows.append([c])
            continue
        best = min(range(len(rows)), key=lambda k: min(abs(_cy(c) - _cy(d)) for d in rows[k]))
        rows[best].append(c)
    return rows


# ------------------------------------------------------------------------------------------------ B 投影

def group_projection(comps, mask, S, bridge=0.1):
    """インクのある高さの連なり（隙間が bridge × S 以下ならつなぐ）を行とする。"""
    _, np = _dependencies()
    bands = [(a, b + 1) for a, b in _runs(np.any(mask > 0, axis=1), max_gap=int(bridge * S))]
    rows = [[] for _ in bands]
    for c in comps:
        k = max(range(len(bands)), key=lambda i: min(c[3], bands[i][1]) - max(c[1], bands[i][0]))
        rows[k].append(c)
    return _order(rows)


# ------------------------------------------------------------------------------------------------ C 上下の範囲の重なり

class _Band:
    def __init__(self, c):
        self.members = [c]
        self.top, self.bottom = c[1], c[3]

    def add(self, c):
        self.members.append(c)
        ref = max(_h(m) for m in self.members)
        core = [m for m in self.members if _h(m) >= 0.6 * ref]
        self.top, self.bottom = _quantile([m[1] for m in core], 0.5), _quantile([m[3] for m in core], 0.5)

    def ratio(self, c):
        ov = min(c[3], self.bottom) - max(c[1], self.top)
        return ov / max(1, min(_h(c), self.bottom - self.top))

    def gap(self, c):
        return max(0, c[1] - self.bottom, self.top - c[3])


def _overlap_bands(comps, S, tau):
    """字の大きさに近い成分から順に、縦の範囲が十分に重なる帯に入れる。重なる帯がなければ新しい帯。"""
    import math
    bands = []
    for c in sorted(comps, key=lambda c: (abs(math.log(max(1, _h(c)) / S)), c[1], c[0])):
        best = max(bands, key=lambda b: b.ratio(c), default=None)
        if best is not None and best.ratio(c) >= tau:
            best.add(c)
        else:
            bands.append(_Band(c))
    return bands


def group_overlap(comps, S, tau=0.5, small=0.4, attach=0.6):
    """縦の範囲の重なりだけで行を作る。どの行とも重ならない小さな部品は、近い行（attach × S 以内）に付ける。"""
    marks = [c for c in comps if max(c[2] - c[0], _h(c)) < small * S]
    marks_id = {id(c) for c in marks}
    bands = _overlap_bands([c for c in comps if id(c) not in marks_id], S, tau)
    extra = []
    for c in marks:
        best = min(bands, key=lambda b: (b.gap(c), -b.ratio(c)), default=None)
        if best is not None and best.gap(c) <= attach * S:
            best.members.append(c)
        else:
            extra.append([c])
    return _order([b.members for b in bands] + extra)


# ------------------------------------------------------------------------------------------------ D 下端と字の高さ

def group_baseline(comps, S, seed_ratio=0.5, tol=0.5, attach=1.0):
    """大きな成分を下端（基準線）でまとめて行の上下の範囲を推定し、残りを最も近い行に入れる。"""
    seeds = sorted([c for c in comps if _h(c) >= seed_ratio * S], key=lambda c: (c[3], c[0]))
    rest = [c for c in comps if _h(c) < seed_ratio * S]
    rows = []
    for c in seeds:
        if rows and c[3] - _quantile([d[3] for d in rows[-1]], 0.5) <= tol * S:
            rows[-1].append(c)
        else:
            rows.append([c])
    ext = [(_quantile([d[1] for d in r], 0.5), _quantile([d[3] for d in r], 0.5)) for r in rows]
    extra = []
    for c in rest:
        def dist(k):
            return max(0, c[1] - ext[k][1], ext[k][0] - c[3])
        k = min(range(len(rows)), key=lambda k: (dist(k), abs(_cy(c) - (ext[k][0] + ext[k][1]) / 2)), default=None)
        if k is not None and dist(k) <= attach * S:
            rows[k].append(c)
        else:
            extra.append([c])
    return _order(rows + extra)


# ------------------------------------------------------------------------------------------------ E 複合

def _extent(row):
    """行の中心部の上下（上に伸びる字・下に伸びる字に引きずられないよう、上端・下端の中央値）。"""
    return _quantile([c[1] for c in row], 0.5), _quantile([c[3] for c in row], 0.5)


def crossing_profile(comps, height):
    """高さ y ごとの「成分の中ほどを通る度合い」。成分ごとに、上端と下端で 0・中央で 1 になる山形を幅で重み付けして足す。

    インクの量の投影と違い、成分の大きさで正規化されるので、行の中心で高く、行と行の間で低くなる
    （上に伸びる字・下に伸びる字の先端は、隣の行に少し入っても値が小さい）。
    """
    _, np = _dependencies()
    prof = np.zeros(height + 1, np.float64)
    for c in comps:
        h = _h(c)
        y = np.arange(c[1], c[3] + 1)
        prof[c[1]:c[3] + 1] += (c[2] - c[0]) * np.minimum(y - c[1], c[3] - y) * 2.0 / h
    return prof


def _valley_cuts(prof, a, b, depth):
    """区間 [a, b] を谷で分ける。谷の値が、両側の山の低い方の depth 倍以下なら切る（再帰的に）。"""
    _, np = _dependencies()
    seg = prof[a:b + 1]
    if len(seg) < 3 or seg.max() <= 0:
        return []
    left = np.maximum.accumulate(seg)
    right = np.maximum.accumulate(seg[::-1])[::-1]
    lower = np.minimum(left, right)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(lower > 0, seg / lower, np.inf)
    ratio[(left <= 0) | (right <= 0)] = np.inf
    k = int(np.argmin(ratio))
    if ratio[k] > depth:
        return []
    # 同じ値が続く谷では中央で切る
    k2 = k
    while k2 + 1 < len(seg) and ratio[k2 + 1] == ratio[k]:
        k2 += 1
    y = a + (k + k2) // 2
    return _valley_cuts(prof, a, y, depth) + [y] + _valley_cuts(prof, y, b, depth)


def group_hybrid(comps, S, small=0.5, depth=0.3, join_gap=0.4, attach=1.0, x_weight=0.5):
    """複合: 投影（成分の中ほどを通る度合い）の谷で帯に分ける → 帯の結合 → 小さな部品の割り当て。

    1. 小さな部品（高さ < small × S。点・短い横線など）を除き、残りの成分で crossing_profile を作り、谷で切って帯にする
       （谷の値が両側の山の低い方の depth 倍以下）。成分は中心のある帯に入れる。この段階では切りすぎてよい
    2. 隣り合う帯の中心部（上端・下端の中央値）の隙間が join_gap × S 以下ならまとめる。隙間の小さい組から順にまとめる。
       上下に分かれた字の上の部品の帯と下の部品の帯は、ここで1つの行になる
    3. 小さな部品は、行の上下の範囲までの縦の距離と、最も近い字との横のずれ（x_weight を掛ける）の和が最小の行に入れる。
       どの行からも attach × S より離れていれば単独の行にする

    前提: 行と行の間の空きが、1つの字の中の部品の隙間より大きい（join_gap × S が両者の境目）。
    「字の列がそろっているか」で帯をまとめる条件も試したが、dev で結果が変わらなかったので入れていない
    （results/vision/v3/hybrid-sweep-dev-1-alignment.json）。
    """
    marks = [c for c in comps if _h(c) < small * S]
    marks_id = {id(c) for c in marks}
    body = [c for c in comps if id(c) not in marks_id]
    if not body:
        return _order([[c] for c in marks])
    top, bottom = min(c[1] for c in body), max(c[3] for c in body)
    prof = crossing_profile(body, bottom)
    cuts = [top] + _valley_cuts(prof, top, bottom, depth) + [bottom + 1]
    rows = [[] for _ in cuts[:-1]]
    for c in body:
        cy = _cy(c)
        k = max(i for i in range(len(rows)) if cuts[i] <= cy)
        rows[k].append(c)
    rows = _order(rows)
    while len(rows) > 1:
        best = None
        for i in range(len(rows) - 1):
            (t1, b1), (t2, b2) = _extent(rows[i]), _extent(rows[i + 1])
            gap = max(t1, t2) - min(b1, b2)
            if gap <= join_gap * S and (best is None or gap < best[0]):
                best = (gap, i)
        if best is None:
            break
        i = best[1]
        rows[i:i + 2] = [rows[i] + rows[i + 1]]
        rows = _order(rows)
    body_rows = [list(r) for r in rows]                 # 小さな部品の割り当ては、本体の成分だけを基準にする
    full = [(min(c[1] for c in r), max(c[3] for c in r)) for r in rows]
    extra = []
    for c in marks:
        cx = (c[0] + c[2]) / 2

        def cost(k):
            dv = max(0, c[1] - full[k][1], full[k][0] - c[3])
            dx = min(max(0, d[0] - cx, cx - d[2]) + abs(cx - (d[0] + d[2]) / 2) for d in body_rows[k])
            return dv, dv + x_weight * min(dx, S)        # 横のずれは S で頭打ち（遠い行の真下・真上より、今いる行を優先する）
        near = [k for k in range(len(body_rows)) if cost(k)[0] <= attach * S]
        if near:
            rows[min(near, key=lambda k: (cost(k)[1], cost(k)[0]))].append(c)
        else:
            extra.append([c])
    return _order(rows + extra)


# ------------------------------------------------------------------------------------------------ 入口

def group_rows(comps, mask, method="center", **kw):
    if not comps:
        return []
    if method == "center":
        return group_center(comps, **kw)
    S = kw.pop("scale", None) or glyph_scale(comps)
    if method == "center_scaled":                        # A の尺度だけを字の大きさ S に替えたもの（閾値の調整だけで直るかを試す）
        return group_center(comps, scale=S, **kw)
    if method == "projection":
        return group_projection(comps, mask, S, **kw)
    if method == "overlap":
        return group_overlap(comps, S, **kw)
    if method == "baseline":
        return group_baseline(comps, S, **kw)
    if method == "hybrid":
        return group_hybrid(comps, S, **kw)
    raise ValueError(f"行認識の方式 {method}")
