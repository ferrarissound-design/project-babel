"""転写の不確実性: 複数の転写候補・画像認識の設定違いで、仮説がどれだけ変わるか。

画像認識の設定を変えると視覚クラスの番号が変わるので、記号名では照合できない。
画像上の位置（根拠の枠）が重なるかで、同じ仮説かどうかを判定する。位置がない入力では記号名で照合する。
"""
from __future__ import annotations

import itertools
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u else 0.0


def _boxes(h):
    return [b for bs in _groups(h).values() for b in bs]


def _groups(h):
    """根拠の画像上の枠を、仮説の中での役割ごとに分ける。
    記号の組（共起・置換可能性）は記号ごと（evidence の role）、行の組は line_a / line_b ごと、それ以外は1つ。
    役割を混ぜて照合すると、片方の記号（または行）だけが共通な別の組が一致してしまう（Codex レビュー、PR #8）。"""
    groups = {}
    for e in h.get("evidence", []):
        if not isinstance(e, dict):
            continue
        if e.get("image_box"):
            groups.setdefault(str(e.get("role", "")), []).append(e["image_box"])
        for side in ("line_a", "line_b"):          # 行の組の根拠は、両方の行の位置を持つ
            if isinstance(e.get(side), dict) and e[side].get("image_box"):
                groups.setdefault(side, []).append(e[side]["image_box"])
    return groups


def _params(h):
    """記号名以外の、族ごとの仮説の鍵（間隔・行頭か行末か・均等か集中かなど）。記号名は転写ごとに変わりうる。"""
    return {k: v for k, v in (h.get("key") or {}).items() if k not in ("symbols", "lines", "pairs", "family")}


def _overlap(bh, bg, iou):
    # 半分以上（切り上げ）。切り捨てだと、3枠のうち1枠の重なりでも通ってしまう（Codex レビュー、PR #9）
    hit = sum(1 for x in bh if any(_iou(x, y) >= iou for y in bg))
    return hit >= max(1, (len(bh) + 1) // 2)


def same_hypothesis(h, g, iou=0.5):
    """同じ族で、族ごとの鍵（距離など）が同じで、根拠の画像上の位置が重なるか。
    根拠に役割がある仮説（記号の組・行の組）は、役割どうしを1対1に対応させ、どの役割も位置が重なることを求める
    （記号名は転写ごとに変わってよい）。位置がない場合は、記号名と鍵がすべて同じか（行の組は対象の行が半分以上重なるか）で照合する。"""
    if h["family"] != g["family"] or _params(h) != _params(g):
        return False
    gh, gg = _groups(h), _groups(g)
    if gh and gg:
        if len(gh) != len(gg):
            return False
        hs, gs = list(gh.values()), list(gg.values())
        return any(all(_overlap(a, b, iou) for a, b in zip(hs, perm)) for perm in itertools.permutations(gs))
    kh, kg = h.get("key") or {}, g.get("key") or {}
    if h["family"] == "linepair":
        a, b = set(kh.get("lines") or []), set(kg.get("lines") or [])
        return bool(a and b) and len(a & b) / len(a | b) >= 0.5
    return kh.get("symbols") == kg.get("symbols") and bool(kh)


def compare(main: dict, alternatives: list[tuple[str, dict]]):
    """主の解析の仮説ごとに、各転写候補でも多重検定を通ったかを記録する。"""
    rows = []
    for h in main["hypotheses"]:
        found = {name: any(same_hypothesis(h, g) for g in alt["hypotheses"]) for name, alt in alternatives}
        rate = sum(found.values()) / len(found) if found else None
        h.setdefault("stability", {})["across_transcriptions"] = round(rate, 2) if rate is not None else None
        if rate is not None and rate < 0.5:
            h["limits"].append(f"別の転写候補 {len(found)} 件のうち {sum(found.values())} 件でしか再現しない")
        rows.append({"id": h["id"], "observation": h["observation"], "reproduced_in": found})
    only_alt = []
    for name, alt in alternatives:
        for g in alt["hypotheses"]:
            if not any(same_hypothesis(h, g) for h in main["hypotheses"]):
                only_alt.append({"transcription": name, "observation": g["observation"], "tier": g["tier"]})
    return {"alternatives": [n for n, _ in alternatives], "by_hypothesis": rows,
            "only_in_alternatives": only_alt[:30],
            "summary": {name: {"hypotheses": len(a["hypotheses"]), "verdict": a["verdict"]["code"],
                               "symbol_types": a["statistics"]["symbol_types"]} for name, a in alternatives}}


def vision_variants(image, similarities, regions=None, threshold=None, remove_rules=True, out_dir=None):
    """同じ画像を、形状の類似度の閾値を変えて転写する。戻り値: [(名前, glyph-report)]。"""
    from babel.vision.glyphs import analyze_image
    out = []
    for s in similarities:
        rep = analyze_image(image, regions=regions, threshold=threshold, similarity=s, remove_rules=remove_rules)
        out.append((f"similarity={s}", rep))
        if out_dir:
            import json
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / f"glyph-report-sim{s}.json").write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
    return out


def annotate(image, analysis, out_png, max_hypotheses=8):
    """仮説の根拠の位置を画像に描く（人が確認するため）。OpenCV が必要。"""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("画像への書き込みには pip install '.[vision]' が必要") from exc
    img = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"画像を読めない: {image}")
    colors = [(40, 40, 230), (40, 170, 40), (230, 120, 30), (180, 40, 180), (30, 160, 200), (120, 120, 30),
              (90, 30, 160), (30, 90, 90)]
    for k, h in enumerate(analysis["hypotheses"][:max_hypotheses]):
        col = colors[k % len(colors)]
        for b in _boxes(h):
            cv2.rectangle(img, (b[0], b[1]), (b[2] - 1, b[3] - 1), col, 1)
            cv2.putText(img, h["id"], (b[0], max(8, b[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_png), img)
    return out_png
