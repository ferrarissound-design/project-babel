"""画像の正解転写（ground truth）の形式: schema "babel-glyph-gt-v1"。

人が画像を見て作る・直すためのデータ。認識器の出力（glyph-report）とは別物で、混ぜない。

{
  "schema": "babel-glyph-gt-v1",
  "image": "page.png", "image_size": [W, H],
  "source": "human" | "synthetic",          # synthetic = 生成器が描いた座標（評価専用）
  "lines": [{"id": "L1", "box": [x1,y1,x2,y2], "verified": true,
             "continues_sentence": false}],    # true: 次の行に文が続く（改行は文の区切りではない）
  "items": [{"id": "g1", "box": [x1,y1,x2,y2], "line": "L1",
             "kind": "glyph" | "decoration" | "rule" | "noise",
             "label": "A" | null,             # 人が付けた字形の名前（同じ形なら同じ名前）。未確認なら null
             "verified": false,               # 人が枠と label を確認したか
             "uncertain": false,              # 人が見ても判断できない
             "same_as": null,                 # 別名で付けた label の統合先（例: "A"）
             "outline": false,                # 枠のような中空の要素（評価では内部ではなく縁だけを見る）
             "proposed": {"class": 12, "by": "vision-0.7"}}],   # 認識器の案（参考。正解ではない）
  "annotators": ["..."], "notes": "..."
}

規則
- 座標は画素、右下は含まない（babel vision と同じ）。
- label は人が付けたものだけ。認識器の分類は proposed に入れ、label には入れない（AI の推定を正解として登録しない）。
- verified=false の項目は評価に使わない。評価できる項目がなければ「測定不能」。
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA = "babel-glyph-gt-v1"
KINDS = ("glyph", "decoration", "rule", "noise")


class GroundTruthError(ValueError):
    pass


def _box(b, where):
    if not (isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, int) and not isinstance(v, bool) for v in b)):
        raise GroundTruthError(f"{where}: box は整数4つ [x1,y1,x2,y2]")
    x1, y1, x2, y2 = b
    if not (x1 < x2 and y1 < y2):
        raise GroundTruthError(f"{where}: box の右下は左上より大きいこと")
    return [x1, y1, x2, y2]


def validate(gt: dict) -> dict:
    """形式を検査し、既定値を補った写しを返す。"""
    if gt.get("schema") != SCHEMA:
        raise GroundTruthError(f"schema は {SCHEMA}")
    out = dict(gt)
    lines, ids = [], set()
    for i, ln in enumerate(gt.get("lines") or []):
        lid = str(ln.get("id") or f"L{i + 1}")
        if lid in ids:
            raise GroundTruthError(f"行 ID {lid} が重複")
        ids.add(lid)
        lines.append({"id": lid, "box": _box(ln.get("box"), f"行 {lid}"), "verified": bool(ln.get("verified", False)),
                      "continues_sentence": bool(ln.get("continues_sentence", False))})
    items, iids = [], set()
    for i, it in enumerate(gt.get("items") or []):
        gid = str(it.get("id") or f"g{i + 1}")
        if gid in iids:
            raise GroundTruthError(f"記号 ID {gid} が重複")
        iids.add(gid)
        kind = it.get("kind", "glyph")
        if kind not in KINDS:
            raise GroundTruthError(f"記号 {gid}: kind は {KINDS} のどれか")
        line = it.get("line")
        if line is not None and str(line) not in ids:
            raise GroundTruthError(f"記号 {gid}: 存在しない行 {line}")
        label = it.get("label")
        items.append({"id": gid, "box": _box(it.get("box"), f"記号 {gid}"), "line": None if line is None else str(line),
                      "kind": kind, "label": None if label in (None, "") else str(label),
                      "verified": bool(it.get("verified", False)), "uncertain": bool(it.get("uncertain", False)),
                      "same_as": it.get("same_as") or None, "proposed": it.get("proposed"),
                      "outline": bool(it.get("outline", False))})
    labels = {it["label"] for it in items if it["label"]}
    for it in items:
        if it["same_as"] and it["same_as"] not in labels:
            raise GroundTruthError(f"記号 {it['id']}: same_as の統合先 {it['same_as']} がどの label にもない")
    out["lines"], out["items"] = lines, items
    out.setdefault("source", "human")
    return out


def canonical_label(item, aliases=None):
    """same_as を辿った字形の名前（別名で付けた同じ形を1つにまとめる）。"""
    lab = item.get("same_as") or item.get("label")
    seen = set()
    while aliases and lab in aliases and lab not in seen:
        seen.add(lab)
        lab = aliases[lab]
    return lab


def aliases(gt):
    return {it["label"]: it["same_as"] for it in gt["items"] if it.get("label") and it.get("same_as")}


def load(path) -> dict:
    return validate(json.loads(Path(path).read_text(encoding="utf-8")))


def save(gt: dict, path):
    gt = validate(gt)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(gt, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def draft_from_report(report: dict, image_name=None) -> dict:
    """認識器の出力から、人が直すための下書きを作る。すべて未確認（verified=false, label=null）。
    認識器の分類は proposed にだけ入れる。"""
    lines, items = [], []
    glyphs = report.get("glyphs") or []
    offset = 0
    for r, row in enumerate(report.get("rows") or []):
        lid = f"L{r + 1}"
        lines.append({"id": lid, "box": list(row["box"]), "verified": False, "continues_sentence": False})
        idx = row.get("glyph_indices") or list(range(offset, offset + len(row.get("classes") or [])))
        offset += len(row.get("classes") or [])
        for gi in idx:
            g = glyphs[gi]
            items.append({"id": f"g{len(items) + 1}", "box": list(g["box"]), "line": lid, "kind": "glyph", "label": None,
                          "verified": False, "uncertain": False, "same_as": None,
                          "proposed": {"class": g.get("class_id"), "match_iou": g.get("match_iou"),
                                       "by": f"babel-vision similarity={(report.get('settings') or {}).get('similarity')}"}})
    return validate({"schema": SCHEMA, "image": image_name or report.get("source_image"), "image_size": report.get("image_size"),
                     "source": "human", "lines": lines, "items": items, "annotators": [],
                     "notes": "認識器の出力から作った下書き。すべて未確認。人が画像を見て枠・label・verified を直すこと"})


def status(gt: dict) -> dict:
    items = gt["items"]
    glyphs = [i for i in items if i["kind"] == "glyph"]
    return {"items": len(items), "glyphs": len(glyphs), "verified_glyphs": sum(i["verified"] for i in glyphs),
            "labelled_verified_glyphs": sum(1 for i in glyphs if i["verified"] and i["label"]),
            "uncertain": sum(i["uncertain"] for i in items), "non_glyph": len(items) - len(glyphs),
            "lines": len(gt["lines"]), "verified_lines": sum(ln["verified"] for ln in gt["lines"]),
            "labels": len({canonical_label(i, aliases(gt)) for i in glyphs if i["verified"] and i["label"]})}
