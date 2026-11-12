"""構造解析の入力: 記号列のコーパス。

受け付ける形式
- BABEL の public コーパス（documents[].sentences）。文字列なら1文字＝1記号、空白は「入力に書かれていた区切り」。
  文を記号名のリストで書くこともできる（複数文字の記号名を使う場合）。
- 画像解析（babel vision）の glyph-report.json。行＝1行、視覚クラス＝記号（g<番号>）。記号の画像上の位置を保持する。
- テキストファイル（1行＝1行、1文字＝1記号）。

入力にある空白（単語区切り）は解析に使わない。正解ではなく、入力側の主張として保存し、
分割候補との一致率を報告するだけにする（評価用の正解をアルゴリズムへ渡さないため）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Corpus:
    lines: list                 # list[tuple[int, ...]] 記号ID列
    symbols: list               # 記号ID → 記号名
    line_info: list             # 行ごとの {"doc":..., "sent":..., "row":...}
    origins: list               # 行ごとの、各記号の元の位置（転写の訂正・攪乱の後も元の位置を辿る）
    boxes: dict = field(default_factory=dict)        # (行, 元の位置) → 画像上の [x1,y1,x2,y2]
    given_boundaries: list | None = None              # 行ごとの、入力に書かれていた区切りの位置の集合
    uncertain: set = field(default_factory=set)       # 人間が「不確か」と印を付けた (行, 元の位置)
    source: dict = field(default_factory=dict)
    corrections_applied: list = field(default_factory=list)

    @property
    def n_symbols(self):
        return sum(len(ln) for ln in self.lines)

    def name(self, i):
        return self.symbols[i]

    def names(self, seq):
        return [self.symbols[i] for i in seq]

    def copy_with_lines(self, lines, origins, symbols=None):
        return Corpus(lines=[tuple(x) for x in lines], symbols=list(symbols or self.symbols), line_info=self.line_info,
                      origins=[list(o) for o in origins], boxes=self.boxes, given_boundaries=None,
                      uncertain=self.uncertain, source=self.source, corrections_applied=self.corrections_applied)

    def subset(self, idx):
        """行の部分集合（元の行番号は line_info に残る）。"""
        c = Corpus(lines=[self.lines[i] for i in idx], symbols=self.symbols, line_info=[self.line_info[i] for i in idx],
                   origins=[self.origins[i] for i in idx], boxes=self.boxes, given_boundaries=None,
                   uncertain=self.uncertain, source=self.source, corrections_applied=self.corrections_applied)
        return c


class _Builder:
    def __init__(self):
        self.ids, self.symbols = {}, []

    def sym(self, name):
        if name not in self.ids:
            self.ids[name] = len(self.symbols)
            self.symbols.append(name)
        return self.ids[name]


def _split_sentence(s):
    """文字列 → (記号名のリスト, 区切りの位置の集合)。"""
    if isinstance(s, (list, tuple)):
        return [str(x) for x in s], None
    syms, bounds = [], set()
    for ch in str(s):
        if ch.isspace():
            if syms:
                bounds.add(len(syms))
            continue
        syms.append(ch)
    bounds.discard(len(syms))
    return syms, bounds


def from_public(public: dict) -> Corpus:
    b = _Builder()
    lines, info, origins, given = [], [], [], []
    has_bounds = bool(public.get("has_word_boundaries"))
    for d_i, d in enumerate(public.get("documents") or []):
        did = str(d.get("id", f"d{d_i + 1:03d}"))
        for s_i, s in enumerate(d.get("sentences") or []):
            syms, bounds = _split_sentence(s)
            if not syms:
                continue
            lines.append(tuple(b.sym(x) for x in syms))
            info.append({"doc": did, "sent": s_i})
            origins.append(list(range(len(syms))))
            given.append(bounds if bounds is not None else set())
    return Corpus(lines=lines, symbols=b.symbols, line_info=info, origins=origins,
                  given_boundaries=given if has_bounds else None,
                  source={"format": "babel-public", "language_id": public.get("language_id"),
                          "level": public.get("level"), "has_word_boundaries": has_bounds})


LOW_MATCH = 0.85      # 視覚クラスへの一致度（IoU）がこれ未満の記号は、誤分類の可能性が高い記号として記録する


def from_glyph_report(report: dict) -> Corpus:
    b = _Builder()
    lines, info, origins, boxes = [], [], [], {}
    glyphs = report.get("glyphs") or []
    offset, low = 0, []
    for r_i, row in enumerate(report.get("rows") or []):
        classes = row.get("classes") or []
        idx = row.get("glyph_indices") or list(range(offset, offset + len(classes)))
        offset += len(classes)
        if not classes:
            continue
        ln = len(lines)
        lines.append(tuple(b.sym(f"g{c}") for c in classes))
        info.append({"doc": row.get("region", "page"), "sent": r_i, "row": r_i, "row_box": row.get("box")})
        origins.append(list(range(len(classes))))
        for p, gi in enumerate(idx):
            if 0 <= gi < len(glyphs):
                boxes[(ln, p)] = list(glyphs[gi]["box"])
                mi = glyphs[gi].get("match_iou")
                if mi is not None and mi < LOW_MATCH:          # None: 一致度を出さない認識方式（不明として扱う）
                    low.append([ln, p])
    return Corpus(lines=lines, symbols=b.symbols, line_info=info, origins=origins, boxes=boxes,
                  source={"format": "glyph-report", "image": report.get("source_image"),
                          "vision_settings": report.get("settings"), "low_confidence_positions": low})


def from_text(text: str) -> Corpus:
    pub = {"documents": [{"id": "text", "sentences": [ln for ln in text.splitlines() if ln.strip()]}],
           "has_word_boundaries": any(" " in ln.strip() for ln in text.splitlines())}
    c = from_public(pub)
    c.source["format"] = "text"
    return c


def from_lines(lines, names=None) -> Corpus:
    """記号名のリストのリスト（テスト・評価用の簡易入力）。"""
    return from_public({"documents": [{"id": "d", "sentences": [list(map(str, ln)) for ln in lines]}]})


def load(path) -> Corpus:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".txt":
        return from_text(raw)
    data = json.loads(raw)
    if isinstance(data, dict) and str(data.get("schema", "")).startswith("babel-glyph"):
        return from_glyph_report(data)
    if isinstance(data, dict) and "documents" in data:
        return from_public(data)
    raise ValueError(f"{path}: 対応していない形式（BABEL の public コーパス、glyph-report.json、.txt のいずれか）")


# ---------------------------------------------------------------------------------------------- 人間による訂正

def apply_corrections(c: Corpus, corr: dict) -> Corpus:
    """人間が転写を訂正する。位置（line, pos）は訂正前の転写の番号で指定する。

    corr = {
      "merge":      [["g3", "g7"]],                          # 同じ記号とみなす（先頭の名前に統一）
      "rename":     {"g12": "g4"},
      "substitute": [{"line": 0, "pos": 3, "symbol": "g5"}],  # 1記号の誤分類
      "split":      [{"line": 0, "pos": 3, "into": ["g1", "g2"]}],   # 1文字が結合されていた
      "join":       [{"line": 0, "pos": 3, "length": 2, "symbol": "g9"}],  # 1文字が分割されていた
      "delete":     [{"line": 0, "pos": 5}],                  # ノイズ・汚れ
      "insert":     [{"line": 0, "pos": 5, "symbol": "g2"}],  # 見落とし（pos の前に挿入）
      "uncertain":  [{"line": 0, "pos": 4}]                   # 判断できない記号（仮説の根拠に含まれると警告）
    }
    """
    b = _Builder()
    for s in c.symbols:
        b.sym(s)
    rename = dict(corr.get("rename") or {})
    for group in corr.get("merge") or []:
        for s in group[1:]:
            rename[str(s)] = str(group[0])
    applied = []
    lines = [[c.symbols[i] for i in ln] for ln in c.lines]
    origins = [list(o) for o in c.origins]
    ops = []
    for kind in ("substitute", "split", "join", "delete", "insert"):
        for op in corr.get(kind) or []:
            ops.append((int(op["line"]), int(op["pos"]), kind, op))
    # 同じ行では後ろの位置から適用し、前の位置の番号をずらさない
    for ln, pos, kind, op in sorted(ops, key=lambda x: (x[0], -x[1])):
        if not (0 <= ln < len(lines)) or not (0 <= pos <= len(lines[ln]) - (0 if kind == "insert" else 1)):
            raise ValueError(f"訂正 {kind} {op}: 行・位置が範囲外")
        row, org = lines[ln], origins[ln]
        if kind == "substitute":
            row[pos] = str(op["symbol"])
        elif kind == "split":
            into = [str(x) for x in op["into"]]
            row[pos:pos + 1] = into
            org[pos:pos + 1] = [org[pos]] * len(into)
        elif kind == "join":
            n = int(op.get("length", 2))
            row[pos:pos + n] = [str(op["symbol"])]
            org[pos:pos + n] = [org[pos]]
        elif kind == "delete":
            del row[pos]
            del org[pos]
        elif kind == "insert":
            row.insert(pos, str(op["symbol"]))
            org.insert(pos, org[pos] if pos < len(org) else (org[-1] if org else 0))
        applied.append({"kind": kind, **op})
    if rename:
        applied.append({"kind": "rename", "map": rename})
    unc = set(c.uncertain)
    for u in corr.get("uncertain") or []:
        unc.add((int(u["line"]), int(u["pos"])))
        applied.append({"kind": "uncertain", **u})
    new_lines = [tuple(b.sym(rename.get(s, s)) for s in row) for row in lines]
    used = sorted({i for ln in new_lines for i in ln})
    remap = {old: k for k, old in enumerate(used)}
    symbols = [b.symbols[i] for i in used]
    out = Corpus(lines=[tuple(remap[i] for i in ln) for ln in new_lines], symbols=symbols, line_info=c.line_info,
                 origins=origins, boxes=c.boxes, given_boundaries=None, uncertain=unc, source=c.source,
                 corrections_applied=c.corrections_applied + applied)
    return out
