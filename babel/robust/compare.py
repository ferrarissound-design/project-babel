"""転写の比較（babel transcription-compare）と、画像1枚の解析（babel robust-structure）の入出力と Markdown。"""
from __future__ import annotations

import json
from pathlib import Path

from ..structure import corpus as sc
from . import crossalign, rowsim

CFG = {"permutations": 4999, "seed": 0}


def load_rows(path):
    """コーパス（BABEL の public 形式・glyph-report・.txt）を、行ごとのクラス番号の列にする。"""
    c = sc.load(path)
    return [list(ln) for ln in c.lines], [dict(i) for i in c.line_info]


def _top(res, n=6):
    return [{k: p[k] for k in ("rows", "score", "z", "p_raw", "p_adj", "significant")} for p in res["pairs"][:n]]


def compare(path_a, path_b, extra=(), cfg=None):
    """2つの転写（同じ画像、行が1対1に対応）を突き合わせる。extra は行の分け方が違ってもよい転写で、単独の検定だけを行う。"""
    cfg = {**CFG, **(cfg or {})}
    A, info_a = load_rows(path_a)
    B, _ = load_rows(path_b)
    out = compare_rows(A, B, info_a, cfg)
    out["a"], out["b"] = str(path_a), str(path_b)
    for path in extra:
        X, _ = load_rows(path)
        r = rowsim.row_similarity(X, cfg=cfg)
        out["extra"][str(path)] = {"stats": _stats(X), "top": _top(r), "pairs": [p for p in r["pairs"] if p["significant"]]}
    return out


def compare_rows(A, B, info_a=None, cfg=None):
    """compare の本体（行ごとのクラス番号の列を直接受け取る）。"""
    cfg = {**CFG, **(cfg or {})}
    info_a = info_a or [{} for _ in A]
    al = crossalign.align(A, B)
    br = crossalign.bracket(A, B, al)
    tests = {"a": rowsim.row_similarity(A, cfg=cfg), "b": rowsim.row_similarity(B, cfg=cfg),
             "fine": rowsim.row_similarity(br["fine"], cfg=cfg), "coarse": rowsim.row_similarity(br["coarse"], cfg=cfg)}
    column = {k: rowsim.row_similarity(X, cfg={**cfg, "null": "column"}) for k, X in (("a", A), ("b", B))}
    sig = {k: set(rowsim.significant_pairs(v)) for k, v in tests.items()}
    union = sorted(set().union(*sig.values()))
    verdicts = []
    for p in union:
        where = [k for k in ("a", "b", "fine", "coarse") if p in sig[k]]
        verdicts.append({"rows": list(p), "significant_in": where,
                         "column_null": [k for k in ("a", "b") if p in set(rowsim.significant_pairs(column[k]))],
                         "class": "どの転写でも有意" if len(where) == 4 else "両方の転写で有意" if {"a", "b"} <= set(where)
                         else "一方の転写だけで有意"})
    out = {"schema": "babel-transcription-compare-v1", "a": None, "b": None, "config": cfg,
           "stats": {"a": _stats(A), "b": _stats(B), "fine_classes": br["fine_classes"], "coarse_classes": br["coarse_classes"]},
           "alignment": {"anchor_rows": al["anchor_rows"],
                         "columns": [[[list(pa), list(pb)] for pa, pb in cs] for cs in al["columns"]],
                         "rows": [{"a": A[k], "b": B[k], "doc": info_a[k].get("doc")} for k in range(len(A))]},
           "pair_agreement": {"anchor_rows": crossalign.pair_agreement(A, B, al, set(al["anchor_rows"])),
                              "all_rows": crossalign.pair_agreement(A, B, al)},
           "hapax": crossalign.hapax_check(A, B, al),
           "tests": {k: {**{x: v[x] for x in ("statistic", "null", "correction", "permutations", "alpha", "min_score", "n_pairs")},
                         "top": _top(v), "pairs": [p for p in v["pairs"] if p["significant"]]} for k, v in tests.items()},
           "sensitivity_column_null": {k: {"top": _top(v), "pairs": [p for p in v["pairs"] if p["significant"]]} for k, v in column.items()},
           "verdicts": verdicts, "pair_details": [pair_detail(A, B, al, *v["rows"]) for v in verdicts], "extra": {}}
    return out


def _shared_ngrams(a, b, sizes=(2, 3, 4)):
    """2つの行が共有する 2〜4 記号列の種類の数（既存の linepair 族が数えるもの）。"""
    def grams(r):
        return {tuple(r[k:k + n]) for n in sizes for k in range(len(r) - n + 1)}
    return len(grams(a) & grams(b))


def pair_detail(A, B, al, i, j):
    """行 i と行 j の対応する記号を、もう一方の転写がどう見たか（同じ／別のクラス／分割・結合）。"""
    def eq(x, y):
        return 1.0 if x == y else 0.0

    def view(X, Y, side):
        maps = []
        for row in (i, j):
            m = {}
            for pa, pb in al["columns"][row]:
                src, dst = (pa, pb) if side == "a" else (pb, pa)
                for q in src:
                    m[q] = (dst, len(src))
            maps.append(m)
        out = []
        for pi, pj, _ in rowsim.lcs_alignment(X[i], X[j], eq):
            (di, ni), (dj, nj) = maps[0].get(pi, ((), 1)), maps[1].get(pj, ((), 1))
            if len(di) == 1 and len(dj) == 1 and ni == 1 and nj == 1:
                other = "same" if Y[i][di[0]] == Y[j][dj[0]] else "different"
            elif not di or not dj:
                other = "missing"
            else:
                other = "split_or_merged"
            out.append({"positions": [pi, pj], "class": X[i][pi], "other": other,
                        "other_classes": [[Y[i][q] for q in di], [Y[j][q] for q in dj]]})
        return out
    return {"rows": [i, j], "a_matches": view(A, B, "a"), "b_matches": view(B, A, "b"),
            "shared_ngrams_a": _shared_ngrams(A[i], A[j]), "shared_ngrams_b": _shared_ngrams(B[i], B[j]),
            "row_a": [A[i], A[j]], "row_b": [B[i], B[j]]}


def _stats(rows):
    from collections import Counter
    c = Counter(x for r in rows for x in r)
    return {"rows": len(rows), "symbols": sum(len(r) for r in rows), "classes": len(c), "hapax": sum(1 for v in c.values() if v == 1),
            "row_lengths": [len(r) for r in rows]}


def _fmt_pairs(top):
    return "、".join(f"({p['rows'][0]}, {p['rows'][1]}) 一致 {p['score']:g}・補正後 p={p['p_adj']:.4f}{'＊' if p['significant'] else ''}" for p in top)


def compare_markdown(res, name_a="A", name_b="B"):
    L = []
    w = L.append
    sa, sb = res["stats"]["a"], res["stats"]["b"]
    w(f"# 転写の比較（{name_a} と {name_b}）\n")
    w("同じ画像の2つの転写を、クラス番号に頼らずに突き合わせた結果。**画像は使っていない。正解の転写もない。**"
      "行の番号は 0 から数える。記号の位置（枠）がないので、長さが違う行の対応は推定。\n")
    w("| 転写 | 行 | 記号 | クラス | 1回だけのクラス |\n|---|---:|---:|---:|---:|")
    w(f"| {name_a} | {sa['rows']} | {sa['symbols']} | {sa['classes']} | {sa['hapax']} |")
    w(f"| {name_b} | {sb['rows']} | {sb['symbols']} | {sb['classes']} | {sb['hapax']} |\n")
    w("## 1. 行ごとの対応\n")
    w(f"長さが同じ行（位置ごとに1対1と仮定）: {res['alignment']['anchor_rows']}。それ以外の行は、長さが同じ行から学んだクラスの対応を使った推定。\n")
    w(f"表記: `{name_a} のクラス={name_b} のクラス`。`a=b,c` は一方の1記号が他方の2記号に当たる（分割・結合の候補）。`-` は相手がない。\n")
    for k, cs in enumerate(res["alignment"]["columns"]):
        r = res["alignment"]["rows"][k]
        cells = []
        for pa, pb in cs:
            cells.append(f"{','.join(str(r['a'][i]) for i in pa) or '-'}={','.join(str(r['b'][j]) for j in pb) or '-'}")
        w(f"- 行 {k}（{r['doc']}）: `{'  '.join(cells)}`")
    n12 = sum(1 for cs in res["alignment"]["columns"] for pa, pb in cs if len(pa) != len(pb) and pa and pb)
    n0 = sum(1 for cs in res["alignment"]["columns"] for pa, pb in cs if not pa or not pb)
    w(f"\n1記号と2記号が対応する箇所: {n12}、相手のない記号: {n0}（どちらも推定。どちらの転写が正しいかは分からない）\n")
    w("## 2. 「同じ記号とみなした位置の組」は一致するか\n")
    w(f"| 範囲 | 位置 | 位置の組 | 両方が同じ | {name_a} だけが同じ | {name_b} だけが同じ | 両方が別 | 調整ランド指数 "
      f"| {name_a} の「同じ」を {name_b} も認める率 | {name_b} の「同じ」を {name_a} も認める率 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, label in (("anchor_rows", "長さが同じ行だけ"), ("all_rows", "全行（推定の対応を含む）")):
        p = res["pair_agreement"][key]
        w(f"| {label} | {p['positions']} | {p['pairs']} | {p['same_in_both']} | {p['same_only_in_a']} | {p['same_only_in_b']} | {p['different_in_both']} "
          f"| {p['adjusted_rand']} | {p['a_same_confirmed_by_b']} | {p['b_same_confirmed_by_a']} |")
    h = res["hapax"]
    w(f"\n1回だけのクラス: {name_a} {h['a_hapax_classes']}、{name_b} {h['b_hapax_classes']}。1対1に対応した位置のうち、"
      f"{name_a} で1回だけの {h['a_hapax_aligned']} 個中 {h['a_hapax_repeated_in_b']} 個は {name_b} では繰り返すクラス、"
      f"{name_b} で1回だけの {h['b_hapax_aligned']} 個中 {h['b_hapax_repeated_in_a']} 個は {name_a} では繰り返すクラス。"
      f"**両方で1回だけ: {h['hapax_in_both']} 個**\n")
    w("## 3. 行の組の類似（順序を保った共通部分列の検定）\n")
    t = res["tests"]["a"]
    w(f"帰無モデル: 全体シャッフル、置換 {t['permutations']} 回、多重検定: {t['correction']}（行の組 {t['n_pairs']} 個）、有意水準 {t['alpha']}、"
      f"支持の下限: 一致 {t['min_score']:g} 記号。＊ = 有意\n")
    w("| 転写 | クラス | 上位の行の組 |\n|---|---:|---|")
    w(f"| {name_a} | {sa['classes']} | {_fmt_pairs(res['tests']['a']['top'][:5])} |")
    w(f"| {name_b} | {sb['classes']} | {_fmt_pairs(res['tests']['b']['top'][:5])} |")
    w(f"| 細かい側（両方が同じとみなした位置だけ同じ） | {res['stats']['fine_classes']} | {_fmt_pairs(res['tests']['fine']['top'][:5])} |")
    w(f"| 粗い側（どちらかが同じとみなした位置は同じ） | {res['stats']['coarse_classes']} | {_fmt_pairs(res['tests']['coarse']['top'][:5])} |")
    for path, x in res["extra"].items():
        w(f"| {Path(path).stem}（行の分け方が違う。単独） | {x['stats']['classes']} | {_fmt_pairs(x['top'][:5])} |")
    w("\n### 判定\n")
    w("| 行の組 | 有意だった転写 | 分類 | 位置を保つ帰無モデルでも有意な転写 |\n|---|---|---|---|")
    names = {"a": name_a, "b": name_b, "fine": "細かい側", "coarse": "粗い側"}
    for v in res["verdicts"]:
        w(f"| ({v['rows'][0]}, {v['rows'][1]}) | {'、'.join(names[k] for k in v['significant_in'])} | {v['class']} "
          f"| {'、'.join(names[k] for k in v['column_null']) or '—'} |")
    w("\n位置を保つ帰無モデル（感度分析）: 行の先頭から数えた位置が同じ記号どうしだけを入れ替える。"
      "「どの行も同じ書き出しで始まる」ような位置の決まりだけで行が似て見える場合は、このモデルでは有意にならない。")
    w("\n### 行の組ごとの内訳: 一方の転写で対応した記号を、もう一方はどう見たか\n")
    word = {"same": "同じクラス", "different": "**別のクラス**", "split_or_merged": "**分割・結合**", "missing": "相手なし"}
    for d in res["pair_details"]:
        i, j = d["rows"]
        w(f"**行 ({i}, {j})** — 共有する 2〜4 記号列（既存の検定が数えるもの）: {name_a} {d['shared_ngrams_a']} 種、{name_b} {d['shared_ngrams_b']} 種\n")
        w(f"- {name_a}: 行 {i} `{' '.join(map(str, d['row_a'][0]))}` / 行 {j} `{' '.join(map(str, d['row_a'][1]))}`")
        w(f"- {name_b}: 行 {i} `{' '.join(map(str, d['row_b'][0]))}` / 行 {j} `{' '.join(map(str, d['row_b'][1]))}`")
        for side, nm, other in (("a_matches", name_a, name_b), ("b_matches", name_b, name_a)):
            cells = [f"{m['class']}→{word[m['other']]}" for m in d[side]]
            w(f"- {nm} で対応した記号（{len(cells)} 個）を {other} で見ると: {'、'.join(cells) or '—'}")
        w("")
    w("- 「どの転写でも有意」な組は、2つの方式の分類の食い違いをどちらに倒しても残る。ただし2つの方式に共通する誤りが作った可能性は、"
      "画像か正解の転写がないと除けない")
    w("- 細かい側・粗い側は、2つの転写の「同じ／違う」の食い違いを挟む2つの極。切り出し（記号の区切り）は A のまま")
    return "\n".join(L) + "\n"


def image_markdown(res):
    L = []
    w = L.append
    w(f"# 認識方式をまたいだ構造の検証（{res['image']}）\n")
    w("1枚の画像を複数の方式で転写し、行の組の類似（順序を保った共通部分列の検定）を方式ごとに行った。"
      f"行の番号は基準の方式（{res['reference']}）の行（0 から）。**正解の転写は使っていない。**\n")
    w("| 方式 | 行 | 記号 | クラス | 有意な行の組（その方式の行の番号） |\n|---|---:|---:|---:|---|")
    for n in res["members"]:
        t = res["transcriptions"][n]
        sig = [p for p in res["tests"][n]["pairs"] if p["significant"]]
        w(f"| {n} | {len(t['rows'])} | {sum(len(r['classes']) for r in t['rows'])} | {(t.get('stats') or {}).get('visual_classes')} "
          f"| {'、'.join('(%d, %d)' % tuple(p['rows']) for p in sig) or '—'} |")
    w("\n## 方式をまたいで再現する行の組\n")
    groups = sorted({g for v in res["votes"] for g in v.get("groups", [])} | {"baseline", "improved-v2", "improved-v3"})
    w(f"| 行の組（基準の行） | 有意だった切り出し群の数（/{len(groups)}） | 切り出し群 | 有意だった方式の数（/{len(res['members'])}） | 方式 "
      "| 分類を経ない検定 |\n|---|---:|---|---:|---|---|")
    soft = {tuple(p["rows"]): p for p in res.get("soft", {}).get("pairs", [])}
    for v in res["votes"]:
        s = soft.get(tuple(v["rows"]))
        w(f"| ({v['rows'][0]}, {v['rows'][1]}) | {v['votes']} | {'、'.join(v.get('groups', []))} | {v.get('raw_votes', len(v['members']))} "
          f"| {'、'.join(v['members'])} | "
          f"{'—' if s is None else ('有意' if s['significant'] else '有意でない') + '（補正後 p=%.4f）' % s['p_adj']} |")
    if not res["votes"]:
        w("| — | 0 | — | 0 | — | — |")
    w("\n読み方（docs/ROBUST_STRUCTURE.md）: 票は**切り出し群**の数で数える。同じ切り出しを使い分類だけを変えた方式は、まとめて1群"
      "（方式の数は参考）。全部の群で有意な組ほど、特定の切り出しの誤りに依存していない。一部の群でしか有意でない組は保留とし、"
      "規則として扱う前に画像で照合すること。全部の群で有意でも、意味・文法機能は分からない。")
    return "\n".join(L) + "\n"


def write(res, out_json, markdown):
    p = Path(out_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    p.with_suffix(".md").write_text(markdown, encoding="utf-8")
