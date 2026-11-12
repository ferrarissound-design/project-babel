"""実画像の評価パイプライン: python -m babel image-study prepare|status|evaluate|structure STUDY_DIR

画像を受け取ったときにすぐ始められるように、次の手順を1つのディレクトリ（STUDY_DIR）にまとめる。

1. prepare   画像を既存方式（baseline）と改善方式（improved, improved-v2）で認識し、比較の画像と JSON を作る。
             人が直すための正解転写の下書き（gt.json、すべて未確認・label なし）を作る。既にあれば上書きしない
2. （人）    python -m babel annotate IMG --gt STUDY_DIR/gt.json で、枠・分割・結合・字形名を直し、確認済みにする
3. status    正解転写の確認の進み具合（評価できる範囲）
4. evaluate  両方式の認識結果を正解転写と比べる（F1・分割・結合・ARI・NMI・行）。確認済みがなければ「測定不能」
5. structure 正解転写（A）・既存方式（B）・改善方式（C = improved、C2 = improved-v2）を同じ設定の構造解析にかけ、
             仮説を画像上の位置で照合して比べる。
             確認済みの正解がなければ「評価待ち」として結果を作らない

画像はリポジトリに入れない（study.json に画像のハッシュと大きさを記録し、別の画像で続けると止める）。
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import groundtruth as G

METHODS = ("baseline", "improved", "improved-v2")
LETTERS = {"baseline": "B", "improved": "C", "improved-v2": "C2"}
PENDING = "評価待ち"
STRUCTURE_CFG = {"permutations": 999, "subsamples": 10, "perturbations": 6, "seed": 0}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class StudyImageMismatch(ValueError):
    """study ディレクトリの記録（study.json・gt.json）と別の画像を使おうとした。"""


def _check_same_image(out, sha, size):
    """既存の study.json・gt.json が同じ画像のものか。違えば何も書かずに止める（Codex レビュー、PR #11）。"""
    sp, gp = out / "study.json", out / "gt.json"
    if sp.exists():
        old = _read(sp).get("image_sha256")
        if old != sha:
            raise StudyImageMismatch(f"{sp} は別の画像（sha256 {old}）の記録。新しい画像（{sha}）で上書きしない。"
                                     "別の study ディレクトリを使う")
    if gp.exists():
        g = _read(gp)
        if g.get("image_sha256") not in (None, sha):
            raise StudyImageMismatch(f"{gp} は別の画像（sha256 {g.get('image_sha256')}）の正解転写。上書きしない")
        if g.get("image_size") and list(g["image_size"]) != list(size):
            raise StudyImageMismatch(f"{gp} の画像の大きさ {g['image_size']} が新しい画像 {list(size)} と違う。上書きしない")


def check_study_consistency(study_dir):
    """評価の前に、正解転写と認識結果が同じ画像のものかを確かめる（gt.json に記録したハッシュと study.json）。"""
    st = _study(study_dir)
    g = _read(Path(study_dir) / "gt.json")
    if g.get("image_sha256") not in (None, st.get("image_sha256")):
        raise StudyImageMismatch(f"gt.json（sha256 {g.get('image_sha256')}）と認識結果（study.json の sha256 "
                                 f"{st.get('image_sha256')}）が別の画像。評価しない")
    if g.get("image_size") and st.get("image_size") and list(g["image_size"]) != list(st["image_size"]):
        raise StudyImageMismatch(f"gt.json の画像の大きさ {g['image_size']} と認識結果 {st['image_size']} が違う。評価しない")
    return st


def _study(study_dir):
    p = Path(study_dir) / "study.json"
    if not p.exists():
        raise FileNotFoundError(f"{study_dir} に study.json がない。先に prepare を実行する")
    return _read(p)


# ------------------------------------------------------------------------------------------------ 1. prepare

def _draw(image, report, path, title):
    import cv2
    img = cv2.imread(str(image), cv2.IMREAD_COLOR)
    for g in report["glyphs"]:
        x1, y1, x2, y2 = g["box"]
        cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), (255, 140, 30), 1)
        cv2.putText(img, str(g["class_id"]), (x1, max(8, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (40, 40, 230), 1, cv2.LINE_AA)
    cv2.putText(img, title, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 120, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)
    return img


def compare_methods(rep_a, rep_b):
    """2つの認識結果の一致（枠の1対1の対応と、対応した記号どうしのクラスの構造）。正解は使わない。"""
    from .evaluate import adjusted_rand, match, nmi
    pa, pb = [g["box"] for g in rep_a["glyphs"]], [g["box"] for g in rep_b["glyphs"]]
    pairs = match(pa, pb)
    ca = [rep_a["glyphs"][i]["class_id"] for i, _, _ in pairs]
    cb = [rep_b["glyphs"][j]["class_id"] for _, j, _ in pairs]
    return {"glyphs": [len(pa), len(pb)], "rows": [len(rep_a["rows"]), len(rep_b["rows"])],
            "classes": [len({g["class_id"] for g in rep_a["glyphs"]}), len({g["class_id"] for g in rep_b["glyphs"]})],
            "matched_boxes": len(pairs), "only_in_first": len(pa) - len(pairs), "only_in_second": len(pb) - len(pairs),
            "class_agreement_ari": round(adjusted_rand(ca, cb), 4) if len(ca) > 1 else None,
            "class_agreement_nmi": round(nmi(ca, cb), 4) if ca else None,
            "note": "2つの方式どうしの一致で、正しさではない"}


def prepare(image, study_dir, regions=None):
    import cv2
    import numpy as np

    from .recognizers import recognize
    image = Path(image)
    if not image.exists():
        raise FileNotFoundError(f"画像がない: {image}（画像なしで処理を続けない）")
    out = Path(study_dir)
    sha = _sha256(image)
    gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"画像を読めない: {image}")
    h, w = gray.shape[:2]
    _check_same_image(out, sha, (w, h))             # 何かを書き始める前に照合する
    out.mkdir(parents=True, exist_ok=True)
    reports = {}
    for m in METHODS:
        rep = recognize(image, m, regions=regions)
        _write(out / m / "glyph-report.json", rep)
        reports[m] = rep
    imgs = [_draw(image, reports[m], out / m / "annotated.png", m) for m in METHODS]
    cv2.imwrite(str(out / "comparison.png"), np.vstack(imgs))
    cmp = {f"{x}_vs_{y}": compare_methods(reports[x], reports[y])
           for x, y in (("baseline", "improved"), ("baseline", "improved-v2"), ("improved", "improved-v2"))}
    _write(out / "methods-comparison.json", cmp)
    gt_path = out / "gt.json"
    created = False
    if not gt_path.exists():
        draft = G.draft_from_report(reports["improved"], image.name)
        draft["image_sha256"] = sha                 # 正解転写がどの画像のものか（評価の前に照合する）
        # 既存方式の案も proposed に残す（どちらも正解ではない）
        from .evaluate import match
        base = reports["baseline"]["glyphs"]
        pairs = {i: j for i, j, _ in match([it["box"] for it in draft["items"]], [g["box"] for g in base])}
        for i, it in enumerate(draft["items"]):
            it["proposed"] = {"improved": it["proposed"]["class"],
                              "baseline": base[pairs[i]]["class_id"] if i in pairs else None,
                              "by": "babel-vision（どちらも未確認の案）"}
        G.save(draft, gt_path)
        created = True
    st = {"schema": "babel-image-study-v1", "image": image.name, "image_sha256": sha, "image_size": [w, h],
          "regions": regions, "methods": list(METHODS), "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "gt_created_by_prepare": created, "opencv": cv2.__version__}
    _write(out / "study.json", st)
    return {"study": st, "methods": cmp, "gt": str(gt_path), "gt_created": created}


# ------------------------------------------------------------------------------------------------ 3. status

def status(study_dir):
    if not (Path(study_dir) / "gt.json").exists():
        raise FileNotFoundError(f"{study_dir}/gt.json がない。先に prepare を実行する")
    gt = G.load(Path(study_dir) / "gt.json")
    st = G.status(gt)
    view = G.evaluation_view(gt)
    complete = st["glyphs"] > 0 and st["verified_glyphs"] == st["glyphs"] and st["verified_lines"] == st["lines"] and \
        all(i["verified"] for i in gt["items"])
    unlabeled = sum(1 for i in gt["items"] if i["verified"] and i["kind"] == "glyph" and not i["label"])
    state = "complete" if complete and not unlabeled else ("partial" if st["verified_glyphs"] else "not_started")
    return {"state": state, **st, "verified_glyphs_without_label": unlabeled, "scope": view["scope"],
            "note": {"complete": "すべての記号と行を人が確認した。評価と構造解析の比較ができる",
                     "partial": "一部だけ確認済み。評価は確認済みの範囲に限る。構造解析の比較は参考値",
                     "not_started": "人が確認した記号がない。評価は測定不能、構造解析の比較は評価待ち"}[state]}


# ------------------------------------------------------------------------------------------------ 4. evaluate

def evaluate_study(study_dir):
    from .evaluate import evaluate
    check_study_consistency(study_dir)
    gt = G.load(Path(study_dir) / "gt.json")
    out = {"status": status(study_dir), "methods": {}}
    for m in METHODS:
        rep = _read(Path(study_dir) / m / "glyph-report.json")
        out["methods"][m] = evaluate(rep, gt)
    _write(Path(study_dir) / "evaluation.json", out)
    return out


# ------------------------------------------------------------------------------------------------ 5. structure

def gt_report(gt):
    """確認済みの正解転写を glyph-report の形にする（行 = 確認済みの行、記号 = 確認済みの文字を左から、クラス = 字形名）。"""
    al = G.aliases(gt)
    lines = sorted([ln for ln in gt["lines"] if ln["verified"]], key=lambda ln: (ln["box"][1], ln["box"][0]))
    glyphs, rows = [], []
    for ln in lines:
        its = sorted([i for i in gt["items"] if i["verified"] and i["kind"] == "glyph" and i["label"] and not i["uncertain"]
                      and i["line"] == ln["id"]], key=lambda i: i["box"][0])
        idx = []
        for it in its:
            glyphs.append({"box": it["box"], "class_id": G.canonical_label(it, al), "match_iou": None, "gt_id": it["id"]})
            idx.append(len(glyphs) - 1)
        if idx:
            rows.append({"region": "gt", "box": ln["box"], "classes": [glyphs[i]["class_id"] for i in idx], "glyph_indices": idx,
                         "gt_line": ln["id"]})
    return {"schema": "babel-glyph-pilot-v1", "source": "ground-truth", "rows": rows, "glyphs": glyphs}


def _translation(rep, gt):
    """予測のクラス → 正解の字形名（位置で対応した確認済みの記号の多数決）。評価にだけ使う。"""
    from .evaluate import match
    al = G.aliases(gt)
    V = [i for i in gt["items"] if i["verified"] and i["kind"] == "glyph" and i["label"]]
    pairs = match([v["box"] for v in V], [g["box"] for g in rep["glyphs"]])
    votes = defaultdict(Counter)
    for i, j, _ in pairs:
        votes[f"g{rep['glyphs'][j]['class_id']}"][f"g{G.canonical_label(V[i], al)}"] += 1
    return {k: {"label": c.most_common(1)[0][0], "votes": dict(c)} for k, c in votes.items()}


def _line_split(rep, gt):
    """予測の行と、確認済みの正解の行の対応（行の分け方の違い）。
    所属する行が確認済みの記号だけを使う（未確認の行の分け方は編集途中なので数えない。Codex レビュー、PR #11）。"""
    from .evaluate import match
    vlines = {ln["id"] for ln in gt["lines"] if ln["verified"]}
    V = [i for i in gt["items"] if i["verified"] and i["kind"] == "glyph" and i["line"] in vlines]
    line_of = {}
    pairs = match([v["box"] for v in V], [g["box"] for g in rep["glyphs"]])
    for i, j, _ in pairs:
        line_of[j] = V[i]["line"]
    rows_multi, per_gt = 0, defaultdict(set)
    for r, row in enumerate(rep["rows"]):
        ls = {line_of[j] for j in row["glyph_indices"] if j in line_of}
        rows_multi += len(ls) > 1
        for ln in ls:
            per_gt[ln].add(r)
    return {"predicted_rows": len(rep["rows"]), "verified_gt_lines": sum(1 for ln in gt["lines"] if ln["verified"]),
            "rows_spanning_several_gt_lines": rows_multi,
            "gt_lines_split_into_several_rows": sum(1 for v in per_gt.values() if len(v) > 1)}


def _boundary_agreement(a_res, auto_res, gt_rep, auto_rep, gt):
    """単語境界の候補（合議）を、正解転写の位置に写して比べる（正解の境界はないので、A との一致）。"""
    from babel.structure.boundaries import boundary_f1

    from .evaluate import match
    def cuts(res):
        for c in res["boundaries"]["candidates"]:
            if c["method"] == "consensus":
                return c["cuts"]
        return []
    pos = {}                                         # 予測の記号の番号 → (A の行, A の行の中の位置)
    a_index = {}
    for r, row in enumerate(gt_rep["rows"]):
        for k, gi in enumerate(row["glyph_indices"]):
            a_index[gi] = (r, k)
    pairs = match([g["box"] for g in gt_rep["glyphs"]], [g["box"] for g in auto_rep["glyphs"]])
    for i, j, _ in pairs:
        pos[j] = a_index[i]
    a_cuts = [set(c) for c in cuts(a_res)]
    mapped = [set() for _ in a_cuts]
    for row, cs in zip(auto_rep["rows"], cuts(auto_res)):
        idx = row["glyph_indices"]
        for c in cs:
            if 0 < c < len(idx) and idx[c] in pos:
                r, k = pos[idx[c]]
                if k > 0 and r < len(mapped):
                    mapped[r].add(k)
    return {"consensus_cut_f1_vs_A": round(boundary_f1(mapped, a_cuts), 4) if a_cuts else None,
            "note": "正解の単語境界はない。A（正解の転写）の合議の境界との一致で、境界の正しさではない"}


def _hyp(h, translation=None):
    """仮説の要約。translation があれば、自動転写の記号に対応する正解の字形名を添える（評価用の読み替え）。"""
    d = {"id": h["id"], "family": h["family"], "tier": h["tier"], "observation": h["observation"],
         "symbols": h.get("symbols") or [], "p_family_corrected": h["statistics"]["p_family_corrected"],
         "p_raw": h["statistics"]["p_raw"], "stability": h["stability"]}
    if translation is not None:
        d["symbols_as_gt"] = [translation.get(x, {}).get("label", "?（正解と対応しない）") for x in d["symbols"]]
    return d


def structure_study(study_dir, cfg=None):
    from babel.structure.analyze import analyze
    from babel.structure.corpus import from_glyph_report
    from babel.structure.transcription import same_hypothesis
    cfg = dict(STRUCTURE_CFG, **(cfg or {}))
    check_study_consistency(study_dir)
    gt = G.load(Path(study_dir) / "gt.json")
    stt = status(study_dir)
    if stt["state"] == "not_started":
        res = {"status": PENDING, "reason": "人が確認した正解の記号がない。正解転写（A）がないので、構造解析の比較はしない", "gt": stt}
        _write(Path(study_dir) / "structure-comparison.json", res)
        return res
    gt_rep = gt_report(gt)
    if sum(len(r["glyph_indices"]) for r in gt_rep["rows"]) < 4:
        res = {"status": PENDING, "reason": "確認済みで字形名のある記号が4つ未満", "gt": stt}
        _write(Path(study_dir) / "structure-comparison.json", res)
        return res
    reps = {"A": gt_rep}
    for m in METHODS:
        reps[LETTERS[m]] = _read(Path(study_dir) / m / "glyph-report.json")
    analyses = {}
    for key, rep in reps.items():
        try:
            analyses[key] = analyze(from_glyph_report(rep), cfg)
        except ValueError as e:
            analyses[key] = {"error": str(e), "hypotheses": []}
    out = {"status": "measured" if stt["state"] == "complete" else "partial", "gt": stt, "config": cfg,
           "settings": "A・B・C とも同じ構造解析（babel.structure）と同じ設定。入力の転写だけが違う", "transcriptions": {}}
    for key, a in analyses.items():
        if "error" in a:
            out["transcriptions"][key] = {"error": a["error"]}
            continue
        out["transcriptions"][key] = {"symbols": a["statistics"]["symbols_total"], "classes": a["statistics"]["symbol_types"],
                                      "lines": a["statistics"]["lines"], "verdict": a["verdict"]["code"],
                                      "hypotheses": [_hyp(h) for h in a["hypotheses"]],
                                      "global_tests": {k: g["p_holm"] for k, g in a["global_tests"].items()}}
    out["comparison"] = {}
    if "error" not in analyses["A"]:
        for key in [LETTERS[m] for m in METHODS]:
            a, b = analyses["A"], analyses[key]
            if "error" in b:
                out["comparison"][key] = {"error": b["error"]}
                continue
            tr = _translation(reps[key], gt)
            found = [h for h in a["hypotheses"] if any(same_hypothesis(h, g) for g in b["hypotheses"])]
            missed = [h for h in a["hypotheses"] if h not in found]
            only = [g for g in b["hypotheses"] if not any(same_hypothesis(h, g) for h in a["hypotheses"])]
            out["comparison"][key] = {
                "A_hypotheses": len(a["hypotheses"]), "auto_hypotheses": len(b["hypotheses"]),
                "reproduced_from_A": [_hyp(h) for h in found], "missed_from_A": [_hyp(h) for h in missed],
                "only_in_auto": [_hyp(h, tr) for h in only],
                "class_translation": tr, "line_split": _line_split(reps[key], gt),
                "boundaries": _boundary_agreement(a, b, gt_rep, reps[key], gt),
                "matching": "仮説は画像上の位置（根拠の枠）と族ごとの鍵で照合する（クラス番号は転写ごとに違うので使わない）"}
    if out["status"] == "partial":
        out["warning"] = "正解転写が一部だけ確認済み。A は確認済みの範囲だけで、B・C は画像全体なので、比較は参考値"
    _write(Path(study_dir) / "structure-comparison.json", out)
    Path(study_dir, "structure-comparison.md").write_text(markdown(out), encoding="utf-8")
    return out


def markdown(res):
    L = ["# 実画像の構造解析の比較", "", f"状態: **{res['status']}**", ""]
    if res.get("warning"):
        L += [f"> {res['warning']}", ""]
    L += ["| 転写 | 記号 | クラス | 行 | 判定 | 仮説 |", "|---|---:|---:|---:|---|---:|"]
    for k, t in res.get("transcriptions", {}).items():
        if "error" in t:
            L.append(f"| {k} | — | — | — | 解析できない: {t['error']} | 0 |")
        else:
            L.append(f"| {k} | {t['symbols']} | {t['classes']} | {t['lines']} | {t['verdict']} | {len(t['hypotheses'])} |")
    for k, c in res.get("comparison", {}).items():
        if "error" in c:
            continue
        L += ["", f"## A と {k}", "",
              f"- A の仮説 {c['A_hypotheses']} 件のうち {k} でも見つかった {len(c['reproduced_from_A'])} 件、見逃した {len(c['missed_from_A'])} 件",
              f"- {k} だけに現れた仮説 {len(c['only_in_auto'])} 件",
              f"- 行の分け方: 予測の行 {c['line_split']['predicted_rows']}（確認済みの正解の行 {c['line_split']['verified_gt_lines']}）、"
              f"複数の正解の行にまたがる予測の行 {c['line_split']['rows_spanning_several_gt_lines']}、"
              f"複数の予測の行に分かれた正解の行 {c['line_split']['gt_lines_split_into_several_rows']}",
              f"- 単語境界（合議）の A との一致 F1: {c['boundaries']['consensus_cut_f1_vs_A']}"]
        for h in c["missed_from_A"]:
            L.append(f"  - 見逃し: {h['observation']}（A での補正後 p = {h['p_family_corrected']}）")
        for h in c["only_in_auto"]:
            tr = "、".join(f"{x}≈{y}" for x, y in zip(h["symbols"], h.get("symbols_as_gt", [])))
            L.append(f"  - {k} だけ: {h['observation']}（補正後 p = {h['p_family_corrected']}"
                     + (f"。正解の字形: {tr}" if tr else "") + "）")
    return "\n".join(L) + "\n"
