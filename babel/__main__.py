"""コマンドライン入口:  python -m babel <command>"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="babel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="コーパスを生成（public と secret を別ディレクトリに書く）")
    g.add_argument("--family", default="KES")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--level", default="A")
    g.add_argument("--docs", type=int, default=130)
    g.add_argument("--out", default="data")
    d = sub.add_parser("decipher", help="public コーパスだけを読んで予測 JSON を書く")
    d.add_argument("public")
    d.add_argument("--out", required=True)
    e = sub.add_parser("evaluate", help="予測と正解を比較")
    e.add_argument("pred")
    e.add_argument("gold")
    e.add_argument("--public")
    x = sub.add_parser("experiment", help="実験一式を実行")
    x.add_argument("--mode", choices=["quick", "full"], default="quick")
    x.add_argument("--seeds", help="カンマ区切り。省略時は quick=開発シード, full=評価シード")
    x.add_argument("--jobs", type=int)
    x.add_argument("--only", help="実験IDをカンマ区切りで（例: E1_main,E2_baseline）")
    x.add_argument("--out", default="results/latest.json")
    r = sub.add_parser("report", help="ダッシュボード用 JSON と採点表を生成")
    r.add_argument("--results", default="results/latest.json")
    r.add_argument("--no-showcase", action="store_true")
    t = sub.add_parser("external", help="初見コーパスの予測・採点・台帳検証")
    t.add_argument("action", choices=["predict", "score", "verify"])
    v = sub.add_parser("vision", help="API不要: 画像から記号候補・反復パターンを抽出")
    v.add_argument("image", help="入力画像のパス")
    v.add_argument("--out-dir", default="vision-output", help="分析結果の保存先")
    v.add_argument("--region", action="append", default=[], metavar="ID:X1,Y1,X2,Y2",
                   help="画像の文字領域を指定（複数指定可、右下座標は含まない）")
    v.add_argument("--threshold", type=int, help="二値化のしきい値（省略すると自動）")
    v.add_argument("--similarity", type=float, default=0.7, help="同一形状とみなすIoU閾値")
    v.add_argument("--keep-rules", action="store_true",
                   help="罫線除去をしない（文字の線が切り出し領域の大部分を占める場合）")
    v.add_argument("--method", default="baseline",
                   help="認識の方式（baseline = 従来、improved など。babel/vision/recognizers.py の METHODS）。"
                        "baseline 以外は --threshold / --similarity を使わない（--region は使う）")
    an = sub.add_parser("annotate", help="正解転写を作る・直すローカルの UI（127.0.0.1 のみ）")
    an.add_argument("image")
    an.add_argument("--gt", required=True, help="正解転写 JSON（babel-glyph-gt-v1）。なければ作る")
    an.add_argument("--report", help="下書きに使う glyph-report.json（すべて未確認・label なしで取り込む）")
    an.add_argument("--port", type=int, default=8765)
    an.add_argument("--open", action="store_true", help="ブラウザを開く")
    ve = sub.add_parser("vision-eval", help="認識結果（glyph-report）と正解転写を比べる")
    ve.add_argument("report")
    ve.add_argument("gt")
    ve.add_argument("--iou", type=float, default=0.5)
    ve.add_argument("--out")
    st = sub.add_parser("image-study", help="実画像の評価（両方式で認識 → 人が正解転写を作る → 評価 → 構造解析の比較）")
    st.add_argument("action", choices=["prepare", "status", "evaluate", "structure"])
    st.add_argument("study", help="研究のディレクトリ（画像はここに複写しない）")
    st.add_argument("--image", help="prepare のときの画像")
    st.add_argument("--region", action="append", default=[], metavar="ID:X1,Y1,X2,Y2", help="prepare のときの文字領域")
    st.add_argument("--permutations", type=int, help="structure のときの置換回数（既定 999）")
    vb = sub.add_parser("vision-benchmark", help="人工画像で認識方式を比べる（results/vision/）")
    vb.add_argument("--split", choices=["dev", "val", "test"], default="dev")
    vb.add_argument("--methods", help="カンマ区切り（省略時はすべて）")
    vb.add_argument("--out")
    s = sub.add_parser("analyze", help="意味を使わない構造発見（反復・境界・文法的な規則の仮説）。翻訳はしない")
    s.add_argument("corpus", help="BABEL コーパス JSON / glyph-report.json / .txt / 画像（画像なら vision を内部で実行）")
    s.add_argument("--out", required=True, help="解析結果 JSON（同じ名前の .md も書く）")
    s.add_argument("--md", help="Markdown レポートの保存先（省略時は --out の拡張子を .md に）")
    s.add_argument("--corrections", help="人間による転写の訂正 JSON（babel/structure/corpus.py の apply_corrections 参照）")
    s.add_argument("--alt", action="append", default=[], help="別の転写候補（複数可）。仮説が転写の違いで変わるかを比べる")
    s.add_argument("--similarity", default="0.7", help="画像入力のとき: 形状の類似度の閾値。カンマ区切りで複数指定すると、"
                                                     "先頭を主、残りを別の転写候補として比べる（例: 0.7,0.6,0.8）")
    s.add_argument("--region", action="append", default=[], metavar="ID:X1,Y1,X2,Y2", help="画像入力のとき: 文字領域")
    s.add_argument("--image", help="glyph-report を入力したとき、根拠の位置を描く元画像")
    s.add_argument("--permutations", type=int, help="置換回数（省略時はコーパスの大きさで自動）")
    s.add_argument("--subsamples", type=int, help="部分標本の数（標本の揺れ）")
    s.add_argument("--perturbations", type=int, help="転写の攪乱の回数（0 で無効）")
    s.add_argument("--perturbation-rate", type=float, help="攪乱の割合（既定 0.05）")
    s.add_argument("--seed", type=int, default=0)
    sb = sub.add_parser("structure-benchmark", help="構造解析の評価（人工・外部・実データ）。結果は results/structure/")
    sb.add_argument("--split", choices=["dev", "eval"], default="dev", help="dev: 調整用シード / eval: 最終評価用シード")
    sb.add_argument("--seeds", help="カンマ区切りで上書き")
    sb.add_argument("--sizes", default="5,13,30,100")
    sb.add_argument("--jobs", type=int)
    sb.add_argument("--out", help="結果 JSON（省略時は results/structure/benchmark-<split>.json）")
    sb.add_argument("--quick", action="store_true", help="置換回数を減らした簡易版（テスト用）")
    a = ap.parse_args(argv)

    if a.cmd == "generate":
        from babel.generator.corpus import build_corpus
        pub, sec = build_corpus(a.family, a.seed, a.level, n_docs=a.docs)
        base = Path(a.out) / f"{a.family}-{a.seed}-{a.level}"
        (base / "public").mkdir(parents=True, exist_ok=True)
        (base / "secret").mkdir(parents=True, exist_ok=True)
        (base / "public" / "corpus.json").write_text(json.dumps(pub, ensure_ascii=False), encoding="utf-8")
        (base / "secret" / "answer.json").write_text(json.dumps(sec, ensure_ascii=False), encoding="utf-8")
        n = sum(len(x["sentences"]) for x in pub["documents"])
        print(f"{base}: {len(pub['documents'])} 文書, {n} 文")
    elif a.cmd == "decipher":
        from babel.decipher.engine import decipher
        pred = decipher(json.loads(Path(a.public).read_text(encoding="utf-8")))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(pred, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"verdict": pred["verdict"]["text"], **pred["summary"]}, ensure_ascii=False))
    elif a.cmd == "evaluate":
        from babel.evaluate.score import score
        pub = json.loads(Path(a.public).read_text(encoding="utf-8")) if a.public else None
        res = score(json.loads(Path(a.pred).read_text(encoding="utf-8")),
                    json.loads(Path(a.gold).read_text(encoding="utf-8")), pub)
        res.pop("calibration", None)
        print(json.dumps(res, ensure_ascii=False, indent=1))
    elif a.cmd == "experiment":
        from babel.experiments.run import run_all
        seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None
        res = run_all(a.mode, seeds, a.jobs, set(a.only.split(",")) if a.only else None)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        m = res["meta"]
        with Path("results/history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({k: m[k] for k in ("started_utc", "commit", "mode", "seeds", "seed_role", "environment",
                                                   "run_id", "total_runs", "failed_runs", "seconds", "engine_version")},
                               ensure_ascii=False) + "\n")
        print(json.dumps({**{k: m[k] for k in ("commit", "mode", "seed_role", "total_runs", "failed_runs", "seconds")},
                          "repro_digest": res["reproducibility"]["digests"][0],
                          "repro_identical": res["reproducibility"]["identical"]}, ensure_ascii=False))
        for c in res["conditions"]:
            if c["experiment"] == "E1_main":
                mm = c["metrics"]
                print(c["label"], {k: mm[k]["mean"] for k in ("seg_f1", "pos_accuracy", "numeral_accuracy",
                                                              "lexicon_precision", "gloss_correct", "gloss_wrong")})
        if m["failed_runs"]:
            print(f"失敗した実行が {m['failed_runs']} 件ある", file=sys.stderr)
            return 1
    elif a.cmd == "report":
        from babel.report.build import build
        card = build(a.results, with_showcase=not a.no_showcase)
        print(f"内部テスト: {card['internal_points']} / {card['evaluated_max']} 点（未評価 {card['unevaluated_max']} 点分）")
        for i in card["items"]:
            print(f"  {i['name']}: {i['points']} / {i['max']} [{i['status']}]")
    elif a.cmd == "vision":
        from babel.vision import save_analysis
        regions = []
        for spec in a.region:
            try:
                name, coords = spec.split(":", 1)
                if not name:
                    raise ValueError("empty id")
                box = [int(t) for t in coords.split(",")]
                if len(box) != 4:
                    raise ValueError("expected four integers")
            except ValueError:
                ap.error("--region は ID:X1,Y1,X2,Y2 の形式で指定してください")
            regions.append({"id": name, "box": box})
        if a.method == "baseline":
            data = save_analysis(a.image, a.out_dir, regions=regions or None,
                                 threshold=a.threshold, similarity=a.similarity,
                                 remove_rules=not a.keep_rules)
        else:
            from babel.vision.glyphs import _corpus
            from babel.vision.recognizers import recognize
            data = recognize(a.image, a.method, regions=regions or None)
            out = Path(a.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "glyph-report.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            for spaced, fn in ((False, "corpus-continuous.json"), (True, "corpus-glyph-tokens.json")):
                (out / fn).write_text(json.dumps(_corpus(data, spaced), ensure_ascii=False,
                                                 indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"out_dir": a.out_dir, **data["stats"]}, ensure_ascii=False))
    elif a.cmd == "annotate":
        from babel.vision.annotate import serve
        serve(a.image, a.gt, report=a.report, port=a.port, open_browser=a.open)
    elif a.cmd == "vision-eval":
        from babel.vision.evaluate import evaluate
        from babel.vision.groundtruth import load
        res = evaluate(json.loads(Path(a.report).read_text(encoding="utf-8")), load(a.gt), iou_thr=a.iou)
        text = json.dumps(res, ensure_ascii=False, indent=1)
        if a.out:
            Path(a.out).write_text(text + "\n", encoding="utf-8")
        print(text)
    elif a.cmd == "image-study":
        from babel.vision import study as S
        if a.action == "prepare":
            if not a.image:
                ap.error("prepare には --image が必要")
            res = S.prepare(a.image, a.study, regions=_parse_regions(a.region, ap))
            print(json.dumps({"gt": res["gt"], "gt_created": res["gt_created"], "methods": res["methods"]}, ensure_ascii=False, indent=1))
        elif a.action == "status":
            print(json.dumps(S.status(a.study), ensure_ascii=False, indent=1))
        elif a.action == "evaluate":
            res = S.evaluate_study(a.study)
            print(json.dumps({m: {k: v.get(k) for k in ("status", "detection", "classes", "lines")} for m, v in res["methods"].items()},
                             ensure_ascii=False, indent=1))
        else:
            res = S.structure_study(a.study, {"permutations": a.permutations} if a.permutations else None)
            print(json.dumps({"status": res["status"], "reason": res.get("reason"),
                              "comparison": {k: {x: (len(v[x]) if isinstance(v.get(x), list) else v.get(x))
                                                 for x in ("A_hypotheses", "auto_hypotheses", "reproduced_from_A", "missed_from_A",
                                                           "only_in_auto")} for k, v in res.get("comparison", {}).items()
                                             if "error" not in v}}, ensure_ascii=False, indent=1))
    elif a.cmd == "vision-benchmark":
        from babel.vision_eval.benchmark import run as vrun
        res = vrun(a.split, a.methods.split(",") if a.methods else None, out=a.out or f"results/vision/methods-{a.split}.json")
        print(json.dumps(res["meta"], ensure_ascii=False))
    elif a.cmd == "analyze":
        return _analyze(a, ap)
    elif a.cmd == "structure-benchmark":
        from babel.structure_eval.benchmark import run
        res = run(a.split, [int(x) for x in a.seeds.split(",")] if a.seeds else None,
                  [int(x) for x in a.sizes.split(",")], jobs=a.jobs, quick=a.quick, out=a.out)
        print(json.dumps(res["summary_line"], ensure_ascii=False))
    elif a.cmd == "external":
        from babel import external
        from babel.experiments.run import git_commit
        if a.action == "predict":
            for e in external.predict(commit=git_commit()):
                print("予測を保存:", e["file"], e["sha256"][:12])
        elif a.action == "score":
            for s in external.score_all():
                print("採点:", s["attempt"]["name"], s["attempt"]["attempt"], s.get("lexicon"), s.get("structure"))
        problems = external.verify()
        for p in problems:
            print("台帳の問題:", p, file=sys.stderr)
        return 1 if problems else 0
    return 0


def _parse_regions(specs, ap):
    regions = []
    for spec in specs:
        try:
            name, coords = spec.split(":", 1)
            box = [int(t) for t in coords.split(",")]
            if not name or len(box) != 4:
                raise ValueError
        except ValueError:
            ap.error("--region は ID:X1,Y1,X2,Y2 の形式で指定してください")
        regions.append({"id": name, "box": box})
    return regions or None


def _analyze(a, ap):
    from babel.structure import corpus as sc
    from babel.structure import transcription
    from babel.structure.analyze import analyze
    from babel.structure.report import markdown
    cfg = {"seed": a.seed}
    for k, v in (("permutations", a.permutations), ("subsamples", a.subsamples), ("perturbations", a.perturbations),
                 ("perturbation_rate", a.perturbation_rate)):
        if v is not None:
            cfg[k] = v
    corr = json.loads(Path(a.corrections).read_text(encoding="utf-8")) if a.corrections else None
    src = Path(a.corpus)
    image = a.image
    alts = []
    if src.suffix.lower() in transcription.IMAGE_SUFFIXES:
        sims = [float(x) for x in a.similarity.split(",")]
        reps = transcription.vision_variants(src, sims, regions=_parse_regions(a.region, ap),
                                             out_dir=Path(a.out).parent / "transcriptions")
        main_corpus = sc.from_glyph_report(reps[0][1])
        alts = [(n, sc.from_glyph_report(r)) for n, r in reps[1:]]
        image = image or str(src)
    else:
        main_corpus = sc.load(src)
    alts += [(p, sc.load(p)) for p in a.alt]
    if corr:
        main_corpus = sc.apply_corrections(main_corpus, corr)
    res = analyze(main_corpus, cfg)
    if alts:
        alt_cfg = dict(cfg, subsamples=0, perturbations=0)
        res["transcription_comparison"] = transcription.compare(res, [(n, analyze(c, alt_cfg)) for n, c in alts])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    md = Path(a.md) if a.md else out.with_suffix(".md")
    md.write_text(markdown(res), encoding="utf-8")
    if image and main_corpus.boxes:
        transcription.annotate(image, res, out.with_suffix(".png"))
    print(json.dumps({"verdict": res["verdict"]["code"], "hypotheses": len(res["hypotheses"]),
                      "stable": res["verdict"]["stable_hypotheses"], "json": str(out), "markdown": str(md)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
