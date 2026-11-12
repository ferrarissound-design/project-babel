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
        data = save_analysis(a.image, a.out_dir, regions=regions or None,
                             threshold=a.threshold, similarity=a.similarity)
        print(json.dumps({"out_dir": a.out_dir, **data["stats"]}, ensure_ascii=False))
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


if __name__ == "__main__":
    sys.exit(main())
