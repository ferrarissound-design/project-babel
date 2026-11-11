"""研究ダッシュボード用の JSON を生成する（docs/data/）。表示は静的な HTML/CSS/JS が行う。"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from babel import external
from babel.decipher.engine import Decipherer
from babel.evaluate.score import _same_concept, score
from babel.experiments import scorecard
from babel.generator.corpus import build_corpus
from babel.generator.languages import FAMILIES

FAMILY_INFO = {
    "KES": "膠着語・接尾辞・母音調和・SOV・10進法",
    "UMB": "接頭辞型・VSO・20進法",
    "SOT": "屈折語（融合語尾・2変化類・一致）・SVO・12進法・不規則動詞",
}


def _dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def showcase(seed=0):
    out = []
    for fam in FAMILIES:
        for lv in "ABC":
            public, secret = build_corpus(fam, seed, lv)
            d = Decipherer(public).fit()
            pred = d.predict()
            sc = score(pred, secret, public)
            aliases = secret.get("aliases", {})
            gold_of = defaultdict(Counter)
            for did, sents in secret["tokens"].items():
                for gs, ps in zip(sents, pred["tokens"][did]):
                    for g, p in zip(gs, ps):
                        if not g.get("lacuna"):
                            gold_of[p["stem"]][g["concept"]] += 1
            lex = []
            for e in sorted(pred["lexicon"], key=lambda e: -e["count"])[:40]:
                g = gold_of[e["stem"]].most_common(1)[0][0] if gold_of[e["stem"]] else None
                lex.append({"stem": e["stem"], "forms": e["forms"][:3], "count": e["count"], "pos": e["pos"],
                            "meaning": e["meaning"], "confidence": e["confidence"], "status": e["status"],
                            "reason": e["reason"], "gold": g,
                            "correct": None if e["meaning"] is None else _same_concept(e["meaning"], g, aliases)})
            examples = []
            for did in public["split"]["heldout"][:3]:
                for si in range(min(3, len(secret["tokens"][did]))):
                    toks = []
                    for g, p in zip(secret["tokens"][did][si], pred["tokens"][did][si]):
                        if g.get("lacuna"):
                            toks.append({"s": "[…]", "lacuna": True})
                            continue
                        toks.append({"s": g["s"], "parts": [p["s"][a:b] for a, b in
                                                            zip([0] + p["cuts"], p["cuts"] + [len(p["s"])])],
                                     "pos": p["pos"], "concept": p["concept"], "feats": p["feats"],
                                     "gold": g["concept"], "gold_feats": g["feats"],
                                     "ok": None if p["concept"] is None else _same_concept(p["concept"], g["concept"], aliases)})
                    examples.append({"doc": did, "tokens": toks})
            gold_num = {}
            for sents, psents in ((secret["tokens"][k], pred["tokens"][k]) for k in secret["tokens"]):
                for gs, ps in zip(sents, psents):
                    for g, p in zip(gs, ps):
                        if not g.get("lacuna") and g["pos"] == "NUM":
                            gold_num[p["norm"]] = int(g["concept"][1:])
            nums = [{"form": f, "value": v, "gold": gold_num.get(f)} for f, v in
                    sorted(pred["numerals"]["values"].items(), key=lambda kv: kv[1])]
            history = []
            for frac in (0.25, 0.5, 0.75):
                sub = dict(public, split={"train": public["split"]["train"][: int(len(public["split"]["train"]) * frac)],
                                          "heldout": []})
                sub["parallel"] = [p for p in public["parallel"] if p["doc"] in set(sub["split"]["train"])]
                history.append(_snapshot(Decipherer(sub).fit(), frac))
            history.append(_snapshot(d, 1.0))
            out.append({
                "family": fam, "typology": FAMILY_INFO[fam], "level": lv, "seed": seed,
                "language_id": public["language_id"],
                "samples": [{"id": doc["id"], "meta": doc["meta"], "sentences": doc["sentences"][:6]}
                            for doc in public["documents"][:3]],
                "frequency": pred["frequency"], "verdict": pred["verdict"], "summary": pred["summary"],
                "structure": {k: {"value": v.get("value"), "confidence": v.get("confidence"),
                                  "gold": secret["structure"].get(k), "reason": v.get("reason")}
                              for k, v in pred["structure"].items() if k != "sum_marker"},
                "sum_marker": pred["structure"]["sum_marker"]["value"],
                "hypotheses": pred["hypotheses"], "variants": pred["variants"]["rules"],
                "affixes": [{"affix": a["affix"], "types": a["types"], "tokens": a["tokens"],
                             "function": a["function"]} for a in pred["morphology"]["affixes"][:14]],
                "numerals": {"status": pred["numerals"]["status"], "reason": pred["numerals"]["reason"],
                             "relative": pred["numerals"]["relative"], "equations": pred["numerals"]["equations"],
                             "confidence": pred["numerals"]["confidence"], "values": nums,
                             "affix_hypotheses": pred["numerals"].get("affix_hypotheses", [])},
                "lexicon": lex, "examples": examples, "history": history,
                "score": {"segmentation_f1": sc["segmentation"]["f1"], "pos_accuracy": sc["pos"]["accuracy"],
                          "lexicon": sc["lexicon"], "gloss_heldout": sc["gloss_heldout"],
                          "numerals": sc["numerals"], "structure": sc["structure"]},
                "errors": sc["errors"][:6],
            })
    return out


def _snapshot(d, frac):
    st = d._structure()
    claimed = sum(1 for s in d.lexicon if d._claim(s)[0])
    return {"fraction": frac, "documents": len(d.train_ids),
            "verb_position": [st["verb_position"]["value"], st["verb_position"]["confidence"]],
            "affix_position": [st["affix_position"]["value"], st["affix_position"]["confidence"]],
            "numeral_base": [st["numeral_base"]["value"], st["numeral_base"]["confidence"]],
            "so_order": [st["so_order"]["value"], st["so_order"]["confidence"]],
            "numerals_solved": len(d.num_values), "meanings_claimed": claimed}


def issues_markdown(issues):
    lines = ["# 発見された問題と対応の記録", "",
             "開発中に見つかった問題を、原因・対応・検証とともに残す。未解決のものも隠さず載せる。", ""]
    for i in issues:
        lines += [f"## {i['id']}. {i['title']}（{i['status']}）", "", f"- 発見: {i['found']}", f"- 原因: {i['cause']}",
                  f"- 対応: {i['fix']}", f"- 検証: {i['verified']}", ""]
    return "\n".join(lines)


def build(results_path="results/latest.json", out_dir="docs/data", with_showcase=True):
    out = Path(out_dir)
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    issues = json.loads(Path("research/issues.json").read_text(encoding="utf-8"))
    limits = json.loads(Path("research/limitations.json").read_text(encoding="utf-8"))
    extras_p = Path("research/verification.json")
    extras = json.loads(extras_p.read_text(encoding="utf-8")) if extras_p.exists() else {}
    ext_scores = []
    sdir = external.ROOT / "scores"
    for f in sorted(sdir.rglob("attempt-*.json")) if sdir.exists() else []:
        ext_scores.append(json.loads(f.read_text(encoding="utf-8")))
    indep = [s for s in ext_scores if s.get("independent") and s.get("blind") and s["attempt"]["first_attempt"]]
    if indep:
        pts, detail = [], []
        for s in indep:
            lx, nm = s.get("lexicon", {}), s.get("numerals", {})
            st = [v for v in s.get("structure", {}).values()]
            acc = sum(v == "correct" for v in st) / len(st) if st else 0.0
            p = 10 * (0.5 * acc + 0.25 * (lx.get("precision") or 0.0) + 0.25 * nm.get("accuracy", 0.0))
            pts.append(p)
            detail.append(f"{s['attempt']['name']}: 構造 {acc:.2f}, 語彙適合率 {lx.get('precision')}, 数詞 {nm.get('accuracy')}")
        extras["external_independent"] = {"points": sum(pts) / len(pts), "detail": detail}
    card = scorecard.build(results, issues, extras)
    slim = dict(results)
    slim["runs"] = [{k: r.get(k) for k in ("experiment", "label", "family", "seed", "ok", "seconds", "error")}
                    for r in results["runs"]]
    _dump(out / "results.json", slim)
    _dump(out / "scorecard.json", card)
    _dump(out / "issues.json", {"issues": issues, "limitations": limits})
    _dump(out / "external.json", {"ledger": external.read_ledger(), "scores": ext_scores,
                                  "problems": external.verify(),
                                  "engine_fingerprint": external.engine_fingerprint()})
    hist_p = Path("results/history.jsonl")
    hist = [json.loads(x) for x in hist_p.read_text(encoding="utf-8").splitlines() if x.strip()] if hist_p.exists() else []
    _dump(out / "history.json", hist)
    if with_showcase:
        _dump(out / "showcase.json", showcase())
    Path("docs/ISSUES.md").write_text(issues_markdown(issues), encoding="utf-8")
    Path("results/scorecard.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
    return card
