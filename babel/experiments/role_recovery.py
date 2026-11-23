"""Article-separated, transductive A–E role-recovery comparisons (not blind tests)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from babel.experiments import real_parallel as rp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/role_recovery"
MANIFEST = OUT / "article-split.json"
SEEDS = (0, 1, 2)
TARGETS = [*rp.TARGETS, ("UD_Galician-PUD", "gl_pud")]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def source_articles(path):
    """Read only document/sentence metadata, without consulting dependency labels."""
    current, mapping = None, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# newdoc id = "):
            current = line.split("=", 1)[1].strip()
        elif line.startswith("# sent_id = "):
            sid = line.split("=", 1)[1].strip()
            if current is None or sid in mapping:
                raise ValueError("Missing article metadata or duplicate source sentence")
            mapping[sid] = current
    return mapping


def make_manifest(path, source_sha):
    mapping = source_articles(path)
    article_phase = {}
    for genre in ("n", "w"):
        articles = sorted(set(mapping.values()) & {a for a in mapping.values() if a.startswith(genre)},
                          key=lambda a: hashlib.sha256(("role-recovery-v1:" + a).encode()).hexdigest())
        boundary = math.floor(len(articles) * 0.7)
        for i, article in enumerate(articles):
            article_phase[article] = "dev" if i < boundary else "final"
    if set(mapping.values()) != set(article_phase):
        raise ValueError("Unexpected source genre")
    manifest = {"schema": "role-recovery-split-v1", "source_sha256": source_sha,
                "article_phase": article_phase, "sentence_article": mapping, "cipher_seeds": SEEDS}
    manifest["membership_sha256"] = digest({"article_phase": article_phase, "sentence_article": mapping})
    return manifest


def prepare():
    path, sha = rp.fetch(*rp.SOURCE)
    manifest = make_manifest(path, sha)
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != json.loads(canonical(manifest)):
        raise ValueError("Refusing to replace a different registered article split")
    MANIFEST.write_text(canonical(manifest) + "\n", encoding="utf-8")
    return manifest


def build_input(code, phase, seed, mismatch, manifest):
    repo = dict((c, r) for r, c in TARGETS)[code]
    en_path, en_sha = rp.fetch(*rp.SOURCE)
    if en_sha != manifest["source_sha256"]:
        raise ValueError("Source input changed after split registration")
    en_tr = {s["sent_id"]: tr for s in rp.parse(en_path) if (tr := rp.translation(s))}
    target_path, target_sha = rp.fetch(repo, code)
    parsed = rp.parse(target_path)
    mapping = manifest["sentence_article"]
    tsents = [s for s in rp.target_sentences(parsed) if s[0] in mapping
              and manifest["article_phase"][mapping[s[0]]] == phase]
    words, _ = rp.rl.cipher([s[1] for s in tsents], seed=seed, allow_extended=True)
    docs, positions = {}, {}
    for (sid, _, _), sentence in zip(tsents, words, strict=True):
        article = mapping[sid]
        doc = docs.setdefault(article, {"id": article, "sentences": []})
        positions[sid] = (article, len(doc["sentences"]))
        doc["sentences"].append(" ".join(sentence))
    chosen = sorted(set(positions) & set(en_tr))
    translations = [en_tr[sid] for sid in chosen]
    if mismatch:
        translations = translations[1:] + translations[:1]
    public = {"language_id": f"role-study-{code}-{phase}-{seed}", "level": "A", "has_word_boundaries": True,
              "documents": list(docs.values()), "known": [],
              "parallel": [{"doc": positions[sid][0], "sent": positions[sid][1], "translation": tr}
                           for sid, tr in zip(chosen, translations, strict=True)]}
    # Only public strings and English structured translations cross the inference boundary.
    sid_for_key = {positions[sid]: sid for sid in positions}
    return json.loads(canonical(public)), {"target_sha256": target_sha, "source_sha256": en_sha,
                                           "sentence_ids": [s[0] for s in tsents]}, sid_for_key


def run_job(job):
    from babel.decipher.role_inference import RoleInference
    from babel.experiments.candidate_audit import direct_triples
    from babel.experiments.clause_probe import target_view
    from babel.experiments.role_recovery_score import order_decision, score_predictions

    code, phase, seed, mismatch = job
    manifest = json.loads(MANIFEST.read_text())
    public, meta, sid_for_key = build_input(code, phase, seed, mismatch, manifest)
    model = RoleInference(public)
    predictions = model.predict_all()
    lock = {"code": code, "phase": phase, "seed": seed, "mismatch": mismatch,
            "public_sha256": digest(public), "split_sha256": manifest["membership_sha256"], "predictions": predictions}
    lock_path = ROOT / ".cache/role_recovery/locked" / phase / f"{code}-{seed}-{int(mismatch)}.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    locked = canonical(lock) + "\n"
    if lock_path.exists() and lock_path.read_text(encoding="utf-8") != locked:
        raise ValueError(f"Refusing to overwrite differing predictions: {lock_path}")
    lock_path.write_text(locked, encoding="utf-8")
    predictions_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    predictions = json.loads(lock_path.read_text(encoding="utf-8"))["predictions"]
    # Predictions are fixed before target dependency annotations are consulted for grading.
    repo = dict((c, r) for r, c in TARGETS)[code]
    path, sha = rp.fetch(repo, code)
    if sha != meta["target_sha256"]:
        raise ValueError("Target input changed before grading")
    raw = {s["sent_id"]: s for s in rp.parse(path)}
    gold = {key: direct_triples(target_view(raw[sid])[0]) for key, sid in sid_for_key.items()}
    gold_order = rp.gold_verb_rel([raw[sid] for sid in meta["sentence_ids"]])
    metrics = {}
    for method, rows in predictions.items():
        metrics[method] = score_predictions(rows, gold)
        decision = order_decision(rows)
        metrics[method]["order"] = dict(decision, gold=gold_order["value"],
                                        judge=rp._judge(decision["value"], gold_order["value"]))
    return {"code": code, "phase": phase, "seed": seed, "mismatch": mismatch, **meta,
            "public_sha256": digest(public), "predictions_sha256": predictions_sha,
            "split_sha256": manifest["membership_sha256"], "timings": model.timings, "methods": metrics}


def source_fingerprint():
    paths = [ROOT / "docs/PREREG_ROLE_RECOVERY.md", MANIFEST,
             *sorted((ROOT / "babel/decipher").glob("*.py")),
             *sorted((ROOT / "babel/experiments").glob("role_recovery*.py"))]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def freeze_selection(selection):
    payload = {"selection": selection, "source_files": source_fingerprint(),
               "dev_results": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (OUT / "dev.json", OUT / "internal-dev.json")},
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    payload["freeze_sha256"] = digest(payload)
    path = OUT / "development-freeze.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Development freeze already exists and differs")
    path.write_text(canonical(payload) + "\n", encoding="utf-8")
    return payload


def check_freeze():
    payload = json.loads((OUT / "development-freeze.json").read_text())
    recorded_sha = payload.get("freeze_sha256")
    if not recorded_sha or recorded_sha != digest({k: v for k, v in payload.items() if k != "freeze_sha256"}):
        raise ValueError("Development freeze contents do not match their recorded hash")
    if payload["source_files"] != source_fingerprint():
        raise ValueError("Algorithm/configuration changed after development freeze")
    for name, sha in payload["dev_results"].items():
        if hashlib.sha256((OUT / name).read_bytes()).hexdigest() != sha:
            raise ValueError("Development results changed after freeze")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--phase", choices=("dev", "final"))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--code", choices=[c for _, c in TARGETS])
    args = parser.parse_args()
    if args.prepare:
        manifest = prepare()
        print(MANIFEST, manifest["membership_sha256"])
        return
    if not args.phase or args.jobs < 1:
        parser.error("--phase and positive --jobs required")
    manifest = json.loads(MANIFEST.read_text())
    if args.phase == "final":
        check_freeze()
    for repo, code in [rp.SOURCE, *TARGETS]:
        rp.fetch(repo, code)
    targets = [c for _, c in TARGETS if not args.code or c == args.code]
    jobs = [(code, args.phase, seed, mismatch) for code in targets for seed in SEEDS for mismatch in (False, True)]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = []
        for row in pool.map(run_job, jobs):
            rows.append(row)
            print(row["code"], row["seed"], row["mismatch"], "complete", flush=True)
    output = {"schema": "role-recovery-real-v1", "phase": args.phase, "split_sha256": manifest["membership_sha256"],
              "seeds": SEEDS, "rows": rows}
    path = OUT / (args.phase + ("-" + args.code if args.code else "") + ".json")
    if path.exists() and json.loads(path.read_text()).get("rows"):
        # Runtime varies; preserve first results rather than silently replacing them.
        raise ValueError(f"Output already exists; retain it and compare reproduction separately: {path}")
    path.write_text(canonical(output) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
