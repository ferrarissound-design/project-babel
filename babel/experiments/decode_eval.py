"""Article-separated evaluation of the v2 decoder on PUD (reuses the PR #31 split and scorer).

Predictions are written to a lock file (with hashes) before any target
annotation is read. Grading reuses ``role_recovery_score.score_predictions``
unchanged, so numbers are directly comparable with A–E in results/role_recovery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from babel.experiments import real_parallel as rp
from babel.experiments import role_recovery as rr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/decode_v2"
LOCK = ROOT / ".cache/decode_v2/locked"
KNOWN19 = [c for _, c in rp.TARGETS]
TUNED8 = ["es_pud", "de_pud", "ja_pud", "ru_pud", "fr_pud", "tr_pud", "zh_pud", "ar_pud"]   # used for design choices
UNTUNED11 = [c for c in KNOWN19 if c not in TUNED8]


def source_fingerprint():
    """SHA-256 of every decoder source file (and the calibration table if present)."""
    files = sorted((ROOT / "babel/decode").glob("*.py")) + sorted((ROOT / "babel/decode").glob("*.json"))
    files += [ROOT / "babel/decipher/morph.py", ROOT / "babel/decipher/segment.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def meta_features(rec):
    """Public-only numbers describing how a best triple was reached (input to the calibrator)."""
    from babel.decode.calibration import features
    return features(rec)


def to_rows(records):
    """Decoder records -> rows accepted by score_predictions (selected/abstain + candidate pools)."""
    rows = []
    for rec in records:
        pools = []
        for name in ("subject", "predicate", "object"):
            role = (rec.get("roles") or {}).get(name) or {}
            pools.append(sorted(c["position"] for c in role.get("candidates", [])))
        t = rec.get("triple")
        selected = [t["subject"], t["predicate"], t["object"]] if t else None
        rows.append({"doc": rec["doc"], "sent": rec["sent"], "source_full": rec["complete_source"],
                     "raw_candidates": pools, "candidates": pools, "selected": selected,
                     "best": selected or rec.get("best_guess"), "confidence": rec.get("confidence", 0.0),
                     "agreement": rec.get("restart_agreement"), "reason": rec.get("reason"),
                     "meta": meta_features(rec)})
    return rows


FREEZE = OUT / "freeze.json"


def freeze(dev_files):
    """Record decoder sources, the calibrator and development results before the final split is run."""
    import subprocess
    payload = {"source_files": source_fingerprint(),
               "dev_results": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in sorted(dev_files)},
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    payload["freeze_sha256"] = rr.digest(payload)
    if FREEZE.exists() and json.loads(FREEZE.read_text()) != payload:
        raise ValueError("a different freeze already exists")
    FREEZE.write_text(rr.canonical(payload) + "\n", encoding="utf-8")
    return payload


def check_freeze():
    payload = json.loads(FREEZE.read_text(encoding="utf-8"))
    if payload["freeze_sha256"] != rr.digest({k: v for k, v in payload.items() if k != "freeze_sha256"}):
        raise ValueError("freeze file does not match its own hash")
    if payload["source_files"] != source_fingerprint():
        raise ValueError("decoder changed after the development freeze; the final split may not be run")
    for f, sha in payload["dev_results"].items():
        if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != sha:
            raise ValueError(f"development result changed after freeze: {f}")
    return payload


def predict_job(job):
    from babel.decode.engine import Decoder

    code, phase, seed, mismatch, config, tag, fingerprint = job
    if source_fingerprint() != fingerprint:
        raise RuntimeError("decoder source changed while the evaluation was running")
    manifest = json.loads(rr.MANIFEST.read_text())
    public, meta, sid_for_key = rr.build_input(code, phase, seed, mismatch, manifest)
    t0 = time.time()
    dec = Decoder(public, config).fit()
    grounded = dec.grounded()
    lock = {"code": code, "phase": phase, "seed": seed, "mismatch": mismatch, "config": config,
            "public_sha256": rr.digest(public), "split_sha256": manifest["membership_sha256"],
            "rows": to_rows(grounded), "seconds": round(time.time() - t0, 2)}
    path = LOCK / tag / phase / f"{code}-{seed}-{int(mismatch)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = rr.canonical({k: v for k, v in lock.items() if k != "seconds"}) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text and phase == "final":
        raise ValueError(f"Refusing to overwrite differing final predictions: {path}")
    path.write_text(text, encoding="utf-8")
    return code, phase, seed, mismatch, str(path), hashlib.sha256(path.read_bytes()).hexdigest(), lock["seconds"], meta


def grade(code, seed, mismatch, path, meta, phase):
    """Read target annotations only after predictions are on disk."""
    from babel.experiments.candidate_audit import direct_triples
    from babel.experiments.clause_probe import target_view
    from babel.experiments.role_recovery_score import score_predictions

    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = json.loads(rr.MANIFEST.read_text())
    _, _, sid_for_key = rr.build_input(code, phase, seed, mismatch, manifest)
    repo = dict((c, r) for r, c in rr.TARGETS)[code]
    tpath, sha = rp.fetch(repo, code)
    if sha != meta["target_sha256"]:
        raise ValueError("target changed")
    raw = {s["sent_id"]: s for s in rp.parse(tpath)}
    gold = {key: direct_triples(target_view(raw[sid])[0]) for key, sid in sid_for_key.items()}
    metrics = score_predictions(lock["rows"], gold)
    # confidence curve: every eligible row's best triple, judged, for coverage/precision analysis
    curve = []
    for row in lock["rows"]:
        if row["source_full"] and row.get("best"):
            key = (row["doc"], int(row["sent"]))
            ok = tuple(map(int, row["best"])) in {tuple(t) for t in gold.get(key, [])}
            curve.append((row["confidence"], int(ok), row.get("meta")))
    return metrics, curve


def run(phase, seeds, mismatches, codes, config, tag, jobs):
    if phase == "final":
        check_freeze()
    fingerprint = source_fingerprint()
    todo = [(c, phase, s, m, config, tag, fingerprint) for c in codes for s in seeds for m in mismatches]
    out = []
    with ProcessPoolExecutor(jobs) as pool:
        for code, ph, seed, mis, path, sha, secs, meta in pool.map(predict_job, todo):
            metrics, curve = grade(code, seed, mis, path, meta, ph)
            public_sha = json.loads(Path(path).read_text(encoding="utf-8"))["public_sha256"]
            out.append({"code": code, "phase": ph, "seed": seed, "mismatch": mis, "predictions_sha256": sha,
                        "public_sha256": public_sha,
                        "seconds": secs, "metrics": metrics, "curve": curve})
    return out


def summarize(rows, codes=KNOWN19):
    agg = {}
    for r in rows:
        if r["code"] not in codes:
            continue
        k = (r["seed"], r["mismatch"])
        a = agg.setdefault(k, {"eligible": 0, "claimed": 0, "correct": 0, "wrong": 0, "abstain": 0, "curve": []})
        for f in ("eligible", "claimed", "correct", "wrong", "abstain"):
            a[f] += r["metrics"][f]
        a["curve"] += r["curve"]
    for a in agg.values():
        a["precision"] = a["correct"] / a["claimed"] if a["claimed"] else None
        a["accuracy"] = a["correct"] / a["eligible"] if a["eligible"] else None
        c = sorted(a.pop("curve"), key=lambda x: -x[0])
        a["oracle_best_correct"] = sum(x[1] for x in c)
        pts = {}
        for cov in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
            k = max(1, int(cov * len(c)))
            pts[cov] = round(sum(x[1] for x in c[:k]) / k, 3) if c else None
        a["precision_at_coverage_of_best"] = pts
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("dev", "final"), default="dev")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--mismatch", choices=("no", "yes", "both"), default="no")
    ap.add_argument("--codes", default=",".join(KNOWN19))
    ap.add_argument("--config", default="{}")
    ap.add_argument("--tag", default="scratch")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out")
    ap.add_argument("--freeze", nargs="+", metavar="DEV_RESULT", help="write results/decode_v2/freeze.json and exit")
    args = ap.parse_args()
    if args.freeze:
        print(json.dumps(freeze(args.freeze), indent=1))
        return
    if args.phase == "final" and args.out and Path(args.out).exists():
        raise SystemExit(f"{args.out} exists: first final results are kept; write a reproduction elsewhere")
    mis = {"no": (False,), "yes": (True,), "both": (False, True)}[args.mismatch]
    rows = run(args.phase, tuple(int(s) for s in args.seeds.split(",")), mis, args.codes.split(","),
               json.loads(args.config), args.tag, args.jobs)
    for r in rows:
        m = r["metrics"]
        print(f'{r["code"]} s{r["seed"]} mis{int(r["mismatch"])}: N={m["eligible"]} correct={m["correct"]} '
              f'wrong={m["wrong"]} abstain={m["abstain"]} best_ok={sum(x[1] for x in r["curve"])} {r["seconds"]}s')
    for k, a in summarize(rows, args.codes.split(",")).items():
        print("SUMMARY", k, json.dumps(a))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rr.canonical({"phase": args.phase, "config": json.loads(args.config),
                                                "source_files": source_fingerprint(), "rows": rows}) + "\n",
                                  encoding="utf-8")


if __name__ == "__main__":
    main()
