"""Re-run frozen role-study cases and compare hashes/metrics without replacing results.

This is a verification utility, added after the development freeze. It does not
define, tune, or modify inference, scoring, split membership, or adoption rules.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from babel.experiments.role_recovery import OUT, canonical, check_freeze, run_job


def compare_case(pair):
    job, expected = pair
    actual = run_job(job)
    a = {key: value for key, value in actual.items() if key != "timings"}
    b = {key: value for key, value in expected.items() if key != "timings"}
    if a != b:
        raise ValueError(f"Frozen prediction/metric reproduction failed: {job}")
    return {"code": job[0], "phase": job[1], "seed": job[2], "mismatch": job[3],
            "predictions_sha256": actual["predictions_sha256"], "all_non_timing_fields_equal": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "final"), required=True)
    parser.add_argument("--all", action="store_true", help="Repeat all 120 cases; default repeats two registered cases")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    frozen = check_freeze()
    original = json.loads((OUT / (args.phase + ".json")).read_text())["rows"]
    chosen = original if args.all else [row for row in original
                                       if (row["code"], row["seed"], row["mismatch"]) in
                                       {("ja_pud", 0, False), ("gl_pud", 2, True)}]
    if len(chosen) != (120 if args.all else 2):
        raise ValueError("Missing registered reproduction cases")
    tasks = [((row["code"], args.phase, row["seed"], row["mismatch"]), row) for row in chosen]
    with ProcessPoolExecutor(args.jobs) as pool:
        results = list(pool.map(compare_case, tasks))
    payload = {"phase": args.phase, "freeze_sha256": frozen["freeze_sha256"], "cases": results,
               "all_equal": all(row["all_non_timing_fields_equal"] for row in results)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.resolve() == (OUT / (args.phase + ".json")).resolve():
        raise ValueError("Reproduction output must not replace initial results")
    args.out.write_text(canonical(payload) + "\n", encoding="utf-8")
    print(args.out, "all_equal:", payload["all_equal"])


if __name__ == "__main__":
    main()
