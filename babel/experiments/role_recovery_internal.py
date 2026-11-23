"""Internal synthetic comparisons and paired standard-decoder regression checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import product
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "0180be30b6df159c564971a8d6bef707505fd511"
FAMILIES = ("KES", "UMB", "SOT")
METHODS = ("A", "B", "C", "D", "E")
SEEDS = {"dev": (0, 1, 2), "final": (600, 601, 602)}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def lock_predictions(path, predictions):
    """Fix deterministic predictions before any annotation scoring; reject rewrites."""
    blob = canonical(predictions) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(blob)
    except FileExistsError:
        if path.read_bytes() != blob:
            raise ValueError(f"Predictions are already fixed with different contents: {path}") from None
    return hashlib.sha256(blob).hexdigest()


def shift_one(public):
    """Rotate translations within the existing training-only parallel set."""
    public = json.loads(canonical(public))
    pairs = public.get("parallel", [])
    translations = [pair["translation"] for pair in pairs]
    if len(translations) > 1:
        for pair, translation in zip(pairs, translations[1:] + translations[:1], strict=True):
            pair["translation"] = translation
    return public


def build_study_corpus(family, seed, level, **parameters):
    """Preserve semantic NP roles before surface case syncretism, in annotation only.

    The generator's SOT numeral construction changes a head's surface case to
    GEN. Capturing the original NP case avoids mislabelling a counted object as
    a possessor. Nested possessor heads are marked by their own recursive call.
    This temporary instrumentation does not change any surface or public data.
    """
    from babel.generator import corpus

    original_factory = corpus.make_language
    role_of_case = {"NOM": "subject", "ACC": "object", "DAT": "recipient", "LOC": "location", "GEN": "possessor"}

    def language_with_roles(*args, **kwargs):
        language = original_factory(*args, **kwargs)
        original_np = language.np

        def np_with_role(self, noun_phrase, case, era):
            words = original_np(noun_phrase, case, era)
            heads = [word for word in words if word["pos"] == "N" and word["concept"] == noun_phrase["head"]
                     and not any(feature.startswith("STUDY_ROLE:") for feature in word["feats"])]
            if not heads:
                raise RuntimeError("Cannot locate the rendered noun-phrase head")
            # Titles precede the head in every generated family; recursive
            # possessors already carry their own role and cannot be overwritten.
            head = heads[-1]
            head["feats"] = sorted([*head["feats"], "STUDY_ROLE:" + role_of_case[case]])
            return words

        language.np = MethodType(np_with_role, language)
        return language

    corpus.make_language = language_with_roles
    try:
        public, annotation = corpus.build_corpus(family, seed, level, **parameters)
    finally:
        corpus.make_language = original_factory
    if b"STUDY_ROLE:" in canonical(public):
        raise RuntimeError("Study-only role annotation leaked into public input")
    return public, annotation


def synthetic_gold(public, gold):
    """Strict concept-and-original-role triples, consumed only after prediction."""
    result = {}
    for pair in public.get("parallel", []):
        translation = pair["translation"]
        heads = {arg["role"]: arg.get("head") for arg in translation.get("args", [])}
        concepts = (heads.get("subject"), translation.get("pred"), heads.get("object"))
        tokens = gold["tokens"][pair["doc"]][pair["sent"]]
        choices = [[], [], []]
        for index, token in enumerate(tokens):
            if token.get("lacuna"):
                continue
            features = set(token["feats"])
            if token["pos"] == "V" and token["concept"] == concepts[1] and concepts[1]:
                choices[1].append(index)
            if token["pos"] != "N":
                continue
            if concepts[0] and token["concept"] == concepts[0] and "STUDY_ROLE:subject" in features:
                choices[0].append(index)
            if concepts[2] and token["concept"] == concepts[2] and "STUDY_ROLE:object" in features:
                choices[2].append(index)
        result[pair["doc"], pair["sent"]] = [triple for triple in product(*choices) if len(set(triple)) == 3]
    return result


def _order_judge(decision, gold):
    if decision is None:
        return "abstain"
    if gold is None:
        return "claim_without_gold"
    return "correct" if decision == gold else "wrong"


def internal_case(task):
    from babel.decipher.role_inference import RoleInference
    from babel.experiments.role_recovery_score import order_decision, score_predictions

    phase, family, seed, condition = task
    no_evidence = condition == "C-no-evidence"
    mismatch = condition.endswith("shift1")
    fraction = 0.25 if condition.startswith("A25") else 0.05
    level = "C" if no_evidence else "A"
    public, annotation = build_study_corpus(family, seed, level, n_docs=65, parallel_frac=fraction)
    if mismatch:
        public = shift_one(public)
    public_blob = canonical(public)
    public = json.loads(public_blob)
    model = RoleInference(public)
    predictions = model.predict_all()
    case_id = f"{family}-{seed}-{condition}"
    lock = ROOT / ".cache/role_recovery/locked" / f"internal-{phase}" / f"{case_id}.json"
    prediction_sha = lock_predictions(lock, predictions)
    predictions = json.loads(lock.read_text(encoding="utf-8"))
    # The predictor receives only the serialized public object. Annotation is
    # consulted after the initial prediction artifact has been made immutable.
    gold_by_key = synthetic_gold(public, annotation)
    methods = {}
    for method in METHODS:
        rows = predictions[method]
        order = order_decision(rows)
        methods[method] = {"scores": score_predictions(rows, gold_by_key), "order": order,
                           "order_judge": _order_judge(order["value"], annotation["structure"].get("verb_position"))}
    return {"case_id": case_id, "phase": phase, "family": family, "seed": seed,
            "condition": condition, "level": level, "mismatch": mismatch, "parallel_fraction": fraction,
            "n_docs": 65, "no_evidence": no_evidence, "parallel_sentences": len(public.get("parallel", [])),
            "input_sha256": hashlib.sha256(public_blob).hexdigest(), "prediction_sha256": prediction_sha,
            "prediction_path": str(lock.relative_to(ROOT)), "timings": model.timings, "methods": methods}


def internal_tasks(phase):
    conditions = ("A05-matched", "A05-shift1", "A25-matched", "A25-shift1", "C-no-evidence")
    return [(phase, family, seed, condition) for family in FAMILIES for seed in SEEDS[phase] for condition in conditions]


def internal_summary(rows):
    summary = {}
    count_keys = ("total", "eligible", "gold_eligible", "claimed", "correct", "wrong", "abstain",
                  "high_confidence_wrong", "unsupported_claimed", "raw_retained", "budget_retained")
    for condition in ("matched", "shift1", "no-evidence"):
        selected = [row for row in rows if (condition == "no-evidence" and row["no_evidence"])
                    or (condition != "no-evidence" and row["condition"].endswith(condition))]
        summary[condition] = {}
        for method in METHODS:
            counts = {key: sum(row["methods"][method]["scores"][key] for row in selected) for key in count_keys}
            counts["precision"] = counts["correct"] / counts["claimed"] if counts["claimed"] else None
            counts["abstain_rate"] = counts["abstain"] / counts["eligible"] if counts["eligible"] else None
            counts["order_correct"] = sum(row["methods"][method]["order_judge"] == "correct" for row in selected)
            counts["order_wrong"] = sum(row["methods"][method]["order_judge"] in ("wrong", "claim_without_gold") for row in selected)
            counts["order_abstain"] = sum(row["methods"][method]["order_judge"] == "abstain" for row in selected)
            summary[condition][method] = counts
    return summary


def run_internal(phase, jobs=2):
    with ProcessPoolExecutor(jobs) as pool:
        rows = list(pool.map(internal_case, internal_tasks(phase)))
    return {"study": "role-recovery-internal", "phase": phase, "seeds": list(SEEDS[phase]),
            "families": list(FAMILIES), "n_docs": 65, "independent_blind": False,
            "gold_definition": "concept and original semantic NP roles, including SOT numeral-case substitutions; annotation only",
            "summary": internal_summary(rows), "rows": rows}


def _regression_worker(repo, family, seed, level, output, helper):
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=str(repo))
    completed = subprocess.run([sys.executable, str(helper), "--repo", str(repo), "--family", family,
                                "--seed", str(seed), "--level", level, "--out", str(output)],
                               cwd=repo, env=env, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"Regression failed: {repo}/{family}/{seed}/{level}\n{completed.stderr}")
    return json.loads(output.read_text(encoding="utf-8"))


def regression_pair(task):
    family, seed, level, baseline, helper = task
    case_id = f"{family}-{seed}-{level}"
    outputs = ROOT / ".cache/role_recovery/regression"
    old = _regression_worker(baseline, family, seed, level, outputs / f"{case_id}-baseline.json", helper)
    current = _regression_worker(ROOT, family, seed, level, outputs / f"{case_id}-current.json", helper)
    if old["input_sha256"] != current["input_sha256"]:
        raise ValueError(f"Regression inputs differ: {case_id}")
    return {"case_id": case_id, "family": family, "seed": seed, "level": level,
            "input_sha256": old["input_sha256"],
            "same_prediction": old["prediction_sha256"] == current["prediction_sha256"],
            "same_scores": old["score"] == current["score"], "baseline": old, "current": current}


def run_regression(jobs=2, baseline_path=None):
    baseline = (baseline_path or ROOT / ".cache/role_recovery/baseline-0180be3").resolve()
    if not (baseline / ".git").exists():
        subprocess.run(["git", "worktree", "add", "--detach", str(baseline), BASELINE], cwd=ROOT, check=True,
                       capture_output=True, text=True)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=baseline, text=True).strip()
    if actual != BASELINE:
        raise ValueError(f"Regression baseline is not the registered commit: {actual}")
    helper = ROOT / ".cache/role_recovery_regression_worker.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_text(REGRESSION_WORKER, encoding="utf-8")
    tasks = [(family, seed, level, baseline, helper) for family in FAMILIES for seed in (0, 1, 2) for level in "ABC"]
    with ThreadPoolExecutor(jobs) as pool:
        rows = list(pool.map(regression_pair, tasks))
    return {"study": "standard-decoder-regression", "baseline_commit": BASELINE, "n_docs": 33,
            "cases": len(rows), "all_predictions_equal": all(row["same_prediction"] for row in rows),
            "all_scores_equal": all(row["same_scores"] for row in rows),
            "excluded_prediction_fields": ["timing"], "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "final"), default="dev")
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if os.environ.get("PYTHONHASHSEED") != "0":
        parser.error("Run with PYTHONHASHSEED=0")
    if args.phase == "final" and not args.regression:
        from babel.experiments.role_recovery import check_freeze
        check_freeze()
    started = time.perf_counter()
    result = run_regression(args.jobs) if args.regression else run_internal(args.phase, args.jobs)
    result["seconds"] = round(time.perf_counter() - started, 6)
    if args.freeze is not None:
        result["freeze_sha256"] = hashlib.sha256(args.freeze.read_bytes()).hexdigest()
    output = args.out or ROOT / "results/role_recovery" / ("standard-regression.json" if args.regression else f"internal-{args.phase}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    print(output)
    if args.regression:
        print(json.dumps({key: result[key] for key in ("cases", "all_predictions_equal", "all_scores_equal")}))
        if not result["all_predictions_equal"] or not result["all_scores_equal"]:
            raise SystemExit(1)
    else:
        print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


REGRESSION_WORKER = '''"""Isolated standard-decoder regression worker."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")

def without_timing(value):
    if isinstance(value, dict):
        return {key: without_timing(item) for key, item in value.items() if key != "timing"}
    if isinstance(value, list):
        return [without_timing(item) for item in value]
    return value

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--family", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--level", required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if os.environ.get("PYTHONHASHSEED") != "0":
    parser.error("PYTHONHASHSEED=0 required")
repo = args.repo.resolve()
sys.path[0] = str(repo)
from babel.decipher import engine
from babel.evaluate.score import score
from babel.generator.corpus import build_corpus
engine_path = Path(engine.__file__).resolve()
if not engine_path.is_relative_to(repo):
    raise RuntimeError(f"Imported wrong checkout: {engine_path}")
started = time.perf_counter()
public, gold = build_corpus(args.family, args.seed, args.level, n_docs=33)
public_blob = canonical(public)
public = json.loads(public_blob)
predicted = engine.decipher(public)
measured = score(predicted, gold, public)
result = {"family": args.family, "seed": args.seed, "level": args.level, "n_docs": 33,
          "input_sha256": hashlib.sha256(public_blob).hexdigest(),
          "prediction_sha256": hashlib.sha256(canonical(without_timing(predicted))).hexdigest(),
          "score_sha256": hashlib.sha256(canonical(measured)).hexdigest(),
          "engine_sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(),
          "score": measured, "seconds": round(time.perf_counter() - started, 6)}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_bytes(canonical(result) + b"\\n")
'''


if __name__ == "__main__":
    main()
