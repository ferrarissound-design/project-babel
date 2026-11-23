"""Registered selection gates and reports for role recovery; never tune on final data."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from babel.experiments import real_parallel as rp
from babel.experiments.role_recovery import OUT, ROOT, SEEDS, canonical, check_freeze, freeze_selection

METHODS = ("A", "B", "C", "D", "E")
KNOWN = tuple(code for _, code in rp.TARGETS)
COUNT_KEYS = ("total", "eligible", "gold_eligible", "claimed", "correct", "wrong", "abstain",
              "high_confidence_wrong", "unsupported_claimed", "raw_retained", "budget_retained",
              "candidate_triples", "gold_candidate_triples", "raw_candidate_triples", "raw_gold_candidate_triples")
CONDITIONS = ("A05-matched", "A05-shift1", "A25-matched", "A25-shift1", "C-no-evidence")


def _integer(value, context):
    if type(value) is not int or value < 0:
        raise ValueError(f"{context}: expected a nonnegative integer")
    return value


def _sha(value, context):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{context}: invalid SHA-256")


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _validate_metrics(metrics):
    for key in COUNT_KEYS:
        _integer(metrics[key], key)
    if metrics["correct"] + metrics["wrong"] != metrics["claimed"]:
        raise ValueError("Correct and wrong counts do not conserve claims")
    if metrics["claimed"] + metrics["abstain"] != metrics["eligible"]:
        raise ValueError("Claims and abstentions do not conserve the source-full denominator")
    if not metrics["gold_eligible"] <= metrics["eligible"] <= metrics["total"]:
        raise ValueError("Inconsistent scoring denominators")
    if metrics["high_confidence_wrong"] > metrics["wrong"]:
        raise ValueError("High-confidence errors exceed errors")
    if metrics["budget_retained"] > metrics["raw_retained"] or metrics["raw_retained"] > metrics["gold_eligible"]:
        raise ValueError("Invalid raw/budget candidate recall")
    for prefix in ("", "raw_"):
        if metrics[prefix + "gold_candidate_triples"] > metrics[prefix + "candidate_triples"]:
            raise ValueError("Correct candidate triples exceed all candidates")
    for key, numerator, denominator in (("precision", "correct", "claimed"), ("wrong_rate", "wrong", "claimed"),
                                        ("abstain_rate", "abstain", "eligible"), ("accuracy", "correct", "eligible")):
        expected = _ratio(metrics[numerator], metrics[denominator])
        actual = metrics[key]
        if (expected is None and actual is not None) or (expected is not None and
                (not isinstance(actual, (int, float)) or not math.isclose(actual, expected, abs_tol=1e-12))):
            raise ValueError(f"Inconsistent {key}")


def _methods(row, internal=False):
    if set(row["methods"]) != set(METHODS):
        raise ValueError("Each case must contain exactly A–E")
    denominators = set()
    for method in METHODS:
        metrics = row["methods"][method]["scores"] if internal else row["methods"][method]
        _validate_metrics(metrics)
        denominators.add(tuple(metrics[k] for k in ("total", "eligible", "gold_eligible")))
    if len(denominators) != 1:
        raise ValueError("Methods do not share scoring denominators")


def validate_real(payload, phase, manifest):
    """Require the full preregistered grid and article separation before aggregation."""
    try:
        if payload["phase"] != phase or payload["seeds"] != list(SEEDS):
            raise ValueError("Wrong real-data phase or cipher seeds")
        if payload["split_sha256"] != manifest["membership_sha256"]:
            raise ValueError("Unregistered article split")
        expected = {(code, seed, shifted) for code in (*KNOWN, "gl_pud") for seed in SEEDS for shifted in (False, True)}
        seen, hashes, sentence_sets = set(), {}, {}
        for row in payload["rows"]:
            key = (row["code"], row["seed"], row["mismatch"])
            if key in seen or type(row["mismatch"]) is not bool or row["phase"] != phase:
                raise ValueError("Duplicate case or wrong phase/condition")
            seen.add(key)
            if row["split_sha256"] != payload["split_sha256"] or row["source_sha256"] != manifest["source_sha256"]:
                raise ValueError("Input source or split differs from registration")
            for field in ("target_sha256", "source_sha256", "public_sha256", "predictions_sha256"):
                _sha(row[field], field)
            previous = hashes.setdefault(row["code"], row["target_sha256"])
            if previous != row["target_sha256"]:
                raise ValueError("Target input differs across methods/seeds/conditions")
            sentences = row["sentence_ids"]
            if len(sentences) != len(set(sentences)):
                raise ValueError("Repeated target sentence")
            for sid in sentences:
                article = manifest["sentence_article"][sid]
                if manifest["article_phase"][article] != phase:
                    raise ValueError("Source article crossed development/final boundary")
            if sentence_sets.setdefault(row["code"], set(sentences)) != set(sentences):
                raise ValueError("Target sentence membership differs across seeds/conditions")
            _methods(row)
        if seen != expected:
            raise ValueError("Real-data comparison must contain all 120 registered cases")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Malformed real-data results: {exc}") from exc
    return payload


def validate_internal(payload, phase):
    try:
        seeds = (0, 1, 2) if phase == "dev" else (600, 601, 602)
        if payload["phase"] != phase or payload["seeds"] != list(seeds) or payload["n_docs"] != 65:
            raise ValueError("Wrong internal phase, seeds or corpus size")
        expected = {(family, seed, condition) for family in ("KES", "UMB", "SOT")
                    for seed in seeds for condition in CONDITIONS}
        seen = set()
        for row in payload["rows"]:
            key = (row["family"], row["seed"], row["condition"])
            if key in seen or row["phase"] != phase or row["n_docs"] != 65:
                raise ValueError("Duplicate internal case or wrong configuration")
            seen.add(key)
            if row["no_evidence"] != (row["condition"] == "C-no-evidence"):
                raise ValueError("Wrong no-evidence condition")
            for field in ("input_sha256", "prediction_sha256"):
                _sha(row[field], field)
            _methods(row, internal=True)
        if seen != expected:
            raise ValueError("Internal comparison must contain all 45 registered cases")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Malformed internal results: {exc}") from exc
    return payload


def aggregate_metrics(rows, method, internal=False):
    """Micro counts within one cipher seed; do not pool seeds as independent data."""
    counts = Counter({key: 0 for key in COUNT_KEYS})
    order = Counter()
    pool = {kind: {"total": 0, "p95_max": None} for kind in ("raw", "budget")}
    runtime = 0.0
    language_precisions = []
    for row in rows:
        record = row["methods"][method]
        metrics = record["scores"] if internal else record
        counts.update({key: metrics[key] for key in COUNT_KEYS})
        if metrics["precision"] is not None:
            language_precisions.append(metrics["precision"])
        decision = record["order"]
        judge = record["order_judge"] if internal else decision["judge"]
        order["correct" if judge == "correct" else "wrong" if judge in ("wrong", "claim_without_gold") else "abstain"] += 1
        order["claimed"] += decision["value"] is not None
        for kind in pool:
            sizes = metrics[kind + "_all_pool_sizes"]
            pool[kind]["total"] += sum(value["total"] for value in sizes.values())
            values = [value["p95"] for value in sizes.values() if value["p95"] is not None]
            if values:
                pool[kind]["p95_max"] = max([*values, pool[kind]["p95_max"] or 0])
        timing = row["timings"]
        runtime += timing["preprocess_seconds"] + timing["soft_fit_seconds" if method in ("B", "E") else "hard_fit_seconds"]
        runtime += timing["selection_seconds"][method]
    result = dict(counts)
    result.update(precision=_ratio(counts["correct"], counts["claimed"]), wrong_rate=_ratio(counts["wrong"], counts["claimed"]),
                  abstain_rate=_ratio(counts["abstain"], counts["eligible"]), accuracy=_ratio(counts["correct"], counts["eligible"]),
                  raw_recall=_ratio(counts["raw_retained"], counts["gold_eligible"]),
                  budget_recall=_ratio(counts["budget_retained"], counts["gold_eligible"]),
                  candidate_precision=_ratio(counts["gold_candidate_triples"], counts["candidate_triples"]),
                  runtime_seconds=runtime, order=dict(order),
                  macro_precision=_ratio(sum(language_precisions), len(language_precisions)),
                  macro_precision_n=len(language_precisions))
    for kind in pool:
        pool[kind]["mean"] = _ratio(pool[kind]["total"], counts["eligible"] * 3)
    result["pools"] = pool
    return result


def _internal_checks(payload, method):
    baseline = aggregate_metrics(payload["rows"], "A", internal=True)
    candidate = aggregate_metrics(payload["rows"], method, internal=True)
    no_evidence = [row for row in payload["rows"] if row["no_evidence"]]
    def unsupported(name):
        return sum(row["methods"][name]["scores"]["claimed"]
                   + row["methods"][name]["scores"]["unsupported_claimed"] for row in no_evidence)
    return {"correct_not_lower": candidate["correct"] >= baseline["correct"],
            "wrong_not_higher": candidate["wrong"] <= baseline["wrong"],
            "no_evidence_claims_not_higher": unsupported(method) <= unsupported("A")}


def evaluate_methods(real_payload, internal_payload):
    """Compute the fixed gates, without reading or accepting final outcomes."""
    methods = {}
    for method in METHODS[1:]:
        details, precisions, increases = {}, [], []
        for seed in SEEDS:
            matched = [r for r in real_payload["rows"] if r["code"] in KNOWN and r["seed"] == seed and not r["mismatch"]]
            shifted = [r for r in real_payload["rows"] if r["code"] in KNOWN and r["seed"] == seed and r["mismatch"]]
            a, b = aggregate_metrics(matched, "A"), aggregate_metrics(matched, method)
            na, nb = aggregate_metrics(shifted, "A"), aggregate_metrics(shifted, method)
            precision_ok = a["precision"] is not None and b["precision"] is not None and b["precision"] >= a["precision"]
            checks = {"correct_plus_three": b["correct"] >= a["correct"] + 3,
                      "wrong_not_higher": b["wrong"] <= a["wrong"], "precision_not_lower": precision_ok,
                      "budget_retention_higher": b["budget_retained"] > a["budget_retained"],
                      "mismatch_claims_not_higher": nb["claimed"] <= na["claimed"],
                      "mismatch_high_confidence_errors_not_higher": nb["high_confidence_wrong"] <= na["high_confidence_wrong"],
                      "mismatch_order_claims_zero": nb["order"].get("claimed", 0) == 0}
            details[str(seed)] = {"checks": checks, "passed": all(checks.values()),
                                  "correct_delta": b["correct"] - a["correct"], "wrong_delta": b["wrong"] - a["wrong"],
                                  "precision": b["precision"], "baseline_precision": a["precision"]}
            precisions.append(b["precision"] if b["precision"] is not None else -1.0)
            increases.append(b["correct"] - a["correct"])
        internal = _internal_checks(internal_payload, method)
        methods[method] = {"seeds": details, "internal": internal,
                           "passed": all(r["passed"] for r in details.values()) and all(internal.values()),
                           "rank": [min(precisions), min(increases)]}
    candidates = [method for method in METHODS[1:] if methods[method]["passed"]]
    candidates.sort(key=lambda method: (-methods[method]["rank"][0], -methods[method]["rank"][1], METHODS.index(method)))
    return {"candidate": candidates[0] if candidates else None, "methods": methods}


def _table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |", "|" + "---|" * len(headers),
                      *("| " + " | ".join(map(cell, row)) + " |" for row in rows)])


def _percent(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def _number(value):
    return "—" if value is None else f"{value:.2f}"


def _counts(value):
    return f"{value['correct']} / {value['wrong']} / {value['abstain']}"


def _group_rows(payload, seed, shifted=False, galician=False):
    return [row for row in payload["rows"] if row["seed"] == seed and row["mismatch"] == shifted
            and (row["code"] == "gl_pud" if galician else row["code"] in KNOWN)]


def final_decision(final, internal_final, regression, freeze):
    """Audit the frozen candidate; final data cannot nominate another candidate."""
    candidate = freeze["selection"]["candidate"]
    gates = evaluate_methods(final, internal_final)
    regression_ok = (regression.get("cases") == 27 and regression.get("all_predictions_equal") is True
                     and regression.get("all_scores_equal") is True)
    galician = []
    if candidate is not None:
        for seed in SEEDS:
            for shifted in (False, True):
                rows = _group_rows(final, seed, shifted, True)
                baseline = aggregate_metrics(rows, "A")["wrong"]
                proposed = aggregate_metrics(rows, candidate)["wrong"]
                galician.append({"seed": seed, "mismatch": shifted, "baseline_wrong": baseline,
                                 "candidate_wrong": proposed, "passed": proposed <= baseline})
    galician_ok = all(case["passed"] for case in galician) if candidate is not None else None
    accepted = bool(candidate and gates["methods"][candidate]["passed"] and galician_ok and regression_ok)
    return {"frozen_candidate": candidate, "adopted": accepted,
            "decision_scope": "registered experimental acceptance; standard integration is a separate step",
            "standard_policy": "A", "standard_integration_performed": False,
            "final_registered_gates": gates["methods"],
            "galician_guard": {"applicable": candidate is not None, "passed": galician_ok, "cases": galician},
            "regression_guard": regression_ok, "freeze_sha256": freeze["freeze_sha256"]}


def render_report(dev, final, internal_dev, internal_final, regression, freeze, manifest, qa=None):
    for payload, phase in ((dev, "dev"), (final, "final")):
        validate_real(payload, phase, manifest)
    for payload, phase in ((internal_dev, "dev"), (internal_final, "final")):
        validate_internal(payload, phase)
    if freeze["selection"] != evaluate_methods(dev, internal_dev):
        raise ValueError("Frozen development choice differs from development-only selection")
    candidate = freeze["selection"]["candidate"]
    decision = final_decision(final, internal_final, regression, freeze)
    final_gates = decision["final_registered_gates"]
    accepted = decision["adopted"]
    conclusion = (f"開発時に固定した{candidate}は登録した最終条件も満たした。後続の標準統合を検討するための候補とする。"
                  if accepted else "登録した採用条件を満たす改善の証拠は得られなかった。標準方式Aを維持する。")
    text = ["# PROJECT BABEL：候補回収の改善実験と独立検証の状況", conclusion,
            "[事前登録](PREREG_ROLE_RECOVERY.md)に従い、品詞制約の軟化B、証拠に応じた閾値C、3役の共同推定D、"
            "複数候補と保留を組み合わせたEを、現行再現Aと比較した。これは実験用の3役推論部品であり、標準解読器の既定動作は変更していない。",
            "## 1. 根本原因と評価の境界",
            "推定品詞との衝突を完全に禁止すると誤ったクラス推定を後から回復できない。双方向事後確率の積に対する"
            "固定0.25は、長い文・不完全な主節対訳・低頻度語で拡散した対応を強く除外する。また、役割ごとの先行位置の選択は"
            "反復語や別の節を混ぜうる。候補の追加だけでは、この識別問題の解決を示さない。",
            "PUDの英語記事単位で開発70%・最終30%を固定し、20言語の同じ記事を同じ側に置いた。各分割の対訳の中で"
            "再学習するtransductive比較であり、学習済み語彙の未知文への帰納的転移ではない。全PUDは既に調査した既知資料で、"
            "新しい独立初見評価ではない。語境界とUD品詞による入力清掃は既存方式を引き継ぐ。",
            "以前の10/3,864は全文を一緒に学習した診断である。今回の各分割の結果とは分母と学習資料が異なるため、"
            "10件との差を改善量として扱わず、同一入力のAと比較する。UDの直接nsubj–VERB–objの一致は構造的一致で、英語主節との訳義一致を保証しない。",
            "## 2. 実装した方式",
            _table(["方式", "変更"], [("A", "硬い品詞制約、0.25、最良概念、格接辞解決、先行位置を再現"),
                                         ("B", "品詞の衝突を重み0.2とする双方向EMで再学習。閾値・選択はA"),
                                         ("C", "独立文書の支持dと対応割合qからr=min(1,d/3)q、閾値0.25−0.20r"),
                                         ("D", "各役割上位5位置から、強度の積/(1+範囲長)最大の異なる3位置を選択"),
                                         ("E", "B+Cと共同推定。上位5組内の比0.8以上、各対応3文書以上の支持がある時だけ主張")]),
            "Cの証拠指標とEの0.8は較正済みの正解確率ではない。誤対応での主張・高代理信頼度の誤答を直接数える。"
            "A/B/Cの実際の先行位置選択は、比較用の上位5候補への切り詰めでは変更しない。",
            "## 3. 実在言語：改善前後、誤答と保留",
            "主分母Nは英語側に3役がある全文。対象側に正解組がない文への出力も誤答に含む。正/誤/保は文数、"
            "適合率=正/全出力、誤答率=誤/全出力、保留率=保/N。分母が0の率は「—」。候補回収率は別の分母G"
            "（Nのうち清掃後の対象側に直接3役の組がある文）を使う。換字シード0/1/2は記号名と同点処理への頑健性を測り、"
            "独立標本として合算しない。以下は既知19言語の集計。",
            _table(["分割", "換字seed", "方式", "N", "G", "生の回収率", "上位5回収率", "正 / 誤 / 保", "適合率", "誤答率", "保留率"],
                   [(phase, seed, method, value["eligible"], value["gold_eligible"], _percent(value["raw_recall"]),
                     _percent(value["budget_recall"]), _counts(value), _percent(value["precision"]),
                     _percent(value["wrong_rate"]), _percent(value["abstain_rate"]))
                    for phase, payload in (("開発", dev), ("最終", final)) for seed in SEEDS for method in METHODS
                    for value in [aggregate_metrics(_group_rows(payload, seed), method)]]),
            "### 候補の拡大と計算時間",
            "候補位置平均はNの全文・3役で加重した1役あたりの数。p95最大は各言語・各役割の95百分位の最大値で、"
            "全文を結合したp95ではない。候補組適合率は上位5の全候補組のうちUDに一致する組の割合。"
            "比較秒は共通前処理＋当該方式の硬い/軟らかい学習＋当該選択処理の合計で、共有計算を各方式に帰属させた値。"
            "実験全体の実際の壁時計時間や、標準エンジンの全帰無検定の時間ではない。",
            _table(["分割", "seed", "方式", "位置平均 生 / 上位5", "p95最大 生 / 上位5", "候補組数", "候補組適合率", "比較秒"],
                   [(phase, seed, method,
                     f"{_number(value['pools']['raw']['mean'])} / {_number(value['pools']['budget']['mean'])}",
                     f"{value['pools']['raw']['p95_max']} / {value['pools']['budget']['p95_max']}",
                     value["candidate_triples"], _percent(value["candidate_precision"]), _number(value["runtime_seconds"]))
                    for phase, payload in (("開発", dev), ("最終", final)) for seed in SEEDS for method in METHODS
                    for value in [aggregate_metrics(_group_rows(payload, seed), method)]]),
            "### 一文ずらした誤対応",
            "以下は既知19言語。対象側で偶然構造が一致した組はこの採点の正解に数えるが、誤対応を正しく翻訳したとは扱わない。"
            "語順の主張は共通の5票・二項検定規則による副次指標で、標準エンジン全体の最終判定ではない。",
            _table(["分割", "seed", "方式", "出力", "正 / 誤", "高代理信頼度の誤答", "保留率", "語順主張"],
                   [(phase, seed, method, value["claimed"], f"{value['correct']} / {value['wrong']}",
                     value["high_confidence_wrong"], _percent(value["abstain_rate"]), value["order"].get("claimed", 0))
                    for phase, payload in (("開発", dev), ("最終", final)) for seed in SEEDS for method in METHODS
                    for value in [aggregate_metrics(_group_rows(payload, seed, True), method)]]),
            "### ガリシア語：別集計",
            _table(["分割", "seed", "方式", "通常 正 / 誤 / 保", "通常適合率", "通常保留率", "通常上位5回収率", "誤対応出力 / 誤答"],
                   [(phase, seed, method, _counts(value), _percent(value["precision"]), _percent(value["abstain_rate"]),
                     _percent(value["budget_recall"]), f"{shifted['claimed']} / {shifted['wrong']}")
                    for phase, payload in (("開発", dev), ("最終", final)) for seed in SEEDS for method in METHODS
                    for value in [aggregate_metrics(_group_rows(payload, seed, False, True), method)]
                    for shifted in [aggregate_metrics(_group_rows(payload, seed, True, True), method)]]),
            "### 最終比較の言語別内訳（換字seed 0、通常対訳）",
            _table(["言語", "N", *(f"{method} 正 / 誤 / 保" for method in METHODS)],
                   [(code, values["A"]["eligible"], *(_counts(values[method]) for method in METHODS))
                    for code in (*KNOWN, "gl_pud")
                    for selected in [[r for r in final["rows"] if r["code"] == code and r["seed"] == 0 and not r["mismatch"]]]
                    for values in [{method: aggregate_metrics(selected, method) for method in METHODS}]]),
            "## 4. 採用判定と失敗の記録",
            f"開発結果だけで固定した候補は **{candidate or 'なし'}**。最終結果で方式を選び直していない。"
            f"開発凍結ハッシュは `{freeze['freeze_sha256']}`。",
            _table(["方式", "開発seed0", "開発seed1", "開発seed2", "内部開発", "最終seed0", "最終seed1", "最終seed2", "内部最終"],
                   [(method, *("合格" if freeze["selection"]["methods"][method]["seeds"][str(seed)]["passed"] else "不合格" for seed in SEEDS),
                     "合格" if all(freeze["selection"]["methods"][method]["internal"].values()) else "不合格",
                     *("合格" if final_gates[method]["seeds"][str(seed)]["passed"] else "不合格" for seed in SEEDS),
                     "合格" if all(final_gates[method]["internal"].values()) else "不合格") for method in METHODS[1:]]),
            "不合格条件の詳細は development-decision.json と final-decision.json の各checksに保存する。採用候補の固定後に最終条件を判定するが、"
            "採用候補がなければ、最終で良く見える方式があっても標準へ昇格させない。",
            conclusion,
            "## 5. 内部比較と既存機能の回帰",
            "KES/UMB/SOT、開発seed0/1/2・最終seed600/601/602、65文書で各分割45条件を比較した。"
            "合成言語の正解は役割だけでなく対訳概念との一致も要求する。同一の生成器に基づくため独立した外部検証ではない。",
            _table(["分割", "条件", "方式", "正 / 誤 / 保", "適合率", "上位5回収率", "意味証拠なしの主張"],
                   [(phase, label, method, _counts(value), _percent(value["precision"]), _percent(value["budget_recall"]),
                     value["claimed"] + value["unsupported_claimed"] if label == "意味証拠なし" else "—")
                    for phase, payload in (("開発", internal_dev), ("最終", internal_final))
                    for label, subset in (("通常対訳", [r for r in payload["rows"] if not r["mismatch"] and not r["no_evidence"]]),
                                          ("誤対応", [r for r in payload["rows"] if r["mismatch"]]),
                                          ("意味証拠なし", [r for r in payload["rows"] if r["no_evidence"]]))
                    for method in METHODS for value in [aggregate_metrics(subset, method, internal=True)]]),
            f"標準エンジンの33文書×3系統×3seed×LEVEL A/B/Cの27条件では、時間以外の予測一致は"
            f"**{'全件一致' if regression.get('all_predictions_equal') else '不一致あり'}**、既存scoreの全指標は"
            f"**{'全件一致' if regression.get('all_scores_equal') else '不一致あり'}**。"
            "これは公開済み1,975実行の全再測定ではなく、今回の変更による標準機能の対比較である。",
            "## 6. 独立ブラインドテスト",
            "**未実施**。現存のexternal/inbox全10件は過去の予測があり、新しい未見の第三者資料は提供されていない。"
            "PUDの予測は採点前にファイルとハッシュを固定したが、既に公開・調査済みの注釈なので独立ブラインド試験とは呼ばない。"
            "過去の修正版を初回結果として扱ったり、開発中にAIが自作した言語で代用したりしていない。",
            "## 7. 未解決の問題",
            "候補の量、1組を正しく選ぶ能力、不確実性を認めて保留する能力は別々である。低頻度語、複数節、"
            "英語主節と対象側の意味対応、推定品詞の誤りを同時に解消できたとは示していない。代理信頼度は較正済み確率ではなく、"
            "小さい分割の出力件数や換字seed間の変動も残る。全PUDで既知の失敗を踏まえた設計なので、"
            "この記事分離だけで未知言語への一般化を証明できない。",
            "## 8. 次に優先する研究",
            "最優先は、英語主節と対象側の同じ意味の節・役割を独立に注釈した資料を用意し、"
            "語彙対応と節の取り違えを分けて測ること。その資料の正解公開前に予測とハッシュを固定し、"
            "候補集合の回収率に加えて1組の出力適合率と誤対応への保留を検証する。"
            "合格・不合格の両方を、結果を見た閾値の再調整で置き換えず記録として保持する。",
            "## 再現用ファイルと検証",
            "[記事分割](../results/role_recovery/article-split.json)、[開発結果](../results/role_recovery/dev.json)、"
            "[最終比較](../results/role_recovery/final.json)、[開発判定](../results/role_recovery/development-decision.json)、"
            "[開発凍結](../results/role_recovery/development-freeze.json)、[最終判定](../results/role_recovery/final-decision.json)、"
            "[内部開発](../results/role_recovery/internal-dev.json)、"
            "[内部最終](../results/role_recovery/internal-final.json)、[標準回帰](../results/role_recovery/standard-regression.json)。"
            "各結果には入力・予測・分割のハッシュ、実行時間を保存している。詳細予測は.cache/role_recovery/lockedに保存し、"
            "異なる内容での上書きを拒否する。推論部品はpublicだけを受け取り、注釈採点はexperiments側で行う。",
            "```sh\npython -m babel.experiments.role_recovery --prepare\n"
            "PYTHONHASHSEED=0 python -m babel.experiments.role_recovery --phase dev --jobs 2\n"
            "PYTHONHASHSEED=0 python -m babel.experiments.role_recovery_internal --phase dev --jobs 2\n"
            "python -m babel.experiments.role_recovery_report --freeze\n"
            "PYTHONHASHSEED=0 python -m babel.experiments.role_recovery --phase final --jobs 2\n"
            "PYTHONHASHSEED=0 python -m babel.experiments.role_recovery_internal --phase final --jobs 2\n"
            "python -m babel.experiments.role_recovery_report\n```",
            "既存の結果・予測・凍結は上書きしない。再現確認は別の出力先へ実行し、時間を除く内容とハッシュを照合する。"]
    if qa:
        text.extend(["### 品質保証の実行記録", _table(["検査", "結果"], [(key, value) for key, value in sorted(qa.items())])])
    return "\n\n".join(text) + "\n"


def _read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="Select using development data only, then lock code/results")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/ROLE_RECOVERY_RESULTS.md")
    args = parser.parse_args()
    manifest = _read("article-split.json")
    dev, internal_dev = _read("dev.json"), _read("internal-dev.json")
    validate_real(dev, "dev", manifest)
    validate_internal(internal_dev, "dev")
    if args.freeze:
        selection = evaluate_methods(dev, internal_dev)
        path = OUT / "development-decision.json"
        if path.exists() and json.loads(path.read_text()) != selection:
            raise ValueError("A different development decision is already fixed")
        path.write_text(canonical(selection) + "\n", encoding="utf-8")
        frozen = freeze_selection(selection)
        print("Development candidate:", selection["candidate"], "freeze:", frozen["freeze_sha256"])
        return
    frozen = check_freeze()
    qa_path = OUT / "quality-assurance.json"
    qa = json.loads(qa_path.read_text()) if qa_path.exists() else None
    final, internal_final, regression = _read("final.json"), _read("internal-final.json"), _read("standard-regression.json")
    report = render_report(dev, final, internal_dev, internal_final, regression, frozen, manifest, qa)
    decision = final_decision(final, internal_final, regression, frozen)
    decision_path = OUT / "final-decision.json"
    if decision_path.exists() and json.loads(decision_path.read_text()) != decision:
        raise ValueError("A different final decision is already fixed")
    decision_path.write_text(canonical(decision) + "\n", encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
