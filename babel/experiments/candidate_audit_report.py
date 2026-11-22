"""Validate candidate-audit summaries and generate a deterministic Japanese report."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from babel.experiments import real_parallel as rp
from babel.experiments.candidate_audit import LABELS, STAGES
from babel.experiments.clause_probe import ROLES

STAGE_NAMES = {
    "units": "語幹単位あり",
    "pos_allowed": "推定品詞の制約を通過",
    "positive_link": "正の対応あり",
    "threshold": "対応の強さ ≥ 0.25",
    "best_concept": "語幹ごとの最良概念",
    "resolved_role": "役割を一意に解決",
    "first_position": "役割ごとの最初の位置",
    "retained": "最後まで残存",
}
EXAMPLE_IDS = ("n01052006", "n01087035", "w02011029")


def _count(value, context):
    if type(value) is not int or value < 0:
        raise ValueError(f"{context}: expected a nonnegative integer")
    return value


def _validate_row(row, mismatch):
    code = row["language"]
    if row["mismatch"] is not mismatch:
        raise ValueError(f"{code}: wrong mismatch condition")
    totals = row["totals"]
    for key in ("parallel", "source_full", "source_full_target_raw", "eligible"):
        _count(totals[key], f"{code}/totals/{key}")
    if not totals["eligible"] <= totals["source_full_target_raw"] <= totals["source_full"] <= totals["parallel"]:
        raise ValueError(f"{code}: inconsistent eligibility counts")
    for label in LABELS:
        previous = totals["eligible"]
        losses = row["first_loss"][label]
        for stage in STAGES:
            value = _count(row["stages"][stage][label], f"{code}/{stage}/{label}")
            loss = _count(losses[stage], f"{code}/first_loss/{label}/{stage}")
            if value > previous or loss != previous - value:
                raise ValueError(f"{code}: non-nested totals or inconsistent first losses for {stage}/{label}")
            previous = value
        if _count(losses["retained"], f"{code}/first_loss/{label}/retained") != previous:
            raise ValueError(f"{code}: retained count differs from the final stage")
        if sum(losses[stage] for stage in (*STAGES, "retained")) != totals["eligible"]:
            raise ValueError(f"{code}: first losses do not sum to eligible sentences")
        pool = _count(row["clause_pool"][label], f"{code}/clause_pool/{label}")
        if pool > row["stages"]["threshold"][label]:
            raise ValueError(f"{code}: unique-role pool exceeds the threshold pool")
    for stage in STAGES:
        values = row["stages"][stage]
        if values["triple"] > min(values[label] for label in LABELS[:3]):
            raise ValueError(f"{code}: joint recall exceeds individual role recall")
    if row["clause_pool"]["triple"] > min(row["clause_pool"][label] for label in LABELS[:3]):
        raise ValueError(f"{code}: unique-role joint recall exceeds individual recall")
    for direction, value in row["votes"].items():
        if direction not in {"initial", "medial", "final"}:
            raise ValueError(f"{code}: unknown vote direction")
        _count(value, f"{code}/votes/{direction}")
    if row["stages"]["first_position"]["triple"] > sum(row["votes"].values()):
        raise ValueError(f"{code}: final joint recall exceeds all votes")


def _validate_example(example, sid):
    if not isinstance(example, dict) or example["sent_id"] != sid:
        raise ValueError(f"Japanese example ID differs: {sid}")
    if type(example["source_full"]) is not bool or type(example["eligible"]) is not bool:
        raise ValueError(f"{sid}: example eligibility flags must be boolean")
    kept = _count(example["kept_triples"], f"{sid}/kept_triples")
    if example["eligible"] != bool(example["source_full"] and kept):
        raise ValueError(f"{sid}: inconsistent example eligibility")
    losses = example["first_loss"]
    if example["eligible"]:
        if not isinstance(losses, dict) or set(losses) != set(LABELS):
            raise ValueError(f"{sid}: missing example first-loss labels")
        if any(losses[label] not in (*STAGES, "retained") for label in LABELS):
            raise ValueError(f"{sid}: unknown example first-loss stage")
    elif losses is not None:
        raise ValueError(f"{sid}: ineligible examples must not have first losses")
    selected = example["selected"]
    if not isinstance(selected, dict) or not set(selected).issubset(ROLES):
        raise ValueError(f"{sid}: malformed selected example roles")
    for role, token in selected.items():
        if not isinstance(token, dict):
            raise ValueError(f"{sid}: malformed selected token")
        _count(token["position"], f"{sid}/selected/{role}/position")
        if any(not isinstance(token[key], str) or not token[key] for key in ("form", "upos")):
            raise ValueError(f"{sid}: selected token must have form and UPOS strings")


def validate_inputs(normal, mismatch):
    """Reject incomplete or incompatible measurements before producing prose."""
    expected = {code for _, code in rp.TARGETS} | {"gl_pud"}
    indexed = []
    try:
        for rows, condition in ((normal, False), (mismatch, True)):
            codes = [row["language"] for row in rows]
            if len(codes) != len(set(codes)) or set(codes) != expected:
                raise ValueError("Each condition must contain exactly one row for all 20 target languages")
            for row in rows:
                _validate_row(row, condition)
            indexed.append({row["language"]: row for row in rows})
        first = normal[0]
        for row in normal + mismatch:
            for key in ("source_sha256", "ud_tag", "engine_version", "config"):
                if row[key] != first[key]:
                    raise ValueError(f"{row['language']}: incompatible {key}")
            for key in ("source_sha256", "target_sha256"):
                value = row[key]
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError(f"{row['language']}: invalid {key}")
        for code in expected:
            if indexed[0][code]["target_sha256"] != indexed[1][code]["target_sha256"]:
                raise ValueError(f"{code}: target hashes differ between conditions")
        if first["config"].get("v16_gates") is not False:
            raise ValueError("This diagnostic requires v16_gates=False")
        for condition in indexed:
            for sid in EXAMPLE_IDS:
                _validate_example(condition["ja_pud"]["examples"][sid], sid)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing or malformed candidate-audit data: {exc}") from exc
    return indexed


def _aggregate(rows):
    result = {"totals": Counter(), "stages": {stage: Counter() for stage in STAGES},
              "first_loss": {label: Counter() for label in LABELS}, "clause_pool": Counter(), "votes": Counter()}
    for row in rows:
        for key in ("totals", "clause_pool", "votes"):
            result[key].update(row[key])
        for stage in STAGES:
            result["stages"][stage].update(row["stages"][stage])
        for label in LABELS:
            result["first_loss"][label].update(row["first_loss"][label])
    return result


def _table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                      "|" + "---|" * len(headers),
                      *("| " + " | ".join(map(cell, row)) + " |" for row in rows)])


def _ratio(numerator, denominator):
    return f"{numerator}/{denominator}（{100 * numerator / denominator:.1f}%）" if denominator else "0/0（算出不可）"


def render_report(normal, mismatch):
    """Return report text; all measured values are derived from validated summaries."""
    correct, control = validate_inputs(normal, mismatch)
    codes = [code for _, code in rp.TARGETS]
    known = _aggregate([correct[code] for code in codes])
    known_control = _aggregate([control[code] for code in codes])
    groups = [("既知19言語", known, known_control),
              ("日本語（上記の内数）", correct["ja_pud"], control["ja_pud"]),
              ("ガリシア語（別集計）", correct["gl_pud"], control["gl_pud"])]
    eligible = known["totals"]["eligible"]
    text = [
        "# 主語・述語・目的語の候補が消える段階",
        "[診断計画](PLAN_CANDIDATE_AUDIT.md)に従い、既知のPUD 19言語と、調査済みのガリシア語を通常対訳・"
        "一文ずらした対訳の40条件で追跡した。新しい解読方式の採用試験でも、初見言語の評価でもない。",
        f"通常対訳の既知19言語では、同じ直接3役の組が閾値段階に残った文は"
        f"{_ratio(known['stages']['threshold']['triple'], eligible)}、最良概念の選択後は"
        f"{_ratio(known['stages']['best_concept']['triple'], eligible)}、最終位置の選択後は"
        f"{_ratio(known['stages']['first_position']['triple'], eligible)}だった。",
        f"3役の組を初めて失った文は、推定品詞の制約で{known['first_loss']['triple']['pos_allowed']}文、"
        f"強さ0.25の閾値で{known['first_loss']['triple']['threshold']}文、最初の位置の選択で"
        f"{known['first_loss']['triple']['first_position']}文だった。最良概念の選択と役割解決の間では、"
        "3役の組が残る文数は減らなかった。候補集合を固定する限り、位置の選び直しだけで回収できる文数には、"
        f"役割解決後の{known['stages']['resolved_role']['triple']}文という上限がある。これは上限の診断であり、"
        "その文を実際に回収する方式を今回実装・評価したわけではない。",
        "## 分母と測定の意味",
        "採点対象の組は、対象言語のUD注釈でVERBとその直接の `nsubj`・`obj` を結ぶ3語。"
        "英語対訳に主語・述語・目的語があり、対象の清掃後の語列にその組が少なくとも1組残る文を、各条件の固定分母にした。"
        "清掃前から対象文自体がコーパスに採用されなかった例は、この分母と下表に含まれない。"
        "語境界と、UD品詞による一部トークンの除外を含む入力清掃は前回と同じであり、"
        "生の未知文字資料からこれらを推定できるかは今回検証していない。",
        _table(["対象", "対訳条件", "全対訳文", "英語側3役あり", "対象の元文にも組あり", "清掃後の固定分母", "清掃で全組を失った文"],
               [(name, condition, row["totals"]["parallel"], row["totals"]["source_full"],
                 row["totals"]["source_full_target_raw"], row["totals"]["eligible"],
                 row["totals"]["source_full_target_raw"] - row["totals"]["eligible"])
                for name, aligned, shifted in groups for condition, row in (("通常", aligned), ("ずらす", shifted))]),
        "役割別の数は、少なくとも1組の採点対象に属するその役割の語が候補にある文の数。"
        "主語・述語・目的語の各列は別の組に由来してもよい。3役の列だけは、同じ組の3語がそろうことを要求する。"
        "いずれも文数であり、語数や組の総数ではない。",
        "この一致は構造的な回収可能性を表す。英語の主節と異なる対象側の節も採点対象になるため、"
        "英語との訳義の一致や翻訳精度ではない。ずらした対訳でも、対象側の構造が偶然そろうことがある。",
        "## 通常対訳の段階別結果",
    ]
    for name, row, _ in groups:
        text.extend([
            f"### {name}（固定分母 {row['totals']['eligible']} 文）",
            _table(["段階", "主語", "述語", "目的語", "同じ組の3役"],
                   [(f"{STAGE_NAMES[stage]} (`{stage}`)", *(row["stages"][stage][label] for label in LABELS))
                    for stage in STAGES]),
        ])
    text.extend([
        "すべての段階は、同じ学習済みモデルの対応の強さと制約を固定して調べた入れ子の候補集合。"
        "`pos_allowed` は教師なしで推定した品詞の制約を使う。"
        "既存の入力清掃後の候補の絞り込みには、対象側UDの品詞を使わない。"
        "ここでの脱落数は、制約を外して再学習した場合の改善効果ではない。",
        "`threshold` は役割に対応しうる概念への強さが0.25以上の候補。`best_concept` では、"
        "語幹ごとに解読器と同じ最大値・同点処理で1概念を選ぶ。曖昧な役割は学習した格接辞で"
        "一意に絞れた場合だけ `resolved_role` に残す。最後に役割ごとに最初の位置だけを選ぶ。",
        "## ずらした対訳との比較",
        "各セルは同じ組の3役が残る文数。ずらした対訳では英語側3役が割り当てられる対象文が変わるため、"
        "通常対訳と固定分母が異なりうる。",
        _table(["対象", "対訳条件", "固定分母", *(f"`{stage}`" for stage in STAGES)],
               [(name, condition, row["totals"]["eligible"], *(row["stages"][stage]["triple"] for stage in STAGES))
                for name, aligned, shifted in groups for condition, row in (("通常", aligned), ("ずらす", shifted))]),
        "## 最初に候補がなくなった段階",
        f"通常対訳の既知19言語、固定分母{eligible}文を各列で1回ずつ数える。"
        "「3役」は、同じ組が少なくとも1組残る状態を初めて失った段階。各役割が別の組に残っていても、3役は脱落しうる。",
        _table(["最初の脱落段階", "主語", "述語", "目的語", "同じ組の3役"],
               [(f"{STAGE_NAMES[stage]} (`{stage}`)", *(known["first_loss"][label][stage] for label in LABELS))
                for stage in (*STAGES, "retained")]),
        "## 言語別の通常対訳",
        _table(["言語", "固定分母", "閾値通過の3役", "最良概念後の3役", "役割解決後の3役", "最初の位置での3役"],
               [(code + ("（別集計）" if code == "gl_pud" else ""), correct[code]["totals"]["eligible"],
                 *(correct[code]["stages"][stage]["triple"]
                   for stage in ("threshold", "best_concept", "resolved_role", "first_position")))
                for code in [*codes, "gl_pud"]]),
        "## 前回の候補集合との照合",
        "[句読点実験](CLAUSE_PROBE.md)の候補集合は、役割が一意の概念だけに強さ0.25以上で結び付く語を含む。"
        "今回の入れ子の段階とは別に集計した。曖昧な概念を格接辞で解決する経路を含まないため、"
        "`threshold` から `best_concept` の間に置く段階ではない。下表は通常対訳での文数。",
        _table(["対象", "主語", "述語", "目的語", "同じ組の3役"],
               [(name, *(row["clause_pool"][label] for label in LABELS)) for name, row, _ in groups]),
        "## 日本語の既報3文",
        "選択語は解読器が最初に割り当てた主語・述語・目的語。脱落段階は、その語の誤りの原因ではなく、"
        "UDの採点対象の候補を最後まで残せたかを示す。固定分母の対象外なら、段階別の脱落は集計しない。",
    ])
    examples = correct["ja_pud"]["examples"]
    example_rows = []
    for sid in EXAMPLE_IDS:
        example = examples[sid]
        selected = [example["selected"].get(role, {}).get("form", "未選択") for role in ROLES]
        eligibility = ("対象" if example["eligible"] else
                       f"対象外（英語3役={'あり' if example['source_full'] else 'なし'}、残る組={example['kept_triples']}）")
        losses = example["first_loss"]
        loss_text = " / ".join(STAGE_NAMES[losses[label]] for label in LABELS) if losses else "集計対象外"
        example_rows.append((sid, *selected, eligibility, loss_text))
    text.extend([
        _table(["文ID", "選択した主語", "選択した述語", "選択した目的語", "固定分母", "主語 / 述語 / 目的語 / 3役の最初の脱落"], example_rows),
        "## 再現と記録",
        f"解読器の版は `{normal[0]['engine_version']}`、UDは `{normal[0]['ud_tag']}`。"
        "入力のSHA-256、実効設定、言語別の集計、既報3文の記録はJSONLに保存した。"
        "集計の生成時に20言語の不足・重複、条件、入力ハッシュ・設定の不一致、候補数の非単調性、"
        "最初の脱落数の保存則を検査する。診断の実行時には選択位置を実際の解読器の行と照合し、"
        "復元した票数・向きが解読器の集計と一致することも確認する。",
        "解読器本体は変更していない。最初の対応付けの候補と票だけを測定し、"
        "帰無対照の再学習を省くため `v16_gates=False` とした。最終的な言語別の語順判定や、"
        "閾値を変えた解読器の性能は今回測定していない。",
        "```sh\n"
        "PYTHONHASHSEED=0 python -m babel.experiments.candidate_audit --jobs 8 --out results/real_parallel/candidate-audit.jsonl\n"
        "PYTHONHASHSEED=0 python -m babel.experiments.candidate_audit --jobs 8 --mismatch --out results/real_parallel/candidate-audit-mismatch.jsonl\n"
        "python -m babel.experiments.candidate_audit_report\n"
        "```",
        "全対訳文の追跡を再生成する場合は、診断コマンドに `--details-out` を追加する（通常の集計には含めない）。",
        "```sh\n"
        "PYTHONHASHSEED=0 python -m babel.experiments.candidate_audit --code ja_pud --jobs 1 "
        "--out /tmp/ja-candidate-summary.jsonl --details-out /tmp/ja-candidate-details.jsonl\n"
        "```",
        "[通常対訳の測定値](../results/real_parallel/candidate-audit.jsonl) · "
        "[ずらした対訳の測定値](../results/real_parallel/candidate-audit-mismatch.jsonl) · "
        "[診断コード](../babel/experiments/candidate_audit.py) · "
        "[報告生成コード](../babel/experiments/candidate_audit_report.py)",
    ])
    return "\n\n".join(text) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, default=Path("results/real_parallel/candidate-audit.jsonl"))
    parser.add_argument("--mismatch", type=Path, default=Path("results/real_parallel/candidate-audit-mismatch.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("docs/CANDIDATE_AUDIT.md"))
    args = parser.parse_args()
    normal, mismatch = [[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                        for path in (args.normal, args.mismatch)]
    report = render_report(normal, mismatch)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
