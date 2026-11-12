"""回帰検査: 1文に1回現れる数詞クラスを動詞と誤認しない。"""
import json
from pathlib import Path

import pytest

from babel.decipher import classes
from babel.decipher.engine import decipher


@pytest.mark.parametrize("numeral_coverage", [0.8, 1.0])
def test_verb_group_tie_uses_coverage_then_type_diversity(numeral_coverage):
    """数詞と動詞はどちらも文内に出たときは1回。そのうち動詞は全ての文に出る。"""
    sentences, infos, groups = [], {}, {}
    for i in range(80):
        verb = "verb" + str(i % 10)
        subject = "subject" + str(i % 9)
        obj = "object" + str(i % 9)
        words = [verb, subject]
        if numeral_coverage == 1.0 or i % 5:
            words.append("num" + str(i % 5))
        words.append(obj)
        sentences.append(words)
        for w, group in ((verb, "P:tense"), (subject, "P:case"),
                         (obj, "P:case")):
            infos[w] = ([], w, [])
            groups[w] = group
        if len(words) == 4:
            w = words[2]
            infos[w] = ([], w, [])
            groups[w] = "S:numeral"
    pos, info = classes.label_classes(sentences, infos, groups)
    assert info["verb_group"] == "P:tense"
    assert pos["verb0"] == "V"
    assert pos["num1"] != "V"
    # 数詞も1回/文の強い競合候補なので、動詞クラスの識別は確実ではない。
    assert info["verb_confidence"] <= 0.5
    v = classes.verb_position(sentences, infos, pos, info["verb_confidence"])
    assert v["value"] == "initial"
    assert v["confidence"] <= 0.5


def test_previously_blind_outsider_is_now_a_regression_only():
    """初見記録は公開済み。ここでは開発用回帰データとしてのみ再利用する。

    v1.2.0 は数詞 S:zhor を動詞に選び、実際の VSO 文を 'medial' と誤答した。
    このテストによる修正後の数値は独立初見の研究スコアに加えない。
    """
    path = (Path(__file__).parents[1] / "external" / "inbox"
            / "outsider-vso-prefix-7-v1" / "corpus.json")
    pub = json.loads(path.read_text(encoding="utf-8"))
    pred = decipher(pub)
    assert pred["classes"]["verb_group"] == "P:shai"
    vp = pred["structure"]["verb_position"]
    assert vp["value"] == "initial"
    assert vp["confidence"] < 1.0
    assert pred["split_used"]["heldout"] == [f"d{i:03d}" for i in range(127, 161)]
