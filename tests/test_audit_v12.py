"""独立監査（v1.1.1）の指摘4件の回帰テスト。"""
import copy
import json

import pytest

from babel import external
from babel.decipher.engine import Decipherer, decipher
from babel.evaluate.score import score
from babel.generator.corpus import build_corpus


# ---- 問題1: 不正な分割指定で評価用の文書が学習に混入しない --------------------------------
@pytest.fixture(scope="module")
def base():
    return build_corpus("KES", 3, "C", n_docs=20)[0]


def _engine(base, split=None, dup=False):
    p = copy.deepcopy(base)
    if dup:
        extra = copy.deepcopy(p["documents"][9])
        extra["id"] = "d002"
        p["documents"].append(extra)
    if split is None:
        p.pop("split", None)
    else:
        p["split"] = split
    return Decipherer(p)


@pytest.mark.parametrize("name, split", [
    ("train が全部存在しないID", {"train": ["x1", "x2"], "heldout": "H"}),
    ("train に存在しないIDが一部", {"train": "T10+x", "heldout": "H"}),
    ("train と heldout に同じID", {"train": "T17", "heldout": "H"}),
    ("heldout のみ", {"heldout": "H"}),
    ("train が空リスト", {"train": [], "heldout": "H"}),
])
def test_heldout_never_leaks_into_training(base, name, split):
    ids = [d["id"] for d in base["documents"]]
    resolve = {"H": ids[16:], "T10+x": ids[:10] + ["x1"], "T17": ids[:17]}
    sp = {k: resolve.get(v, v) if isinstance(v, str) else v for k, v in split.items()}
    d = _engine(base, sp)
    assert not set(sp["heldout"]) & set(d.train_ids), name
    assert set(d.heldout_ids) == set(sp["heldout"]), name
    if not [x for x in sp.get("train", []) if x in ids and x not in sp["heldout"]]:
        assert d.input_warnings, "有効な train がないのに黙って分割を変えた"


def test_train_only_and_unspecified_split(base):
    ids = [d["id"] for d in base["documents"]]
    d = _engine(base, {"train": ids[:16]})
    assert set(d.train_ids) == set(ids[:16]) and d.heldout_ids == []
    d = _engine(base, None)
    assert set(d.train_ids) == set(ids) and d.heldout_ids == []


def test_all_invalid_split_is_reported_not_silent(base):
    d = _engine(base, {"train": ["x"], "heldout": ["y"]})
    assert d.pub["_split_status"] == "invalid" and d.input_warnings


def test_duplicate_document_ids_keep_both_contents(base):
    ids = [d["id"] for d in base["documents"]]
    d = _engine(base, {"train": ids[:16], "heldout": ids[16:]}, dup=True)
    assert len(d.tokens["d002"]) == len(base["documents"][1]["sentences"])     # 元の d002 が上書きされない
    assert "d002#dup1" in d.tokens and "d002#dup1" in d.train_ids
    d = _engine(base, {"train": ids[2:16], "heldout": ["d002"] + ids[16:]}, dup=True)
    assert {"d002", "d002#dup1"} <= set(d.heldout_ids) and not {"d002", "d002#dup1"} & set(d.train_ids)


def test_split_used_is_recorded_in_prediction(base):
    ids = [d["id"] for d in base["documents"]]
    p = copy.deepcopy(base)
    p["split"] = {"heldout": ids[16:]}
    su = decipher(p)["split_used"]
    assert su["status"] == "train_derived_from_heldout" and su["train"] == 16


# ---- 問題2: 既知語を推定した語の正解に数えない ------------------------------------------
def test_given_words_are_separated_in_table_scoring():
    public = {"documents": [{"id": "d1", "sentences": ["ka ta"]}], "known": [{"form": "ka", "meaning": "king"}]}
    pred = {"lexicon": [
        {"stem": "ka", "forms": ["ka"], "status": "claimed", "meaning": "king", "confidence": 0.97, "reason": "既知語として与えられた"},
        {"stem": "ta", "forms": ["ta"], "status": "claimed", "meaning": "give", "confidence": 0.8, "reason": "対訳との対応"},
        {"stem": "mi", "forms": ["mi"], "status": "undetermined", "meaning": None, "confidence": 0.0, "reason": ""}]}
    gold = {"lexicon": [{"form": "ka", "meaning": "king"}, {"form": "ta", "meaning": "give"}, {"form": "mi", "meaning": "sea"}]}
    r = score(pred, gold, public)
    assert r["lexicon"]["correct"] == 2                        # 総合（方式1と同じ定義）は既知語を含む
    li = r["lexicon_inferred"]
    assert li["given_excluded"] == 1 and li["evaluated"] == 2
    assert li["correct"] == 1 and li["claimed"] == 1 and li["accuracy"] == 0.5 and li["coverage"] == 0.5


def test_rescoring_does_not_overwrite_old_score_files(tmp_path):
    public, secret = build_corpus("KES", 4, "C", n_docs=30)
    inbox = tmp_path / "inbox" / "m"
    inbox.mkdir(parents=True)
    (inbox / "corpus.json").write_text(json.dumps(public, ensure_ascii=False), encoding="utf-8")
    external.predict(tmp_path)
    (tmp_path / "answers" / "m").mkdir(parents=True)
    (tmp_path / "answers" / "m" / "answer.json").write_text(json.dumps(secret, ensure_ascii=False), encoding="utf-8")
    old = tmp_path / "scores" / "m" / "attempt-001.json"
    old.parent.mkdir(parents=True)
    old.write_text('{"legacy": true}', encoding="utf-8")
    external.score_all(tmp_path)
    assert old.read_text(encoding="utf-8") == '{"legacy": true}'
    assert (tmp_path / "scores" / "m" / "attempt-001.scoring-v2.json").exists()


# ---- 問題3: 同じ概念が複数の役割に出ても、概念名だけで役割を決めない ----------------------------
def _fake_engine():
    pub = build_corpus("KES", 0, "A", n_docs=12)[0]
    d = Decipherer(pub)
    d.affix_fn = {}
    from collections import Counter
    d.order_counts = {"so": Counter(), "adj": Counter(), "verb": Counter()}
    return d


def test_ambiguous_concept_roles_do_not_vote_wrong_order():
    """「Zの召使いがZを打った」: 所有者のZ（文頭）を目的語の位置と取り違えて OS の票を入れない。"""
    d = _fake_engine()
    it = {"units": [("S:z", 0), ("S:serv", 1), ("S:z", 2), ("S:hit", 3)], "sent": ["z", "serv", "z", "hit"]}
    row = [(0, "S:z", "Z", ["head:possessor", "head:object"]), (1, "S:serv", "servant", ["head:subject"]),
           (2, "S:z", "Z", ["head:possessor", "head:object"]), (3, "S:hit", "strike", ["pred"])]
    d._count_orders([(it, row)])
    assert d.order_counts["so"]["OS"] == 0
    assert d.role_stats["ambiguous_skipped"] == 2


def test_ambiguous_roles_resolved_by_learned_case_affix():
    """格接辞の機能が学習済みなら、曖昧な役割を1つに絞って語順を数える。"""
    d = _fake_engine()
    d.affix_fn = {"S:ga": {"R:subject": 0.9}, "S:ni": {"R:recipient": 0.9}, "S:o": {"R:object": 0.9}}
    it = {"units": [("S:mer", 0), ("A:S:ga", 0), ("S:mer", 1), ("A:S:ni", 1), ("S:fish", 2), ("A:S:o", 2), ("S:give", 3)],
          "sent": ["merga", "merni", "fisho", "give"]}
    row = [(0, "S:mer", "merchant", ["head:subject", "head:recipient"]),
           (1, "S:mer", "merchant", ["head:subject", "head:recipient"]),
           (2, "S:fish", "fish", ["head:object"]), (3, "S:give", "give", ["pred"])]
    d._count_orders([(it, row)])
    assert d.order_counts["so"]["SO"] == 1 and d.role_stats["ambiguous_resolved_by_case"] == 2


def test_same_concept_generator_option_leaves_default_corpora_unchanged():
    a, _ = build_corpus("SOT", 5, "A", n_docs=20)
    b, _ = build_corpus("SOT", 5, "A", n_docs=20, same_concept_rate=0.0)
    assert a == b


# ---- 問題4: 単語境界のない文章で、train だけで学習するインダクティブ方式 -------------------------
def test_inductive_segmentation_does_not_use_heldout_text():
    from babel.decipher import segment
    pub, _ = build_corpus("UMB", 2, "C", boundaries=False, lacuna=0.0, n_docs=30)
    held = set(pub["split"]["heldout"])
    tr = [s for d in pub["documents"] if d["id"] not in held for s in d["sentences"]]
    he = [s for d in pub["documents"] if d["id"] in held for s in d["sentences"]]
    _, seg1 = segment.inductive_segment(tr, he)
    _, seg2 = segment.inductive_segment(tr, list(reversed(he)))
    assert seg1 == list(reversed(seg2))       # heldout の文は互いに影響しない（凍結したモデルを1文ずつ当てる）
    assert segment.unigram_segment(tr) == segment.inductive_segment(tr, [])[0]


def test_engine_reports_segmentation_mode():
    pub, _ = build_corpus("KES", 1, "C", boundaries=False, lacuna=0.0, n_docs=20)
    assert decipher(pub)["segmentation_used"] == "transductive"
    assert decipher(pub, {"segmentation": "inductive"})["segmentation_used"] == "inductive"
    pub2, _ = build_corpus("KES", 1, "C", n_docs=10)
    assert decipher(pub2)["segmentation_used"] is None


# ---- Codex レビュー（PR #4）の指摘2件 ---------------------------------------------------
def test_inductive_segmentation_fits_only_on_train_ids(monkeypatch):
    """train にも heldout にもない文書の文字列を、分割の学習に使わない。"""
    from babel.decipher import segment
    seen = {}
    orig = segment.inductive_segment

    def spy(tr, te, *a, **k):
        seen["train"], seen["test"] = list(tr), list(te)
        return orig(tr, te, *a, **k)
    monkeypatch.setattr(segment, "inductive_segment", spy)
    pub = {"has_word_boundaries": False, "documents": [
        {"id": "train", "sentences": ["kamitaso", "kamira"]}, {"id": "unused", "sentences": ["ZZZZQQQQ"]},
        {"id": "test", "sentences": ["kamiso"]}], "split": {"train": ["train"], "heldout": ["test"]}}
    d = Decipherer(pub, {"segmentation": "inductive"})
    assert seen["train"] == ["kamitaso", "kamira"] and "ZZZZQQQQ" in seen["test"]
    assert set(d.tokens) == {"train", "unused", "test"}


def test_duplicate_alias_does_not_collide_with_existing_id():
    pub = {"documents": [{"id": "x#dup1", "sentences": ["aa bb"]}, {"id": "x", "sentences": ["cc dd"]},
                         {"id": "x", "sentences": ["ee ff"]}]}
    d = Decipherer(pub)
    ids = [x["id"] for x in d.pub["documents"]]
    assert len(ids) == len(set(ids)) == 3
    assert d.tokens["x#dup1"] == [["aa", "bb"]] and d.tokens["x"] == [["cc", "dd"]]
