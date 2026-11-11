from babel.generator.corpus import build_corpus, null_corpus
from babel.generator.languages import FAMILIES, STRUCTURE, Language


def _sentences(pub):
    return [s for d in pub["documents"] for s in d["sentences"]]


def test_scale_meets_initial_target():
    total = 0
    for fam in FAMILIES:
        pub, _ = build_corpus(fam, 0, "A")
        assert len(pub["documents"]) >= 100
        total += len(_sentences(pub))
    assert total >= 3000


def test_generation_is_deterministic():
    a, _ = build_corpus("UMB", 3, "A", n_docs=20)
    b, _ = build_corpus("UMB", 3, "A", n_docs=20)
    assert a == b


def test_seeds_give_different_vocabulary():
    assert Language("KES", 0).roots["king"] != Language("KES", 1).roots["king"]


def test_three_families_differ_in_typology():
    assert len({STRUCTURE[f]["verb_position"] for f in FAMILIES}) == 3
    assert len({STRUCTURE[f]["numeral_base"] for f in FAMILIES}) == 3
    assert {STRUCTURE[f]["affix_position"] for f in FAMILIES} == {"suffix", "prefix"}


def test_required_linguistic_features_are_present():
    for fam in FAMILIES:
        _, sec = build_corpus(fam, 0, "A")
        toks = [t for sents in sec["tokens"].values() for s in sents for t in s]
        feats = {f for t in toks for f in t["feats"]}
        assert {"N:PL", "T:PAST", "T:FUT", "R:object", "R:possessor", "D:AGENT"} <= feats      # 格・数・時制・派生
        assert any("+" in (t["concept"] or "") for t in toks)                                  # 複合語
        assert any(t["pos"] == "NUM" for t in toks)                                            # 数詞
        assert any(t["s"] != t["canon"] for t in toks if not t.get("lacuna"))                  # 表記揺れ・音変化
        assert any(t.get("lacuna") for t in toks)                                              # 欠損
        assert sec["grammar"]["homonyms"] and sec["grammar"]["late_replacements"]              # 同音異義・語彙交替
        sems = {t["sem"] for t in toks}
        assert {"name_person", "name_place"} <= sems                                           # 人名・地名


def test_numeral_systems_compose_correctly():
    for fam, base in (("KES", 10), ("UMB", 20), ("SOT", 12)):
        lang = Language(fam, 5)
        for n in (1, 7, base, base + 3, 3 * base + 1, 157):
            assert sum(int(w["concept"][1:]) for w in lang.number(n)) == n


def test_account_totals_are_arithmetically_consistent():
    pub, sec = build_corpus("KES", 2, "B", lacuna=0.0)
    checked = 0
    for d in pub["documents"]:
        if d["meta"]["genre"] != "account":
            continue
        sents = sec["tokens"][d["id"]]
        vals = [sum(int(t["concept"][1:]) for t in s if t["pos"] == "NUM") for s in sents]
        assert sum(vals[:-1]) == vals[-1]
        checked += 1
    assert checked > 10


def test_null_corpus_has_same_shape():
    pub, sec = null_corpus(0, "random", n_docs=20)
    assert sec["null"] == "random" and len(pub["documents"]) == 20
