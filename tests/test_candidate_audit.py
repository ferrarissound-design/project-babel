from collections import Counter
from itertools import product

from babel.decipher.engine import Decipherer
from babel.experiments import candidate_audit
from babel.experiments.candidate_audit import (
    STAGES,
    direct_triples,
    first_losses,
    score_stages,
    stage_candidates,
)


def test_lower_scoring_gold_concept_is_lost_only_when_best_is_chosen():
    item = {"units": [("S:x", 0)], "concepts": [("king", "head:subject"), ("take", "pred")]}
    stages = stage_candidates(item, {("S:x", "C:king"): 0.3, ("S:x", "C:take"): 0.4}, {})
    assert stages["threshold"][0] == {0}
    assert stages["best_concept"][0] == set()
    assert stages["first_position"][1] == {0}
    # The decoder breaks equal score ties by the concept string, independently of insertion order.
    tie = {("S:x", "C:take"): 0.4, ("S:x", "C:king"): 0.4}
    assert stage_candidates(item, tie, {})["best_concept"] == stages["best_concept"]


def test_pos_rejection_zero_link_and_threshold_have_distinct_losses():
    item = {"units": [("S:x", 0)], "concepts": [("take", "pred")]}
    rejected = stage_candidates(item, {}, {}, constraint=lambda u, c: False)
    assert rejected["units"][1] == {0} and rejected["pos_allowed"][1] == set()
    zero = stage_candidates(item, {}, {})
    assert zero["pos_allowed"][1] == {0} and zero["positive_link"][1] == set()
    weak = stage_candidates(item, {("S:x", "C:take"): 0.249}, {})
    assert weak["positive_link"][1] == {0} and weak["threshold"][1] == set()
    boundary = stage_candidates(item, {("S:x", "C:take"): 0.25}, {})
    assert boundary["first_position"][1] == {0}


def test_first_position_can_shadow_a_correct_repeated_word():
    item = {"units": [("S:king", 0), ("S:take", 1), ("S:grain", 2), ("S:king", 3)],
            "concepts": [("king", "head:subject"), ("take", "pred"), ("grain", "head:object")]}
    links = {(unit, "C:" + unit[2:]): 0.8 for unit, _ in item["units"]}
    stages = stage_candidates(item, links, {})
    scores = score_stages(stages, [(3, 1, 2)])
    assert scores["resolved_role"] == {"subject": 1, "predicate": 1, "object": 1, "triple": 1}
    assert scores["first_position"] == {"subject": 0, "predicate": 1, "object": 1, "triple": 0}
    assert first_losses(scores) == {"subject": "first_position", "predicate": "retained",
                                    "object": "retained", "triple": "first_position"}


def test_ambiguous_concepts_can_be_resolved_by_case_but_conflicting_cues_cannot():
    item = {"units": [("S:merchant", 0), ("A:S:subj", 0), ("S:merchant", 2), ("A:S:obj", 2)],
            "concepts": [("merchant", "head:subject"), ("merchant", "head:object")]}
    links = {("S:merchant", "C:merchant"): 0.9}
    affixes = {"S:subj": {"R:subject": 0.9}, "S:obj": {"R:object": 0.9}}
    stages = stage_candidates(item, links, affixes)
    assert stages["best_concept"][0] == stages["best_concept"][2] == {0, 2}
    assert tuple(stages["first_position"]) == ({0}, set(), {2})
    conflicting = dict(item, units=item["units"] + [("A:S:obj", 0)])
    assert stage_candidates(conflicting, links, affixes)["resolved_role"][0] == set()


def test_compound_parts_are_separate_candidates_at_the_same_word_position():
    item = {"units": [("S:a", 0), ("S:b", 0)],
            "concepts": [("king", "head:subject"), ("take", "pred")]}
    stages = stage_candidates(item, {("S:a", "C:king"): 0.9, ("S:b", "C:take"): 0.8}, {})
    assert tuple(stages["first_position"]) == ({0}, {0}, set())


def test_separate_role_hits_do_not_claim_a_complete_gold_triple():
    stages = {stage: ({0}, {4}, {2}) for stage in STAGES}
    scores = score_stages(stages, [(0, 1, 2), (3, 4, 5)])
    assert scores["units"] == {"subject": 1, "predicate": 1, "object": 1, "triple": 0}
    assert first_losses(scores)["triple"] == "units"


def test_gold_requires_direct_active_subject_and_object_of_a_verb():
    def token(i, head, relation, pos="NOUN"):
        return {"id": i, "head": head, "deprel": relation, "upos": pos}

    tokens = [token(1, 2, "nsubj"), token(2, 0, "root", "VERB"), token(3, 2, "obj"),
              token(4, 2, "nsubj:pass"), token(5, 3, "obj"), token(6, 2, "nsubj")]
    assert set(direct_triples(tokens)) == {(0, 1, 2), (5, 1, 2)}
    tokens[1]["upos"] = "AUX"
    assert direct_triples(tokens) == []


def test_final_stage_reproduces_real_decoder_votes_without_annotation():
    combinations = list(product(range(2), repeat=3))
    sentences = [f"ka{s} ti{v} mo{o}" for s, v, o in combinations] * 10
    parallel = [{"doc": "d", "sent": i, "translation": {"pred": f"verb{v}", "args": [
        {"role": "subject", "head": f"subj{s}"}, {"role": "object", "head": f"obj{o}"}]}}
        for i, (s, v, o) in enumerate(combinations)]
    public = {"documents": [{"id": "d", "sentences": sentences}], "parallel": parallel, "level": "A"}
    decoder = Decipherer(public, {"v16_gates": False, "null_gates": False, "pos_constraint": False,
                                 "use_morph": False, "use_variants": False}).fit()
    observed = Counter()
    for item, links in zip(decoder.aligner.items, decoder.aligner.links, strict=True):
        stages = stage_candidates(item, links, decoder.affix_fn)
        for left, right in zip(STAGES, STAGES[1:]):
            assert all(b <= a for a, b in zip(stages[left], stages[right], strict=True))
        subject, predicate, obj = stages["first_position"]
        if subject and predicate and obj:
            s, v, o = next(iter(subject)), next(iter(predicate)), next(iter(obj))
            observed["initial" if v < min(s, o) else "final" if v > max(s, o) else "medial"] += 1
    assert observed and observed == decoder.order_counts["verb_rel"]


def test_run_keeps_cleaning_and_source_eligibility_outside_candidate_denominator(tmp_path, monkeypatch):
    def sentence(sid, subject, verb, obj):
        rows = [f"# sent_id = {sid}",
                f"1\t{subject}\tking\tNOUN\t_\t_\t2\tnsubj\t_\t_",
                f"2\t{verb}\ttake\tVERB\t_\t_\t0\troot\t_\t_"]
        if obj:
            rows.append(f"3\t{obj}\tgrain\tNOUN\t_\t_\t2\tobj\t_\t_")
        return "\n".join(rows) + "\n\n"

    source = tmp_path / "source.conllu"
    source.write_text(sentence("a", "king", "takes", "grain")
                      + sentence("b", "king", "takes", "grain")
                      + sentence("c", "king", "takes", None), encoding="utf-8")
    target = tmp_path / "target.conllu"
    target.write_text(sentence("a", "kura", "tali", "moru")
                      + sentence("b", "123", "tali", "moru")
                      + sentence("c", "kura", "tali", "moru"), encoding="utf-8")
    monkeypatch.setattr(candidate_audit.rp, "fetch", lambda repo, code: (
        (source, "source-sha") if code == "en_pud" else (target, "target-sha")))
    result = candidate_audit.run((("UD_Test-PUD", "test_pud"), False))
    assert result["totals"] == {"parallel": 3, "source_full": 2, "source_full_target_raw": 2, "eligible": 1}
    assert result["source_sha256"] == "source-sha" and result["target_sha256"] == "target-sha"
    assert [row["eligible"] for row in result["sentences"]] == [True, False, False]
    assert result["sentences"][1]["raw_triples"] == 1 and result["sentences"][1]["kept_triples"] == 0
    assert result == candidate_audit.run((("UD_Test-PUD", "test_pud"), False))
