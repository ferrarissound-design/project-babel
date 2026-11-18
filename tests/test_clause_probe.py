from collections import Counter

from babel.experiments.clause_probe import clause_heads, grade, joint, role_candidates


def test_joint_selects_distinct_positions_in_one_segment():
    choices = [{0: 0.6}, {1: 0.5, 3: 0.9}, {2: 0.6}]
    assert joint(choices, [0, 0, 0, 1]) == (0, 1, 2)
    assert joint(choices, [0, 1, 1, 1]) is None
    assert joint([{0: 1}, {0: 1}, {1: 1}], [0, 0]) is None


def test_ambiguous_concept_is_excluded_from_candidate_pool():
    item = {"concepts": [("a", "pred"), ("a", "head:subject"), ("b", "head:object")],
            "units": [("S:x", 0), ("S:y", 1)]}
    links = {("S:x", "C:a"): 0.9, ("S:y", "C:b"): 0.25}
    assert role_candidates(item, links) == [{}, {}, {1: 0.25}]


def test_clause_membership_and_direct_roles_are_scored_separately():
    def tok(i, head, relation, pos):
        return {"id": i, "head": head, "deprel": relation, "upos": pos}

    tokens = [tok(1, 2, "nsubj", "NOUN"), tok(2, 0, "root", "VERB"), tok(3, 2, "obj", "NOUN"),
              tok(4, 3, "acl:relcl", "VERB"), tok(5, 4, "obj", "NOUN")]
    anchors = clause_heads({"tokens": tokens})
    counts = Counter()
    for triple in [(0, 1, 2), (2, 1, 0), (0, 3, 4), None]:
        grade(triple, tokens, anchors, counts)
    assert counts == {"votes": 3, "same_clause": 2, "verb": 3, "exact": 1}
