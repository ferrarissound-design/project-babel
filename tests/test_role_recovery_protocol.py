import json

import pytest

from babel.experiments import role_recovery as study


def test_article_split_uses_metadata_and_keeps_complete_articles(tmp_path):
    path = tmp_path / "input.conllu"
    path.write_text("".join(f"# newdoc id = {genre}{i:05d}\n# sent_id = {genre}{i:05d}001\n"
                            f"# sent_id = {genre}{i:05d}002\n1\tx\tx\tNOUN\t_\t_\t0\troot\t_\t_\n"
                            for genre in "nw" for i in range(10)))
    manifest = study.make_manifest(path, "a" * 64)
    assert len(manifest["article_phase"]) == 20
    for genre in "nw":
        assert sum(phase == "dev" for doc, phase in manifest["article_phase"].items() if doc.startswith(genre)) == 7
    for sid, article in manifest["sentence_article"].items():
        assert sid[:-3] == article
    path.write_text(path.read_text().replace("NOUN", "VERB").replace("root", "obj"))
    assert study.make_manifest(path, "a" * 64) == manifest


def test_metadata_without_document_or_with_duplicate_sentence_is_rejected(tmp_path):
    path = tmp_path / "bad.conllu"
    path.write_text("# sent_id = n00001001\n")
    with pytest.raises(ValueError):
        study.source_articles(path)
    path.write_text("# newdoc id = n00001\n# sent_id = n00001001\n# sent_id = n00001001\n")
    with pytest.raises(ValueError):
        study.source_articles(path)


def test_committed_article_split_matches_source_metadata_and_is_disjoint():
    manifest = json.loads(study.MANIFEST.read_text())
    articles = manifest["article_phase"]
    dev = {sid for sid, article in manifest["sentence_article"].items() if articles[article] == "dev"}
    final = set(manifest["sentence_article"]) - dev
    assert dev and final and not dev & final
    assert {manifest["sentence_article"][sid] for sid in dev}.isdisjoint(
        {manifest["sentence_article"][sid] for sid in final})
    assert study.digest({"article_phase": articles, "sentence_article": manifest["sentence_article"]}) == manifest["membership_sha256"]
