import json

from babel import external
from babel.generator.corpus import build_corpus


def _setup(tmp_path):
    public, secret = build_corpus("KES", 4, "C", n_docs=30)
    inbox = tmp_path / "inbox" / "mystery"
    inbox.mkdir(parents=True)
    (inbox / "corpus.json").write_text(json.dumps(public, ensure_ascii=False), encoding="utf-8")
    return secret


def test_prediction_is_recorded_once_and_never_overwritten(tmp_path):
    _setup(tmp_path)
    made = external.predict(tmp_path, commit="abc")
    assert len(made) == 1 and made[0]["attempt"] == 1 and made[0]["first_attempt"]
    assert external.predict(tmp_path, commit="abc") == []        # 同じ解読器・同じコーパスなら再生成しない
    assert external.verify(tmp_path) == []


def test_tampering_with_a_prediction_is_detected(tmp_path):
    _setup(tmp_path)
    external.predict(tmp_path)
    f = tmp_path / "predictions" / "mystery" / "attempt-001.json"
    f.write_text(f.read_text(encoding="utf-8").replace("undetermined", "claimed", 1), encoding="utf-8")
    assert any("書き換え" in p for p in external.verify(tmp_path))


def test_unlisted_prediction_file_is_detected(tmp_path):
    _setup(tmp_path)
    external.predict(tmp_path)
    (tmp_path / "predictions" / "mystery" / "attempt-002.json").write_text("{}", encoding="utf-8")
    assert any("記録のない" in p for p in external.verify(tmp_path))


def test_scoring_happens_only_after_answer_is_added(tmp_path):
    secret = _setup(tmp_path)
    external.predict(tmp_path)
    assert external.score_all(tmp_path) == []
    ans = tmp_path / "answers" / "mystery"
    ans.mkdir(parents=True)
    (ans / "answer.json").write_text(json.dumps(secret, ensure_ascii=False), encoding="utf-8")
    scores = external.score_all(tmp_path)
    assert len(scores) == 1 and scores[0]["blind"] is True and scores[0]["independent"] is False


def test_text_format_and_answer_leak_guard(tmp_path):
    folder = tmp_path / "inbox" / "txt"
    folder.mkdir(parents=True)
    (folder / "corpus.txt").write_text("# comment\nka ta mi\nso ra\n\nmi ta\n", encoding="utf-8")
    pub = external.load_corpus(folder)
    assert [len(d["sentences"]) for d in pub["documents"]] == [2, 1] and pub["level"] == "C"
    (folder / "corpus.txt").unlink()
    (folder / "corpus.json").write_text(json.dumps({"documents": [], "lexicon": []}), encoding="utf-8")
    try:
        external.load_corpus(folder)
        raise AssertionError("正解入りのコーパスを受理してしまった")
    except ValueError:
        pass
