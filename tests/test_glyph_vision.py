"""Vision pipeline tests use entirely synthetic images, no external service."""
import json

import pytest

from babel.vision.glyphs import analyze_image, repeated_patterns, save_analysis

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def _draw_test_image(tmp_path, light=False):
    background, ink = (255, 0) if light else (0, 255)
    frame = np.full((90, 160, 3), background, dtype=np.uint8)
    for y in (22, 60):
        for x, glyph in ((8, "X"), (40, "O"), (72, "X"), (104, "O")):
            cv2.putText(frame, glyph, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (ink, ink, ink), 2, cv2.LINE_8)
    target = tmp_path / ("light.png" if light else "dark.png")
    assert cv2.imwrite(str(target), frame)
    return target


@pytest.mark.parametrize("light", [False, True])
def test_repeated_shapes_and_patterns(tmp_path, light):
    img = _draw_test_image(tmp_path, light)
    data = analyze_image(img)
    assert data["stats"]["rows"] == 2
    assert data["stats"]["glyph_candidates"] == 8
    assert data["stats"]["visual_classes"] == 2
    first, second = [row["classes"] for row in data["rows"]]
    assert first == second
    assert first[0] == first[2] and first[1] == first[3] and first[0] != first[1]
    assert any(p["occurrences"] >= 2 and len(p["symbols"]) == 2 for p in data["repeated_patterns"])
    assert "meaning" not in data["glyphs"][0]


def test_save_json_corpus_hypotheses_and_annotation(tmp_path):
    img = _draw_test_image(tmp_path)
    out = tmp_path / "result"
    data = save_analysis(img, out, regions=[{"id": "panel", "box": [0, 0, 160, 90]}])
    assert (out / "glyph-annotated.png").exists()
    assert json.loads((out / "glyph-report.json").read_text())["stats"] == data["stats"]
    continuous = json.loads((out / "corpus-continuous.json").read_text())
    spaced = json.loads((out / "corpus-glyph-tokens.json").read_text())
    assert not continuous["has_word_boundaries"] and spaced["has_word_boundaries"]
    assert continuous["known"] == spaced["known"] == []
    assert continuous["parallel"] == spaced["parallel"] == []
    assert [s.replace(" ", "") for s in spaced["documents"][0]["sentences"]] == continuous["documents"][0]["sentences"]
    assert all(s for s in continuous["documents"][0]["sentences"])


def test_region_validation_and_repeated_pattern_boundaries(tmp_path):
    img = _draw_test_image(tmp_path)
    with pytest.raises(ValueError, match="bounds"):
        analyze_image(img, regions=[{"id": "wrong", "box": [-1, 0, 10, 10]}])
    with pytest.raises(ValueError, match="unique"):
        analyze_image(img, regions=[{"id": "dup", "box": [0, 0, 10, 10]},
                                    {"id": "dup", "box": [10, 0, 20, 10]}])
    with pytest.raises(ValueError, match="threshold"):
        analyze_image(img, threshold=999)
    with pytest.raises(ValueError, match="similarity"):
        analyze_image(img, similarity=0)
    assert repeated_patterns([{"classes": [1, 2]}, {"classes": [3, 4]}]) == []
