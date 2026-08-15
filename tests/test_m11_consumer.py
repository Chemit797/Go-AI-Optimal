import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "predict_m11_score_sprint.py"
SPEC = importlib.util.spec_from_file_location("predict_m11_score_sprint", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_high_specialist_changes_only_prediction_gated_positions():
    m6 = np.asarray([[0.5, 1.0, -2.0]], dtype=np.float32)
    m9 = np.asarray([[1.5, 3.0, 2.0]], dtype=np.float32)
    m9_base = np.zeros_like(m6)
    blended, _ = MODULE._fuse_response(
        "blend", 1.0, m6, m9, m9_base, 0.75, 0.25
    )
    specialist, formula = MODULE._fuse_response(
        "high_specialist", 1.0, m6, m9, m9_base, 0.75, 0.25
    )
    assert np.allclose(blended, m9)
    assert np.isclose(specialist[0, 0], blended[0, 0])
    assert np.isclose(specialist[0, 1], 0.75 * m9[0, 1] + 0.25 * m6[0, 1])
    assert np.isclose(specialist[0, 2], 0.75 * m9[0, 2] + 0.25 * m6[0, 2])
    assert "I(abs(R6)>=threshold)" in formula


def test_semantic_shrink_is_a_bounded_residual_blend():
    current = np.asarray([[1.0, 3.0]], dtype=np.float32)
    semantic = np.asarray([[5.0, -1.0]], dtype=np.float32)
    assert np.allclose(MODULE._semantic_shrink(current, semantic, 0.0), current)
    assert np.allclose(
        MODULE._semantic_shrink(current, semantic, 0.1),
        np.asarray([[1.4, 2.6]], dtype=np.float32),
    )
    assert np.allclose(MODULE._semantic_shrink(current, semantic, 1.0), semantic)
