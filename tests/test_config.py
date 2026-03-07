"""Tests for pyzm.models.config — YAML boolean handling.

Ensures that Python bools (True/False) and string values ('yes'/'no')
are both handled correctly by config parsing.
"""

import pytest

from pyzm.models.config import DetectorConfig, _bool


# ---------------------------------------------------------------------------
# _bool helper
# ---------------------------------------------------------------------------

class TestBoolHelper:
    def test_python_true(self):
        assert _bool(True) is True

    def test_python_false(self):
        assert _bool(False) is False

    def test_string_yes(self):
        assert _bool("yes") is True

    def test_string_no(self):
        assert _bool("no") is False

    def test_string_true(self):
        assert _bool("true") is True

    def test_string_false(self):
        assert _bool("false") is False

    def test_none_default_false(self):
        assert _bool(None) is False

    def test_none_default_true(self):
        assert _bool(None, default=True) is True


# ---------------------------------------------------------------------------
# Helpers to build minimal ml_options dicts
# ---------------------------------------------------------------------------

def _minimal_ml_options(
    *,
    global_general_extras: dict | None = None,
    section_general_extras: dict | None = None,
    seq_extras: dict | None = None,
):
    """Return a minimal ml_options dict with one object model."""
    general = {"model_sequence": "object"}
    if global_general_extras:
        general.update(global_general_extras)

    section_general = {}
    if section_general_extras:
        section_general.update(section_general_extras)

    seq = {
        "name": "test_model",
        "object_weights": "/tmp/w.weights",
    }
    if seq_extras:
        seq.update(seq_extras)

    return {
        "general": general,
        "object": {
            "general": section_general,
            "sequence": [seq],
        },
    }


# ---------------------------------------------------------------------------
# enabled
# ---------------------------------------------------------------------------

class TestEnabled:
    def test_enabled_false_bool_disables(self):
        cfg = DetectorConfig.from_dict(_minimal_ml_options(seq_extras={"enabled": False}))
        assert cfg.models[0].enabled is False

    def test_enabled_no_string_disables(self):
        cfg = DetectorConfig.from_dict(_minimal_ml_options(seq_extras={"enabled": "no"}))
        assert cfg.models[0].enabled is False

    def test_enabled_true_bool_enables(self):
        cfg = DetectorConfig.from_dict(_minimal_ml_options(seq_extras={"enabled": True}))
        assert cfg.models[0].enabled is True

    def test_enabled_yes_string_enables(self):
        cfg = DetectorConfig.from_dict(_minimal_ml_options(seq_extras={"enabled": "yes"}))
        assert cfg.models[0].enabled is True

    def test_enabled_default_is_true(self):
        cfg = DetectorConfig.from_dict(_minimal_ml_options())
        assert cfg.models[0].enabled is True


# ---------------------------------------------------------------------------
# match_past_detections (global)
# ---------------------------------------------------------------------------

class TestMatchPastDetections:
    def test_bool_true_enables(self):
        cfg = DetectorConfig.from_dict(
            _minimal_ml_options(global_general_extras={"match_past_detections": True})
        )
        assert cfg.match_past_detections is True

    def test_string_yes_enables(self):
        cfg = DetectorConfig.from_dict(
            _minimal_ml_options(global_general_extras={"match_past_detections": "yes"})
        )
        assert cfg.match_past_detections is True

    def test_bool_false_disables(self):
        cfg = DetectorConfig.from_dict(
            _minimal_ml_options(global_general_extras={"match_past_detections": False})
        )
        assert cfg.match_past_detections is False

    def test_string_no_disables(self):
        cfg = DetectorConfig.from_dict(
            _minimal_ml_options(global_general_extras={"match_past_detections": "no"})
        )
        assert cfg.match_past_detections is False


# ---------------------------------------------------------------------------
# match_past_detections (type override)
# ---------------------------------------------------------------------------

class TestTypeOverrideMatchPastDetections:
    def test_bool_true_enables(self):
        cfg = DetectorConfig.from_dict(
            _minimal_ml_options(section_general_extras={"match_past_detections": True})
        )
        from pyzm.models.config import ModelType
        assert cfg.type_overrides[ModelType.OBJECT].match_past_detections is True

    def test_string_yes_enables(self):
        cfg = DetectorConfig.from_dict(
            _minimal_ml_options(section_general_extras={"match_past_detections": "yes"})
        )
        from pyzm.models.config import ModelType
        assert cfg.type_overrides[ModelType.OBJECT].match_past_detections is True

    def test_bool_false_disables(self):
        cfg = DetectorConfig.from_dict(
            _minimal_ml_options(section_general_extras={"match_past_detections": False})
        )
        from pyzm.models.config import ModelType
        assert cfg.type_overrides[ModelType.OBJECT].match_past_detections is False


# ---------------------------------------------------------------------------
# save_unknown_faces
# ---------------------------------------------------------------------------

class TestSaveUnknownFaces:
    def test_bool_true_enables(self):
        cfg = DetectorConfig.from_dict(
            _minimal_ml_options(seq_extras={"save_unknown_faces": True})
        )
        assert cfg.models[0].save_unknown_faces is True

    def test_string_yes_enables(self):
        cfg = DetectorConfig.from_dict(
            _minimal_ml_options(seq_extras={"save_unknown_faces": "yes"})
        )
        assert cfg.models[0].save_unknown_faces is True

    def test_bool_false_disables(self):
        cfg = DetectorConfig.from_dict(
            _minimal_ml_options(seq_extras={"save_unknown_faces": False})
        )
        assert cfg.models[0].save_unknown_faces is False


# ---------------------------------------------------------------------------
# disable_locks
# ---------------------------------------------------------------------------

class TestDisableLocks:
    def test_bool_true_enables(self):
        cfg = DetectorConfig.from_dict(
            _minimal_ml_options(seq_extras={"disable_locks": True})
        )
        assert cfg.models[0].disable_locks is True

    def test_string_yes_enables(self):
        cfg = DetectorConfig.from_dict(
            _minimal_ml_options(seq_extras={"disable_locks": "yes"})
        )
        assert cfg.models[0].disable_locks is True

    def test_bool_false_disables(self):
        cfg = DetectorConfig.from_dict(
            _minimal_ml_options(seq_extras={"disable_locks": False})
        )
        assert cfg.models[0].disable_locks is False


# ---------------------------------------------------------------------------
# ModelConfig.options population
# ---------------------------------------------------------------------------

class TestModelConfigOptions:
    """H1: options dict must be populated from YAML sequence items."""

    def test_alpr_options_populated(self):
        ml = {
            "general": {"model_sequence": "alpr"},
            "alpr": {
                "general": {},
                "sequence": [{
                    "alpr_service": "plate_recognizer",
                    "alpr_key": "test",
                    "alpr_url": "http://test",
                    "platerec_stats": "yes",
                    "platerec_regions": ["us"],
                    "alpr_api_type": "local",
                    "openalpr_cmdline_binary": "/usr/bin/alpr",
                }],
            },
        }
        cfg = DetectorConfig.from_dict(ml)
        opts = cfg.models[0].options
        assert opts["platerec_stats"] == "yes"
        assert opts["platerec_regions"] == ["us"]
        assert opts["alpr_api_type"] == "local"
        assert opts["openalpr_cmdline_binary"] == "/usr/bin/alpr"

    def test_face_unknown_face_name_in_options(self):
        ml = {
            "general": {"model_sequence": "face"},
            "face": {
                "general": {},
                "sequence": [{
                    "face_detection_framework": "dlib",
                    "unknown_face_name": "stranger",
                }],
            },
        }
        cfg = DetectorConfig.from_dict(ml)
        assert cfg.models[0].options["unknown_face_name"] == "stranger"

    def test_known_keys_not_in_options(self):
        ml = {
            "general": {"model_sequence": "object"},
            "object": {
                "general": {},
                "sequence": [{"object_min_confidence": 0.5, "name": "yolo"}],
            },
        }
        cfg = DetectorConfig.from_dict(ml)
        assert "object_min_confidence" not in cfg.models[0].options
        assert "name" not in cfg.models[0].options
