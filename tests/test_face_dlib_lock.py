import sys
import unittest
from unittest.mock import MagicMock

import numpy as np
from pyzm.models.config import ModelConfig, ModelFramework, ModelType


class TestFaceDlibLockRelease(unittest.TestCase):
    """H4: Lock must be released even if face_recognition raises."""

    def test_lock_released_on_exception(self):
        # Inject mock modules for the lazy imports inside detect()
        mock_cv2 = MagicMock()
        mock_fr = MagicMock()
        mock_imutils = MagicMock()

        sys.modules.setdefault("cv2", mock_cv2)
        sys.modules.setdefault("imutils", mock_imutils)
        sys.modules.setdefault("face_recognition", mock_fr)

        from pyzm.ml.backends.face_dlib import FaceDlibBackend

        config = ModelConfig(
            type=ModelType.FACE,
            framework=ModelFramework.FACE_DLIB,
            known_faces_dir="/tmp/known",
        )
        backend = FaceDlibBackend(config)
        backend._knn = MagicMock()  # pretend loaded
        backend._auto_lock = True
        backend.acquire_lock = MagicMock()
        backend.release_lock = MagicMock()

        mock_fr.face_locations.side_effect = RuntimeError("GPU OOM")

        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            backend.detect(dummy)

        backend.acquire_lock.assert_called_once()
        backend.release_lock.assert_called_once()
