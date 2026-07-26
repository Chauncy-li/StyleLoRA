"""Small image container used by NuPlan sensor observation types.

This module is part of the NuPlan devkit API and is required even for
track-based simulations because ``observation_type`` imports ``Sensors`` and
its image annotation at module import time.
"""

from functools import cached_property

import cv2
import numpy as np
import PIL.Image as PilImg


class Image:
    """Wrap a PIL image and expose cached NumPy/OpenCV representations."""

    def __init__(self, pil_img: PilImg.Image):
        self._pil_img = pil_img

    @property
    def as_pil(self) -> PilImg.Image:
        """Return the wrapped PIL image."""

        return self._pil_img

    def as_numpy_nocache(self) -> np.ndarray:
        """Return an RGB uint8 array without caching it."""

        return np.array(self._pil_img, dtype=np.uint8)

    @cached_property
    def as_numpy(self) -> np.ndarray:
        """Return a cached RGB uint8 array."""

        return self.as_numpy_nocache()

    def as_cv2_nocache(self) -> np.ndarray:
        """Return a BGR uint8 array without caching it."""

        return cv2.cvtColor(np.array(self._pil_img, np.uint8), cv2.COLOR_RGB2BGR)

    @cached_property
    def as_cv2(self) -> np.ndarray:
        """Return a cached BGR uint8 array."""

        return self.as_cv2_nocache()
