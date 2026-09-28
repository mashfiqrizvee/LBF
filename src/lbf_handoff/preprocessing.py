"""Common deterministic face preprocessing used before all three frontends.

Raw photographs and the dlib landmark model are intentionally not bundled.
This module is the readable recipe for regenerating aligned inputs when those
licensed/source datasets are available to the research group.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import dlib
import numpy as np


@dataclass(frozen=True)
class PreprocessingConfig:
    detector_upsample: int = 1
    aligned_size: int = 160
    # These are pixel coordinates in the accepted 160-by-160 frame.
    left_eye_target: tuple[float, float] = (54.4, 64.0)
    right_eye_target: tuple[float, float] = (105.6, 64.0)
    ellipse_center: tuple[float, float] = (0.50, 0.52)
    ellipse_axes: tuple[float, float] = (0.43, 0.48)
    paper_size: int = 70


@dataclass(frozen=True)
class PreprocessingResult:
    success: bool
    reason: str
    face_count: int
    aligned_gray: np.ndarray | None
    aligned_rgb: np.ndarray | None
    aligned_landmarks: np.ndarray | None
    paper_pixels: np.ndarray | None


def affine_matrix(landmarks: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Fit the accepted eye-centre similarity transform.

    Only rotation, uniform scale, and translation are allowed.  This is the
    exact geometry used to create the accepted tensors; an unrestricted affine
    transform would change local facial shape.
    """
    if landmarks.shape != (68, 2):
        raise ValueError(f"Expected 68-by-2 landmarks, observed {landmarks.shape}")
    left_eye = landmarks[36:42].mean(axis=0)
    right_eye = landmarks[42:48].mean(axis=0)
    source_vector = right_eye - left_eye
    target_left = np.asarray(config.left_eye_target, dtype=np.float64)
    target_right = np.asarray(config.right_eye_target, dtype=np.float64)
    target_vector = target_right - target_left
    source_length = float(np.linalg.norm(source_vector))
    if source_length <= 1e-8:
        raise ValueError("Degenerate eye geometry")
    scale = float(np.linalg.norm(target_vector)) / source_length
    angle = np.arctan2(target_vector[1], target_vector[0]) - np.arctan2(source_vector[1], source_vector[0])
    cosine, sine = np.cos(angle), np.sin(angle)
    linear = scale * np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    translation = target_left - linear @ left_eye
    matrix = np.column_stack((linear, translation)).astype(np.float32)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:, :2])) < 1e-8:
        raise ValueError("Degenerate face alignment transform")
    return matrix


def transform_landmarks(landmarks: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([landmarks.astype(np.float64), np.ones(68)])
    return homogeneous @ matrix.astype(np.float64).T


def ellipse_mask(size: int, config: PreprocessingConfig) -> np.ndarray:
    """The P1 mask suppresses background while keeping one fixed geometry."""
    mask = np.zeros((size, size), dtype=np.uint8)
    center = tuple(round(value * (size - 1)) for value in config.ellipse_center)
    axes = tuple(round(value * (size - 1)) for value in config.ellipse_axes)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, thickness=-1)
    return mask


def paper_pixels(aligned_gray: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Create the 4,900 P1 intensities before fit-set mean thresholding."""
    mask = ellipse_mask(config.aligned_size, config)
    masked = cv2.bitwise_and(aligned_gray, aligned_gray, mask=mask)
    equalized = cv2.equalizeHist(masked)
    resized = cv2.resize(equalized, (config.paper_size, config.paper_size), interpolation=cv2.INTER_AREA)
    return resized.reshape(-1).astype(np.uint8, copy=False)


class FacePreprocessor:
    """Detect exactly one face; failures are recorded instead of guessed around."""

    def __init__(self, predictor_path: Path, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(predictor_path))

    def process(self, path: Path) -> PreprocessingResult:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            return PreprocessingResult(False, "image_read_failed", 0, None, None, None, None)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, self.config.detector_upsample)
        if len(faces) != 1:
            return PreprocessingResult(False, f"face_count_{len(faces)}", len(faces), None, None, None, None)
        try:
            shape = self.predictor(gray, faces[0])
            landmarks = np.asarray([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float32)
            matrix = affine_matrix(landmarks, self.config)
            size = (self.config.aligned_size, self.config.aligned_size)
            aligned_gray = cv2.warpAffine(gray, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            aligned_bgr = cv2.warpAffine(bgr, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            aligned_landmarks = transform_landmarks(landmarks, matrix)
            return PreprocessingResult(
                True, "", 1, aligned_gray, cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB),
                aligned_landmarks, paper_pixels(aligned_gray, self.config),
            )
        except (RuntimeError, ValueError, cv2.error) as error:
            return PreprocessingResult(False, type(error).__name__, 1, None, None, None, None)
