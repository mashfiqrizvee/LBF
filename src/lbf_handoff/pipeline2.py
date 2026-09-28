"""Pipeline 2: local DCT bits around facial landmarks."""

from __future__ import annotations

import cv2
import numpy as np

from .bloom import Block


def zigzag_indices(size: int = 8) -> list[tuple[int, int]]:
    """Accepted low-frequency diagonal order, excluding the DC coefficient.

    The historical code called this a zigzag although it does not reverse every
    second diagonal.  We preserve its exact row order here because changing the
    order changes which coefficients enter a 12-bit prefix.
    """
    return [
        (row, diagonal - row)
        for diagonal in range(1, 2 * size - 1)
        for row in range(size)
        if 0 <= diagonal - row < size
    ]


def dct_matrix(size: int = 8) -> np.ndarray:
    """Orthonormal DCT-II matrix used in the accepted implementation."""
    matrix = np.empty((size, size), dtype=np.float32)
    for frequency in range(size):
        for position in range(size):
            scale = np.sqrt(1.0 / size) if frequency == 0 else np.sqrt(2.0 / size)
            matrix[frequency, position] = scale * np.cos(
                np.pi * (position + 0.5) * frequency / size
            )
    return matrix


def extract_features(
    aligned_rgb: np.ndarray,
    landmarks: np.ndarray,
    patch_width: int = 32,
    grid: int = 8,
    coefficients: int = 48,
) -> np.ndarray:
    """Exact P2 sampling from aligned images, returning N-by-68-by-48.

    `patch_width` describes the facial area covered, while `grid` is the number
    of bilinear samples across it.  Sampling directly at subpixel coordinates
    avoids a hidden crop-and-resize difference at landmark boundaries.
    """
    images = np.asarray(aligned_rgb, dtype=np.uint8)
    points = np.asarray(landmarks, dtype=np.float32)
    if images.ndim != 4 or images.shape[-1] != 3 or points.shape != (len(images), 68, 2):
        raise ValueError("Expected N RGB images and matching N-by-68-by-2 landmarks")
    transform = dct_matrix(grid)
    coordinates = zigzag_indices(grid)[:coefficients]
    offsets = (np.arange(grid, dtype=np.float32) - (grid - 1) / 2) * (patch_width / grid)
    features = np.empty((len(images), 68, coefficients), dtype=np.float32)
    for index in range(len(images)):
        gray = cv2.cvtColor(images[index], cv2.COLOR_RGB2GRAY).astype(np.float32)
        map_x = (
            points[index, :, 0, None, None] + offsets[None, None, :]
            + np.zeros((68, grid, 1), dtype=np.float32)
        ).reshape(68 * grid, grid)
        map_y = (
            points[index, :, 1, None, None] + offsets[None, :, None]
            + np.zeros((68, 1, grid), dtype=np.float32)
        ).reshape(68 * grid, grid)
        patches = cv2.remap(
            gray, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        ).reshape(68, grid, grid)
        patches = (patches - patches.mean(axis=(1, 2), keepdims=True)) / np.maximum(
            patches.std(axis=(1, 2), keepdims=True), 1e-6
        )
        transformed = np.einsum("ij,pjk,lk->pil", transform, patches, transform, optimize=True)
        features[index] = transformed[
            :, [item[0] for item in coordinates], [item[1] for item in coordinates]
        ]
    return features


def normalized_dct_features(patches: np.ndarray, dct_size: int = 8) -> np.ndarray:
    """Convert N-by-L-by-H-by-W landmark patches to 63 AC coefficients.

    Normalization removes patch brightness and contrast before the DCT.  A
    nearly constant patch is divided by one rather than by zero.
    """
    patches = np.asarray(patches, dtype=np.float32)
    if patches.ndim != 4:
        raise ValueError("Expected image-by-landmark-by-height-by-width patches")
    positions = zigzag_indices(dct_size)
    output = np.empty((patches.shape[0], patches.shape[1], len(positions)), dtype=np.float32)
    for image in range(patches.shape[0]):
        for landmark in range(patches.shape[1]):
            patch = cv2.resize(patches[image, landmark], (dct_size, dct_size), interpolation=cv2.INTER_AREA)
            patch -= float(patch.mean())
            scale = float(patch.std())
            patch /= scale if scale > 1e-8 else 1.0
            transformed = cv2.dct(patch)
            output[image, landmark] = [transformed[row, column] for row, column in positions]
    return output


def fit_medians(fit_features: np.ndarray) -> np.ndarray:
    """Fit thresholds on fit identities, independently by landmark/frequency."""
    values = np.asarray(fit_features, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("Expected image-by-landmark-by-frequency features")
    return np.median(values, axis=0)


def encode(features: np.ndarray, medians: np.ndarray, landmarks: list[int], bits: int = 12) -> np.ndarray:
    values = np.asarray(features)
    thresholds = np.asarray(medians)
    if values.shape[1:] != thresholds.shape:
        raise ValueError("DCT features and fitted medians have different shapes")
    binary = values[:, landmarks, :bits] > thresholds[None, landmarks, :bits]
    return binary.reshape(len(values), len(landmarks) * bits)


def layout(landmarks: list[int], bits: int = 12) -> list[Block]:
    return [Block(i * bits, (i + 1) * bits, f"landmark_{landmark:02d}") for i, landmark in enumerate(landmarks)]
