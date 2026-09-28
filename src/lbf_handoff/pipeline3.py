"""Pipeline 3: ArcFace spatial sampling and the fitted sign encoder."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from sklearn.decomposition import PCA

from .pipeline2 import layout


ARCFACE_INPUT = 112
ARCFACE_TARGET = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32
)


def five_points(landmarks68: np.ndarray) -> np.ndarray:
    return np.stack([
        landmarks68[36:42].mean(axis=0), landmarks68[42:48].mean(axis=0),
        landmarks68[30], landmarks68[48], landmarks68[54],
    ]).astype(np.float32)


def similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares rotation, uniform scale, and translation as a 2-by-3 matrix."""
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_d, target_d = source - source_mean, target - target_mean
    covariance = target_d.T @ source_d / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    direction = np.diag([1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt))])
    rotation = u @ direction @ vt
    scale = np.trace(np.diag(singular) @ direction) / ((source_d ** 2).sum() / len(source))
    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = target_mean - scale * rotation @ source_mean
    return matrix


def arcface_warp(aligned_rgb: np.ndarray, landmarks68: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = similarity_transform(five_points(landmarks68), ARCFACE_TARGET)
    crop = cv2.warpAffine(aligned_rgb, matrix, (ARCFACE_INPUT, ARCFACE_INPUT), flags=cv2.INTER_LINEAR)
    mapped = landmarks68.astype(np.float32) @ matrix[:, :2].T + matrix[:, 2]
    return crop, mapped


def bilinear_sample(maps: np.ndarray, points: np.ndarray, input_size: int = ARCFACE_INPUT) -> np.ndarray:
    """Sample N-by-C-by-H-by-W maps at N-by-L input-image coordinates."""
    count, channels, height, width = maps.shape
    x = np.clip((points[..., 0] + 0.5) * width / input_size - 0.5, 0, width - 1)
    y = np.clip((points[..., 1] + 0.5) * height / input_size - 0.5, 0, height - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    rows = np.arange(count)[:, None]
    top = maps[rows, :, y0, x0] * (1 - fx) + maps[rows, :, y0, x1] * fx
    bottom = maps[rows, :, y1, x0] * (1 - fx) + maps[rows, :, y1, x1] * fx
    return (top * (1 - fy) + bottom * fy).astype(np.float32)


def seeded_rotation(dimensions: int, bits: int, seed: int = 42) -> np.ndarray:
    orthogonal, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(dimensions, dimensions)))
    return orthogonal[:, :bits].astype(np.float32)


def fit_encoder(descriptors: np.ndarray, dimensions: int = 64, bits: int = 12, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA-whitening on fit identities and fold rotation into one matrix."""
    pca = PCA(n_components=dimensions, svd_solver="randomized", random_state=seed)
    pca.fit(np.asarray(descriptors, dtype=np.float64))
    whiten = pca.components_.T / np.sqrt(np.maximum(pca.explained_variance_, 1e-12))
    matrix = whiten @ seeded_rotation(dimensions, bits, seed).astype(np.float64)
    return pca.mean_.astype(np.float32), matrix.astype(np.float32)


def encode(descriptors: np.ndarray, mean: np.ndarray, matrix: np.ndarray, landmarks: list[int]) -> np.ndarray:
    binary = (np.asarray(descriptors, dtype=np.float32) - mean) @ matrix > 0.0
    selected = binary[:, landmarks, :]
    return selected.reshape(len(selected), -1)


def load_encoder(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one of the two frozen fit-only encoders included in `frozen/`."""
    with np.load(path) as artifact:
        mean = np.asarray(artifact["mean"], dtype=np.float32)
        matrix = np.asarray(artifact["matrix"], dtype=np.float32)
    if mean.shape != (512,) or matrix.shape != (512, 12):
        raise ValueError(f"Unexpected P3 encoder shapes: {mean.shape}, {matrix.shape}")
    return mean, matrix


class ArcFaceStageExtractor:
    """Thin lazy ONNX wrapper so result reproduction does not require ONNX."""

    def __init__(self, model_path: Path, stage_tensor: str = "BatchNormalization_126") -> None:
        import onnx
        import onnxruntime
        model = onnx.load(str(model_path))
        outputs = {output.name for output in model.graph.output}
        if stage_tensor not in outputs:
            model.graph.output.append(onnx.helper.make_tensor_value_info(stage_tensor, onnx.TensorProto.FLOAT, None))
        self.session = onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.stage_tensor = stage_tensor

    def __call__(self, crops_rgb: np.ndarray) -> np.ndarray:
        tensor = (crops_rgb.astype(np.float32).transpose(0, 3, 1, 2) - 127.5) / 127.5
        (maps,) = self.session.run([self.stage_tensor], {self.input_name: tensor})
        return maps.astype(np.float32)
