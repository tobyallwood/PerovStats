from dataclasses import dataclass
import numpy as np


@dataclass
class Grain:
    grain_id: int
    grain_area: float | None = None
    grain_circularity_rating: float | None = None


@dataclass
class ImageData:
    topostats_object: any
    mask: np.ndarray | None = None
    high_pass: np.ndarray | None = None
    low_pass: np.ndarray | None = None
    grains: dict[int, Grain] | None = None
    file_directory: str | None = None
    filename: str | None = None
    mask_rgb: np.ndarray | None = None
    grains_per_nm2: float | None = None
    mask_size_x_nm: float | None = None
    mask_size_y_nm: float | None = None
    mask_area_nm: float | None = None
    num_grains: int | None = None
    cutoff_freq_nm: float | None = None
    cutoff: float | None = None


@dataclass
class PerovStats:
    images: list[ImageData] | None = None
    config: dict[str, any] | None = None
