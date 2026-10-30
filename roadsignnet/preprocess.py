"""A single deterministic preprocessing path for training and inference."""

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_SIZE = 32
PREPROCESSING = {"width": 32, "height": 32, "channels": 3, "color_mode": "RGB",
                 "scale": 255.0, "resize": "opencv_INTER_AREA", "exif_transpose": True}
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 25_000_000


def preprocess_image(source: str | Path | bytes) -> np.ndarray:
    """Decode a bounded image, orient, convert to RGB, resize and scale to [0,1]."""
    if isinstance(source, bytes):
        payload = source
    else:
        path = Path(source)
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("Image exceeds the 20 MB limit.")
        payload = path.read_bytes()
    if not payload or len(payload) > MAX_BYTES:
        raise ValueError("Image is empty or exceeds the 20 MB limit.")
    try:
        with Image.open(BytesIO(payload)) as image:
            if image.format not in {"JPEG", "PNG", "PPM"}:
                raise ValueError("Use a JPEG, PNG or GTSRB PPM image.")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("Image exceeds the 25 megapixel limit.")
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Cannot decode this image. Choose a valid JPG or PNG.") from exc
    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0
