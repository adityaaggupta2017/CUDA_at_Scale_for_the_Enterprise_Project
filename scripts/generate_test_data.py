#!/usr/bin/env python3
"""
generate_test_data.py  –  Synthetic grayscale PGM image generator.

CUDA at Scale for the Enterprise – Independent Project

Generates a specified number of 256×256 8-bit grayscale PGM (P5) images
covering five pattern categories so that all four GPU operations
(blur, edge detection, histogram equalization, sharpening) produce visually
meaningful results:

  Category 0 – Linear gradient
  Category 1 – Concentric circles (good for blur / edge visibility)
  Category 2 – Low-contrast Gaussian blobs (benefits from hist. equalisation)
  Category 3 – Checkerboard + Gaussian noise (exercises sharpening)
  Category 4 – Mixed pattern + salt-and-pepper noise
"""

import argparse
import os
import struct
import sys

import numpy as np


# ---------------------------------------------------------------------------
# PGM writer (no external image library required)
# ---------------------------------------------------------------------------

def save_pgm(filename: str, array: np.ndarray) -> None:
    """Save a 2-D uint8 NumPy array as a binary PGM (P5) file."""
    h, w = array.shape
    header = f"P5\n{w} {h}\n255\n".encode()
    with open(filename, "wb") as f:
        f.write(header)
        f.write(array.astype(np.uint8).tobytes())


# ---------------------------------------------------------------------------
# Pattern generators
# ---------------------------------------------------------------------------

def make_gradient(size: int, rng: np.random.Generator) -> np.ndarray:
    """Linear intensity gradient with a random angle."""
    angle = rng.uniform(0, 2 * np.pi)
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    val = np.cos(angle) * xx + np.sin(angle) * yy
    val = (val - val.min()) / (val.max() - val.min() + 1e-9)
    return (val * 255).astype(np.uint8)


def make_circles(size: int, rng: np.random.Generator) -> np.ndarray:
    """Concentric rings centred at a random point, with additive noise."""
    cx = rng.uniform(0.3, 0.7) * size
    cy = rng.uniform(0.3, 0.7) * size
    x = np.arange(size, dtype=np.float32)
    y = np.arange(size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    freq = rng.uniform(0.04, 0.1)
    val = 0.5 + 0.5 * np.sin(freq * r * 2 * np.pi)
    noise = rng.normal(0, 0.05, (size, size))
    val = np.clip(val + noise, 0, 1)
    return (val * 255).astype(np.uint8)


def make_blobs(size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Low-contrast Gaussian blobs.
    Demonstrates that histogram equalisation reveals hidden structure.
    """
    img = np.zeros((size, size), dtype=np.float32)
    num_blobs = rng.integers(5, 15)
    x = np.arange(size, dtype=np.float32)
    y = np.arange(size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    for _ in range(num_blobs):
        cx = rng.uniform(0, size)
        cy = rng.uniform(0, size)
        sigma = rng.uniform(15, 50)
        amp   = rng.uniform(0.2, 0.8)
        blob  = amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) /
                              (2 * sigma ** 2))
        img  += blob
    img = np.clip(img, 0, 1)
    # Compress to low-contrast range [20, 120] to make equalisation effective.
    img = img * 100 + 20
    return img.astype(np.uint8)


def make_checkerboard(size: int, rng: np.random.Generator) -> np.ndarray:
    """Checkerboard with Gaussian noise (tests sharpening)."""
    tile = rng.integers(8, 32)
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)
    board = ((xx // tile + yy // tile) % 2).astype(np.float32)
    noise = rng.normal(0, 0.12, (size, size))
    val   = np.clip(board + noise, 0, 1)
    return (val * 255).astype(np.uint8)


def make_mixed(size: int, rng: np.random.Generator) -> np.ndarray:
    """Random geometric shapes + salt-and-pepper noise."""
    img = np.zeros((size, size), dtype=np.float32)

    # Random rectangles.
    for _ in range(rng.integers(3, 8)):
        x0, y0 = rng.integers(0, size, 2)
        x1, y1 = rng.integers(0, size, 2)
        if x0 > x1: x0, x1 = x1, x0  # noqa: E701
        if y0 > y1: y0, y1 = y1, y0  # noqa: E701
        val = rng.uniform(0.2, 1.0)
        img[y0:y1, x0:x1] = val

    # Random ellipses.
    x = np.arange(size, dtype=np.float32)
    y = np.arange(size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    for _ in range(rng.integers(2, 6)):
        cx = rng.uniform(0, size)
        cy = rng.uniform(0, size)
        rx = rng.uniform(10, size // 4)
        ry = rng.uniform(10, size // 4)
        mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
        img[mask] = rng.uniform(0.3, 0.9)

    # Salt-and-pepper noise.
    noise_frac = rng.uniform(0.02, 0.08)
    salt  = rng.random((size, size)) < noise_frac / 2
    pepper = rng.random((size, size)) < noise_frac / 2
    img[salt]   = 1.0
    img[pepper] = 0.0

    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# Mapping from category index to generator function.
_GENERATORS = [
    make_gradient,
    make_circles,
    make_blobs,
    make_checkerboard,
    make_mixed,
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic grayscale PGM test images.")
    parser.add_argument("--output", default="data/input",
                        help="Output directory (default: data/input)")
    parser.add_argument("--count", type=int, default=200,
                        help="Total number of images to generate (default: 200)")
    parser.add_argument("--size", type=int, default=256,
                        help="Image side length in pixels (default: 256)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    num_categories = len(_GENERATORS)

    print(f"Generating {args.count} synthetic {args.size}×{args.size} "
          f"PGM images in '{args.output}' …")

    for i in range(args.count):
        category = i % num_categories
        gen      = _GENERATORS[category]
        img      = gen(args.size, rng)
        filename = os.path.join(args.output, f"image_{i:04d}.pgm")
        save_pgm(filename, img)

        if (i + 1) % 50 == 0 or (i + 1) == args.count:
            print(f"  {i + 1}/{args.count} images written …")

    print(f"Done – {args.count} images saved to '{args.output}'.")


if __name__ == "__main__":
    main()
