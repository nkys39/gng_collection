"""Generate shape images for GNG testing."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def generate_double_ring(
    output_path: str,
    size: int = 400,
    r1_inner: int = 60,
    r1_outer: int = 100,
    r2_inner: int = 140,
    r2_outer: int = 180,
    bg_color: str = "#FFFFFF",
    ring_color: str = "#87CEEB",
) -> None:
    """Generate a double ring (donut) shape image.

    Args:
        output_path: Path to save the image.
        size: Image size (width and height).
        r1_inner: Inner radius of inner ring.
        r1_outer: Outer radius of inner ring.
        r2_inner: Inner radius of outer ring.
        r2_outer: Outer radius of outer ring.
        bg_color: Background color (hex).
        ring_color: Ring color (hex).
    """
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    center = size // 2

    # Draw outer ring (large circle - hole)
    draw.ellipse(
        [center - r2_outer, center - r2_outer, center + r2_outer, center + r2_outer],
        fill=ring_color,
    )
    draw.ellipse(
        [center - r2_inner, center - r2_inner, center + r2_inner, center + r2_inner],
        fill=bg_color,
    )

    # Draw inner ring (large circle - hole)
    draw.ellipse(
        [center - r1_outer, center - r1_outer, center + r1_outer, center + r1_outer],
        fill=ring_color,
    )
    draw.ellipse(
        [center - r1_inner, center - r1_inner, center + r1_inner, center + r1_inner],
        fill=bg_color,
    )

    img.save(output_path)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate shape images for GNG testing")
    parser.add_argument(
        "--shape",
        type=str,
        default="double_ring",
        choices=["double_ring"],
        help="Shape type to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: <shape>.png in current directory)",
    )
    parser.add_argument("--size", type=int, default=400, help="Image size")

    args = parser.parse_args()

    output_path = args.output or f"{args.shape}.png"

    if args.shape == "double_ring":
        generate_double_ring(output_path, size=args.size)


if __name__ == "__main__":
    main()
