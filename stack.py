#!/usr/bin/env python3
"""
Astrophotography Stacker
- Star detection and alignment using astroalign
- Sigma-clipping stack for noise rejection
- Histogram stretch for visibility
"""

import os
import sys
import glob
import numpy as np
from PIL import Image

try:
    import astroalign as aa
    HAS_ASTROALIGN = True
except ImportError:
    HAS_ASTROALIGN = False

try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False


def load_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.dng', '.raw', '.cr2', '.nef', '.arw', '.orf'):
        if not HAS_RAWPY:
            print(f"  rawpy not installed, converting {path} via ffmpeg")
            import subprocess
            out = path + ".tiff"
            subprocess.run(["ffmpeg", "-i", path, "-y", out], check=True,
                           capture_output=True)
            return np.array(Image.open(out).convert("RGB"), dtype=np.float64)
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=16)
            return rgb.astype(np.float64) / 65535.0 * 255.0
    return np.array(Image.open(path).convert("RGB"), dtype=np.float64)


def save_image(arr, path):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path, optimize=True)
    print(f"  Saved: {path} ({os.path.getsize(path) / 1024:.0f} KB)")


def histogram_stretch(img, low_pct=0.5, high_pct=99.5):
    result = np.zeros_like(img)
    for c in range(3):
        ch = img[:, :, c]
        nz = ch[ch > 0]
        if len(nz) == 0:
            continue
        low = np.percentile(nz, low_pct)
        high = np.percentile(ch, high_pct)
        if high > low:
            result[:, :, c] = np.clip((ch - low) / (high - low) * 255, 0, 255)
    return result


def manual_align_and_stack(images):
    """Fallback: simple center-crop + average stack (no star alignment)."""
    h = min(im.shape[0] for im in images)
    w = min(im.shape[1] for im in images)
    cropped = []
    for im in images:
        cy, cx = im.shape[0] // 2, im.shape[1] // 2
        cropped.append(im[cy - h // 2:cy - h // 2 + h, cx - w // 2:cx - w // 2 + w])
    return np.mean(cropped, axis=0)


def sigma_clip_stack(images, sigma=2.0):
    stack = np.stack(images, axis=0)
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0)
    mask = np.abs(stack - mean) < sigma * np.maximum(std, 1.0)
    return np.sum(stack * mask, axis=0) / np.maximum(np.sum(mask, axis=0), 1)


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "input"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    os.makedirs(output_dir, exist_ok=True)

    # Find all image files
    exts = ["*.png", "*.jpg", "*.jpeg", "*.tiff", "*.tif", "*.dng",
            "*.raw", "*.cr2", "*.nef", "*.arw", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
        files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    files = sorted(set(files))

    if not files:
        print("ERROR: No image files found in", input_dir)
        sys.exit(1)

    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  {os.path.basename(f)} ({os.path.getsize(f) / 1024:.0f} KB)")

    # Load all images
    print("\nLoading images...")
    images = []
    for f in files:
        try:
            img = load_image(f)
            images.append(img)
            print(f"  Loaded {os.path.basename(f)}: {img.shape}")
        except Exception as e:
            print(f"  FAILED to load {os.path.basename(f)}: {e}")

    if len(images) < 2:
        print("Need at least 2 images to stack!")
        sys.exit(1)

    # Star alignment + stacking
    if HAS_ASTROALIGN:
        print(f"\nUsing astroalign for star detection and registration...")
        try:
            # Use first image as reference (keep its original size/orientation)
            ref = images[0]
            registered = [ref]

            for i in range(1, len(images)):
                try:
                    # astroalign handles rotation, scale and translation between frames
                    transform, (source_cat, target_cat) = aa.find_transform(
                        images[i], ref, detection_sigma=5, max_control_points=50
                    )
                    aligned = aa.apply_transform(transform, images[i], ref)
                    registered.append(aligned)
                    print(f"  Frame {i}: aligned (triangles matched)")
                except Exception as e:
                    print(f"  Frame {i}: alignment failed ({e}), using unaligned")
                    registered.append(images[i])

            print(f"\nSigma-clipping stack ({len(registered)} frames, sigma=2.0)...")
            stacked = sigma_clip_stack(registered, sigma=2.0)
        except Exception as e:
            print(f"\nAstroalign failed: {e}")
            print("Falling back to simple averaging...")
            stacked = sigma_clip_stack(images, sigma=2.0)
    else:
        print("\nastroalign not available, using simple average stack...")
        stacked = manual_align_and_stack(images)

    # Save raw stacked result
    print("\nSaving results...")
    save_image(stacked, os.path.join(output_dir, "stacked_raw.png"))

    # Save histogram-stretched version
    stretched = histogram_stretch(stacked)
    save_image(stretched, os.path.join(output_dir, "stacked_stretched.png"))

    # Save enhanced version with contrast boost
    enhanced = histogram_stretch(stacked, low_pct=2.0, high_pct=99.0)
    # Slight saturation boost
    hsv = np.zeros_like(enhanced)
    avg = np.mean(enhanced, axis=2, keepdims=True)
    factor = 1.3
    enhanced = np.clip(avg + factor * (enhanced - avg), 0, 255)
    save_image(enhanced, os.path.join(output_dir, "stacked_enhanced.png"))

    print(f"\nDone! {len(images)} frames stacked into {output_dir}/")


if __name__ == "__main__":
    main()
