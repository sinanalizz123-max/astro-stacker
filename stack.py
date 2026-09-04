#!/usr/bin/env python3
"""
Astrophotography Stacker
Pipeline:
  1. Load all frames and normalize to a common resolution
  2. Star-align each frame to the reference using astroalign
  3. Background-subtract each aligned frame (removes sky fog/black-lift)
  4. Sigma-clipping stack for noise rejection
  5. Save raw, stretched, and enhanced versions
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


def load_image(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.float64)


def save_image(arr, path):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path, optimize=True)
    print(f"  Saved: {path} ({os.path.getsize(path) / 1024:.0f} KB)")


def resize_keep_aspect(img, target_hw):
    """Resize image to target height,width (stretch exactly to fit)."""
    if img.shape[:2] == target_hw:
        return img
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    pil = pil.resize((target_hw[1], target_hw[0]), Image.LANCZOS)
    return np.array(pil, dtype=np.float64)


def sigma_clip_stack(images, sigma=2.0):
    stack = np.stack(images, axis=0)
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0)
    mask = np.abs(stack - mean) < sigma * np.maximum(std, 1.0)
    return np.sum(stack * mask, axis=0) / np.maximum(np.sum(mask, axis=0), 1)


def estimate_background(img, margin_frac=0.15):
    """Estimate additive sky background using dark border pixels."""
    h, w = img.shape[:2]
    m = int(min(h, w) * margin_frac)
    border = np.concatenate([
        img[:m, :].reshape(-1, 3),
        img[-m:, :].reshape(-1, 3),
        img[:, :m].reshape(-1, 3),
        img[:, -m:].reshape(-1, 3),
    ])
    return np.median(border, axis=0)


def background_subtract(img):
    """Subtract estimated additive sky background so sky ~0."""
    bg = estimate_background(img)
    return np.clip(img - bg, 0, 255)


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


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "frames"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    os.makedirs(output_dir, exist_ok=True)

    exts = ["*.png", "*.jpg", "*.jpeg", "*.tiff", "*.tif", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
        files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    files = sorted(set(files))

    if not files:
        print("ERROR: No image files found in", input_dir)
        sys.exit(1)

    print(f"Found {len(files)} files")

    print("\nLoading images...")
    images = []
    for f in files:
        try:
            img = load_image(f)
            images.append(img)
            print(f"  {os.path.basename(f)}: {img.shape}")
        except Exception as e:
            print(f"  FAILED {os.path.basename(f)}: {e}")

    if len(images) < 2:
        print("Need at least 2 frames!")
        sys.exit(1)

    # --- Choose common resolution: most frequent shape -------------------
    from collections import Counter
    shape_counts = Counter(im.shape[:2] for im in images)
    target_hw = shape_counts.most_common(1)[0][0]
    print(f"\nCommon resolution: {target_hw}")

    print("Normalizing all frames to common resolution...")
    for i in range(len(images)):
        if images[i].shape[:2] != target_hw:
            images[i] = resize_keep_aspect(images[i], target_hw)

    # --- Step 1: Align every frame to the reference ---------------------
    registered = []
    if HAS_ASTROALIGN:
        print("\nStep 1: Star alignment with astroalign...")
        ref = images[0]
        registered.append(ref)
        for i in range(1, len(images)):
            try:
                aligned, footprint = aa.register(images[i], ref)
                if aligned.shape[:2] != target_hw:
                    aligned = resize_keep_aspect(aligned, target_hw)
                registered.append(aligned)
                print(f"  Frame {i}: aligned")
            except Exception as e:
                print(f"  Frame {i}: alignment failed ({e}), using unaligned")
                registered.append(images[i])
    else:
        print("\nastroalign not available - using unaligned frames")
        registered = images

    # --- Step 2: Background subtraction (remove fog) --------------------
    print("\nStep 2: Background subtraction...")
    for i, im in enumerate(registered):
        bg = estimate_background(im)
        registered[i] = background_subtract(im)
        print(f"  Frame {i}: bg was {np.round(bg.astype(int)).tolist()}")

    # --- Step 3: Sigma-clip stack ---------------------------------------
    print(f"\nStep 3: Sigma-clip stack ({len(registered)} frames, sigma=2.0)...")
    stacked = sigma_clip_stack(registered, sigma=2.0)

    # --- Save -------------------------------------------------------------
    print("\nSaving raw stack...")
    save_image(stacked, os.path.join(output_dir, "stacked_raw.png"))

    print("Saving stretched version...")
    stretched = histogram_stretch(stacked)
    save_image(stretched, os.path.join(output_dir, "stacked_stretched.png"))

    print("Saving enhanced version...")
    enhanced = histogram_stretch(stacked, low_pct=2.0, high_pct=99.0)
    avg = np.mean(enhanced, axis=2, keepdims=True)
    enhanced = np.clip(avg + 1.3 * (enhanced - avg), 0, 255)
    save_image(enhanced, os.path.join(output_dir, "stacked_enhanced.png"))

    print(f"\nDone! {len(images)} frames stacked into {output_dir}/")


if __name__ == "__main__":
    main()