#!/usr/bin/env python3
"""
Image subtraction: Subtract background from all images to highlight defects.
Uses first good image as reference or computes average background.
"""

import json
import os
from PIL import Image
import numpy as np
from pathlib import Path

IN_JSON = "dataset_annotations.json"
OUT_DIR = "defect_subtraction"
REFERENCE_IMG = "images/Missing_hole/01_missing_hole_01.jpg"

def load_image_array(img_path, target_size=(800, 600)):
    """Load image as numpy array and resize to target size, normalized to [0, 1]."""
    try:
        img = Image.open(img_path).convert("RGB")
        # Resize to target size
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0
    except:
        return None

def image_difference(img1_arr, img2_arr):
    """Compute absolute difference between two images (same shape required)."""
    if img1_arr.shape != img2_arr.shape:
        # Shouldn't happen if resized consistently
        return None
    diff = np.abs(img1_arr - img2_arr)
    # Normalize to [0, 255]
    diff = (diff * 255).astype(np.uint8)
    return diff

def highlight_defects(img_arr, reference_arr, threshold=30):
    """Highlight pixels that differ from reference by more than threshold."""
    # Convert to grayscale for comparison
    img_gray = np.mean(img_arr, axis=2)
    ref_gray = np.mean(reference_arr, axis=2)
    
    # Compute difference
    diff = np.abs(img_gray - ref_gray)
    
    # Create mask of significant differences
    mask = (diff * 255) > threshold
    
    # Create output: highlight differences in red
    output = img_arr.copy()
    output[mask, 0] = np.clip(output[mask, 0] + 0.3, 0, 1)  # Boost red
    output[mask, 1] = np.clip(output[mask, 1] * 0.5, 0, 1)  # Reduce green
    output[mask, 2] = np.clip(output[mask, 2] * 0.5, 0, 1)  # Reduce blue
    
    return (output * 255).astype(np.uint8)

def main():
    if not os.path.exists(IN_JSON):
        print(f"Missing {IN_JSON}")
        return

    with open(IN_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load reference image
    if not os.path.exists(REFERENCE_IMG):
        print(f"Reference image not found: {REFERENCE_IMG}")
        print("Using first image as reference")
        first_img_path = list(data.keys())[0]
        reference_img = load_image_array(first_img_path)
        ref_display = first_img_path
    else:
        reference_img = load_image_array(REFERENCE_IMG)
        ref_display = REFERENCE_IMG

    if reference_img is None:
        print(f"Failed to load reference image: {ref_display}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    processed = 0
    failed = 0
    
    print(f"Processing {len(data)} images with reference: {ref_display}")
    print("Saving subtraction (difference) images...")

    for i, (img_path, objs) in enumerate(data.items()):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(data)}...", end='\r')
        
        # Normalize path
        img_path_norm = img_path.replace('\\', '/')
        full_img_path = os.path.join("images", img_path_norm.replace("images/", ""))
        
        if not os.path.exists(full_img_path):
            failed += 1
            continue

        # Load image
        img_arr = load_image_array(full_img_path)
        if img_arr is None:
            failed += 1
            continue

        # Compute difference
        diff_img = image_difference(img_arr, reference_img)
        
        if diff_img is None:
            failed += 1
            continue
        
        # Create output path
        rel_path = img_path_norm.replace("images/", "")
        out_path = os.path.join(OUT_DIR, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # Save as PIL Image
        diff_pil = Image.fromarray(diff_img)
        diff_pil.save(out_path, "JPEG", quality=85)
        processed += 1

    print(f"\n✓ Processed {processed} images (failed: {failed})")
    print(f"  Output: {OUT_DIR}/")
    
    # Create summary HTML
    build_subtraction_gallery(data, processed)

def build_subtraction_gallery(data, count):
    """Build comparison gallery showing original vs difference."""
    out_html = "subtraction_gallery.html"
    
    html = """
<html>
<head>
<meta charset="utf-8">
<title>PCB Defect Subtraction Gallery</title>
<style>
body { font-family: Arial, sans-serif; background: #f5f5f5; padding: 20px; }
.container { max-width: 1200px; margin: 0 auto; }
.header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 20px; }
.item { background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.item-title { background: #333; color: white; padding: 12px; font-size: 14px; }
.comparison { display: grid; grid-template-columns: 1fr 1fr; }
.image-side { position: relative; }
.image-label { position: absolute; top: 8px; left: 8px; background: rgba(0,0,0,0.6); color: white; padding: 6px 10px; border-radius: 4px; font-size: 12px; font-weight: bold; }
.image-side img { width: 100%; height: auto; display: block; }
</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>PCB Defect Subtraction Gallery</h1>
<p>Left: Original Image | Right: Difference from Reference (defects highlighted)</p>
<p>Processed: <strong>""" + str(count) + """ images</strong></p>
</div>

<div class="gallery">
"""

    for img_path in list(data.keys())[:50]:  # Show first 50 for demo
        img_url = img_path.replace('\\', '/')
        img_display = img_url.split('/')[-1]
        diff_url = f"defect_subtraction/{img_url.split('images/')[-1]}"
        
        html += f"""
<div class="item">
  <div class="item-title">{img_display}</div>
  <div class="comparison">
    <div class="image-side">
      <div class="image-label">Original</div>
      <img src="{img_url}" alt="original">
    </div>
    <div class="image-side">
      <div class="image-label">Difference</div>
      <img src="{diff_url}" alt="difference">
    </div>
  </div>
</div>
"""

    html += """
</div>
</div>
</body>
</html>
"""

    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ Created {out_html}")

if __name__ == '__main__':
    main()
