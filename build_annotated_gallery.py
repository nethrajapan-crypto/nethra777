#!/usr/bin/env python3
"""
Build annotated images with bbox overlays and create an interactive gallery.
"""

import json
import os
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

IN_JSON = "dataset_annotations.json"
OUT_DIR = "annotated_images_display"
OUT_HTML = "defect_gallery_annotated.html"

COLORS = {
    "missing_hole": "#FF6B6B",
    "mouse_bite": "#4ECDC4",
    "open_circuit": "#FFE66D",
    "short": "#95E1D3",
    "spur": "#C7CEEA",
    "spurious_copper": "#FF8B94"
}

def get_font():
    """Return a font; fallback to default if unavailable."""
    try:
        return ImageFont.truetype("arial.ttf", 16)
    except:
        return ImageFont.load_default()

def draw_bbox_on_image(img_path, objs, output_path):
    """Draw bboxes on image and save."""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Failed to open {img_path}: {e}")
        return False

    draw = ImageDraw.Draw(img, "RGBA")
    font = get_font()

    for obj in objs:
        bbox = obj.get("bbox", [])
        cls = obj.get("class", "unknown")
        color = COLORS.get(cls, "#FFFFFF")
        # Convert hex to RGBA
        rgba = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (255,)
        line_rgba = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (200,)

        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=line_rgba, width=3)
            # Draw label background + text
            label = cls
            bbox_text = draw.textbbox((x1, y1), label, font=font)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
            draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=rgba)
            draw.text((x1 + 2, y1 - text_h - 2), label, fill=(255, 255, 255), font=font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "JPEG", quality=85)
    return True

def main():
    if not os.path.exists(IN_JSON):
        print(f"Missing {IN_JSON}")
        return

    with open(IN_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    count = 0

    print("Generating annotated images...")
    for img_path, objs in data.items():
        # Normalize path
        img_path_normalized = img_path.replace('\\', '/')
        full_img_path = os.path.join("images", img_path_normalized.replace("images/", ""))
        
        if not os.path.exists(full_img_path):
            print(f"Image not found: {full_img_path}")
            continue

        # Create output path with same subfolder structure
        rel_path = img_path_normalized.replace("images/", "")
        out_path = os.path.join(OUT_DIR, rel_path)
        
        if draw_bbox_on_image(full_img_path, objs, out_path):
            count += 1

    print(f"Generated {count} annotated images in {OUT_DIR}")

    # Build gallery
    build_gallery(data, count)

def build_gallery(data, count):
    """Build filtered HTML gallery."""
    classes = sorted(set(obj["class"] for objs in data.values() for obj in objs))
    
    # Group images by class
    by_class = {c: [] for c in classes}
    for img_path, objs in data.items():
        for obj in objs:
            cls = obj["class"]
            rel_path = img_path.replace('\\', '/').replace("images/", "")
            annotated_rel = f"annotated_images_display/{rel_path}"
            if annotated_rel not in by_class[cls]:
                by_class[cls].append((img_path.replace('\\', '/').split('/')[-1], annotated_rel))

    html_parts = ["""
<html>
<head>
<meta charset="utf-8">
<title>Defect Gallery with Annotations</title>
<style>
body { font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }
.header { margin-bottom: 30px; }
.stats { font-size: 16px; margin: 10px 0; }
.filter { margin: 10px 0; }
button { padding: 8px 16px; margin: 4px; font-size: 14px; border: 1px solid #ccc; background: #fff; cursor: pointer; }
button.active { background: #007bff; color: white; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 15px; margin: 20px 0; }
.card { background: white; border: 1px solid #ddd; border-radius: 4px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.card img { width: 100%; height: auto; display: block; }
.card-info { padding: 10px; }
.class-label { display: inline-block; padding: 4px 8px; border-radius: 3px; color: white; font-size: 12px; font-weight: bold; }
</style>
<script>
function filter_class(cls) {
  var btns = document.querySelectorAll('.filter button');
  btns.forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  
  var cards = document.querySelectorAll('.card');
  cards.forEach(c => {
    if (cls === 'all' || c.dataset.class === cls) {
      c.style.display = 'block';
    } else {
      c.style.display = 'none';
    }
  });
}
</script>
</head>
<body>
<div class="header">
<h1>PCB Defect Gallery with Detected Annotations</h1>
<div class="stats">
Total Annotated Images: <strong>{count}</strong><br>
Total Defects Detected: <strong>2,953</strong> across 6 classes
</div>
</div>

<div class="filter">
<strong>Filter by Class:</strong><br>
<button class="active" onclick="filter_class('all')">All Classes</button>
""".format(count=count)]

    class_colors = {
        "missing_hole": "#FF6B6B",
        "mouse_bite": "#4ECDC4",
        "open_circuit": "#FFE66D",
        "short": "#95E1D3",
        "spur": "#C7CEEA",
        "spurious_copper": "#FF8B94"
    }

    for cls in classes:
        html_parts.append(f'<button onclick="filter_class(\'{cls}\')">{cls} ({len(by_class[cls])})</button>')

    html_parts.append("</div>\n<div class='gallery'>")

    # Add cards for each image
    for cls in classes:
        for name, rel_path in sorted(by_class[cls]):
            color = class_colors.get(cls, "#ccc")
            html_parts.append(f"""
<div class="card" data-class="{cls}">
  <img src="{rel_path}" alt="{name}">
  <div class="card-info">
    <div style="font-size: 12px; color: #555;">{name}</div>
    <div class="class-label" style="background: {color};">{cls}</div>
  </div>
</div>
""")

    html_parts.append("</div></body></html>")

    html = "".join(html_parts)
    with open(OUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Wrote {OUT_HTML}")

if __name__ == '__main__':
    main()
