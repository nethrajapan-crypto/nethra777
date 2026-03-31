import json
import os

IN_JSON = "dataset_annotations.json"
OUT_HTML = "defect_gallery.html"

def make_gallery(data):
    imgs = sorted(data.keys())
    rows = []
    for p in imgs:
        webp = p.replace('\\', '/').replace('\\\\', '/')
        rows.append(f'<div class="tile"><img src="{webp}" alt="{os.path.basename(webp)}"><div class="caption">{os.path.basename(webp)}</div></div>')

    html = f"""
<html>
<head>
<meta charset="utf-8">
<title>Defect Images Gallery</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif; padding:18px}}
.count{{margin-bottom:12px}}
.grid{{display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:12px}}
.tile{{border:1px solid #ddd; padding:6px; background:#fff}}
.tile img{{width:100%; height:auto; display:block}}
.caption{{font-size:12px; color:#333; margin-top:6px}}
</style>
</head>
<body>
<h2>Defect Images Gallery</h2>
<div class="count">Images with annotations: {len(imgs)}</div>
<div class="grid">
{''.join(rows)}
</div>
</body>
</html>
"""
    with open(OUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Wrote {OUT_HTML} with {len(imgs)} images")

def main():
    if not os.path.exists(IN_JSON):
        print(f"Missing {IN_JSON}")
        return
    with open(IN_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    make_gallery(data)

if __name__ == '__main__':
    main()
