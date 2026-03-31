import argparse
import os
from xml.etree import ElementTree as ET
from PIL import Image, ImageDraw, ImageFont


def parse_bboxes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text if obj.find('name') is not None else 'obj'
        b = obj.find('bndbox')
        xmin = int(b.find('xmin').text)
        ymin = int(b.find('ymin').text)
        xmax = int(b.find('xmax').text)
        ymax = int(b.find('ymax').text)
        boxes.append((name, (xmin, ymin, xmax, ymax)))
    return boxes


def draw_boxes(image_path, xml_path, out_path=None):
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    boxes = parse_bboxes(xml_path)

    # Try to load a default font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for name, (xmin, ymin, xmax, ymax) in boxes:
        draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=4)
        text = name
        # determine text size with best available method
        if font:
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                try:
                    fb = font.getbbox(text)
                    tw, th = fb[2] - fb[0], fb[3] - fb[1]
                except Exception:
                    tw, th = (len(text) * 6, 11)
        else:
            tw, th = (len(text) * 6, 11)
        tx = xmin
        ty = max(0, ymin - th - 2)
        draw.rectangle([tx, ty, tx + tw + 4, ty + th + 2], fill='red')
        draw.text((tx + 2, ty), text, fill='white', font=font)

    if out_path is None:
        base, ext = os.path.splitext(image_path)
        out_path = base + '_annotated' + ext

    img.save(out_path)
    print('Saved annotated image to', out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', '-i', required=True, help='Path to image file')
    p.add_argument('--xml', '-x', required=True, help='Path to annotation XML')
    p.add_argument('--out', '-o', help='Output path (optional)')
    args = p.parse_args()

    draw_boxes(args.image, args.xml, args.out)


if __name__ == '__main__':
    main()
