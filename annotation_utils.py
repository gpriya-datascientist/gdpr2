"""
Read Pascal VOC XML annotations and use them for precise box placement.
When an XML annotation exists for an image, use those boxes instead of DINOv2 patch localization.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


def load_annotations(image_path: str) -> Optional[list[dict]]:
    """Load Pascal VOC XML annotation for an image if it exists."""
    img_path = Path(image_path)
    # Look for XML in same folder or annotations/ subfolder
    candidates = [
        img_path.with_suffix('.xml'),
        img_path.parent / 'annotations' / (img_path.stem + '.xml'),
        img_path.parent / 'xml' / (img_path.stem + '.xml'),
        img_path.parent.parent / 'annotations' / (img_path.stem + '.xml'),
        img_path.parent.parent / 'xml' / (img_path.stem + '.xml'),
    ]
    xml_path = next((p for p in candidates if p.exists()), None)
    if not xml_path:
        return None
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        boxes = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bb = obj.find('bndbox')
            xmin = int(bb.find('xmin').text)
            ymin = int(bb.find('ymin').text)
            xmax = int(bb.find('xmax').text)
            ymax = int(bb.find('ymax').text)
            boxes.append({
                'name': name,
                'x_min': xmin / w,
                'y_min': ymin / h,
                'x_max': xmax / w,
                'y_max': ymax / h,
                'score': 1.0,
                'source': 'annotation'
            })
        return boxes if boxes else None
    except Exception as e:
        return None


def save_annotation(image_path: str, image_w: int, image_h: int, boxes: list[dict]):
    """Save DINOv2 predicted boxes as Pascal VOC XML annotation."""
    img_path = Path(image_path)
    ann_dir = img_path.parent / 'annotations'
    ann_dir.mkdir(exist_ok=True)
    xml_path = ann_dir / (img_path.stem + '.xml')

    objs = ''
    for box in boxes:
        xmin = int(box['x_min'] * image_w)
        ymin = int(box['y_min'] * image_h)
        xmax = int(box['x_max'] * image_w)
        ymax = int(box['y_max'] * image_h)
        name = box.get('name', box.get('defect_type', 'anomaly'))
        objs += f'''  <object>
    <name>{name}</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>{xmin}</xmin>
      <ymin>{ymin}</ymin>
      <xmax>{xmax}</xmax>
      <ymax>{ymax}</ymax>
    </bndbox>
  </object>\n'''

    xml = f'''<annotation>
  <folder>annotations</folder>
  <filename>{img_path.name}</filename>
  <source><database>PressVision2</database></source>
  <size><width>{image_w}</width><height>{image_h}</height><depth>1</depth></size>
  <segmented>0</segmented>
{objs}</annotation>'''
    xml_path.write_text(xml, encoding='utf-8')
    return str(xml_path)


def count_annotations(data_dir: str) -> dict:
    """Count how many images have XML annotations in a dataset directory."""
    base = Path(data_dir)
    stats = {}
    for split in ['good', 'bad']:
        folder = base / split
        if not folder.exists():
            continue
        images = list(folder.glob('*.jpg')) + list(folder.glob('*.jpeg')) + list(folder.glob('*.png'))
        annotated = sum(1 for img in images if load_annotations(str(img)) is not None)
        stats[split] = {'total': len(images), 'annotated': annotated}
    return stats
