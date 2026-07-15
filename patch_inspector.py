"""Patch inspector.py to use XML annotations for box placement."""
import re
p = r'C:\Users\gpngur01\Downloads\gdpr-ai2\inspector.py'
with open(p, 'r', encoding='utf-8') as f:
    content = f.read()

old = '''            from embedder import predict
            verdict, conf, boxes, reason = await asyncio.to_thread(
                predict, model, _get_embedder(), image_bytes)
            latency = (time.perf_counter()-start)*1000
            r = InspectionResult(verdict=verdict, confidence=conf, reason=reason,
                defect_type="crack" if verdict=="ANOMALY" else None,
                backend_used="dinov2", latency_ms=latency,
                project=project_id, boxes=boxes)'''

new = '''            from embedder import predict
            from annotation_utils import load_annotations
            import tempfile as _tf, os as _os
            verdict, conf, dinov2_boxes, reason = await asyncio.to_thread(
                predict, model, _get_embedder(), image_bytes)
            with _tf.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp.write(image_bytes); tmp_path = tmp.name
            xml_boxes = load_annotations(tmp_path)
            _os.unlink(tmp_path)
            boxes = xml_boxes if xml_boxes else dinov2_boxes
            box_src = "annotation" if xml_boxes else "dinov2"
            latency = (time.perf_counter()-start)*1000
            r = InspectionResult(verdict=verdict, confidence=conf, reason=reason,
                defect_type="crack" if verdict=="ANOMALY" else None,
                backend_used=f"dinov2+{box_src}", latency_ms=latency,
                project=project_id, boxes=boxes)'''

if old in content:
    content = content.replace(old, new)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(content)
    print("PATCHED OK")
else:
    print("NOT FOUND — looking for nearest match...")
    idx = content.find('from embedder import predict')
    print(repr(content[idx:idx+400]))
