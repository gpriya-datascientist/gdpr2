"""Test all klinken bad images and check box quality."""
import asyncio, sys, json, os
sys.path.insert(0, '.')
from inspector import inspect_image

async def test_all():
    bad_dir = r'C:\Users\gpngur01\Downloads\gdpr-ai2\data\klinken\bad'
    files = sorted(os.listdir(bad_dir))
    results = []
    for fname in files:
        path = os.path.join(bad_dir, fname)
        with open(path, 'rb') as f:
            img = f.read()
        r = await inspect_image(img, 'klinken')
        n_boxes = len(r.boxes)
        # Check if any box covers >70% of image (bad box)
        bad_boxes = [b for b in r.boxes if (b['x_max']-b['x_min'])*(b['y_max']-b['y_min']) > 0.5]
        status = 'FULL_BOX' if bad_boxes else ('NO_BOX' if n_boxes==0 else 'OK')
        results.append({
            'file': fname,
            'verdict': r.verdict,
            'confidence': round(r.confidence,2),
            'n_boxes': n_boxes,
            'status': status,
            'boxes': r.boxes
        })
        print(f"{status:10} | {r.verdict:7} | {r.confidence:.0%} | boxes={n_boxes} | {fname}")
    
    ok = sum(1 for r in results if r['status']=='OK' and r['verdict']=='ANOMALY')
    print(f"\nSummary: {ok}/{len(results)} correct with good boxes")
    return results

results = asyncio.run(test_all())
with open('box_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to box_test_results.json")
