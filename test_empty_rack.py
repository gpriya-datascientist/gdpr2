"""Test all Klinken_Empty_Rack images accuracy."""
import asyncio, sys, json, os
sys.path.insert(0, '.')
from inspector import inspect_image

async def test():
    base = r'C:\Users\gpngur01\Downloads\gdpr-ai2\data\Klinken_Empty_Rack'
    print("=== GOOD images (should predict GOOD) ===")
    good_correct = 0
    for f in sorted(os.listdir(f'{base}/good')):
        with open(f'{base}/good/{f}','rb') as fh: img=fh.read()
        r = await inspect_image(img, 'klinken_empty')
        ok = r.verdict == 'GOOD'
        if ok: good_correct += 1
        print(f"  {'OK' if ok else 'WRONG'} | {r.verdict} {r.confidence:.0%} | {f[:30]}")

    print("\n=== BAD images (should predict ANOMALY) ===")
    bad_correct = 0
    for f in sorted(os.listdir(f'{base}/bad')):
        with open(f'{base}/bad/{f}','rb') as fh: img=fh.read()
        r = await inspect_image(img, 'klinken_empty')
        ok = r.verdict == 'ANOMALY'
        if ok: bad_correct += 1
        n_boxes = len(r.boxes)
        box_ok = all((b['x_max']-b['x_min'])*(b['y_max']-b['y_min']) < 0.3 for b in r.boxes)
        print(f"  {'OK' if ok else 'WRONG'} | {r.verdict} {r.confidence:.0%} | boxes={n_boxes} {'GOOD_BOX' if box_ok else 'BAD_BOX'} | {f[:30]}")

    total = good_correct + bad_correct
    print(f"\nResult: {total}/18 correct ({total/18*100:.0f}%)")

asyncio.run(test())
