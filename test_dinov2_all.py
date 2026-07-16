"""Test DINOv2 on all good and bad images across all projects."""
import asyncio, sys, os
sys.path.insert(0,'.')
from inspector import inspect_image, PROJECTS

BASE = r'C:\Users\gpngur01\Downloads\gdpr-ai2'

async def test_project(proj_id, proj):
    good_dir = os.path.join(BASE, proj['good_dir'])
    bad_dir  = os.path.join(BASE, proj['bad_dir'])
    results  = {'tp':0,'tn':0,'fp':0,'fn':0}
    for f in os.listdir(bad_dir):
        if not f.endswith('.jpg'): continue
        with open(os.path.join(bad_dir,f),'rb') as fh: img=fh.read()
        r = await inspect_image(img, proj_id)
        if r.verdict=='ANOMALY': results['tp']+=1
        else: results['fn']+=1
    for f in os.listdir(good_dir):
        if not f.endswith('.jpg'): continue
        with open(os.path.join(good_dir,f),'rb') as fh: img=fh.read()
        r = await inspect_image(img, proj_id)
        if r.verdict=='GOOD': results['tn']+=1
        else: results['fp']+=1
    total = results['tp']+results['tn']+results['fp']+results['fn']
    correct = results['tp']+results['tn']
    acc = round(correct/total*100) if total else 0
    print(f"  {proj['name']}: {correct}/{total} ({acc}%) | TP:{results['tp']} TN:{results['tn']} FP:{results['fp']} FN:{results['fn']}")
    return proj_id, acc, correct, total

async def main():
    import inspector as insp
    insp.FORCE_LLM = False
    print("DINOv2 accuracy test — all projects\n" + "="*50)
    results = []
    for proj_id, proj in PROJECTS.items():
        r = await test_project(proj_id, proj)
        results.append(r)
    print("\nSUMMARY")
    print(f"{'Project':<25} {'Accuracy':<10} {'Correct'}")
    print("-"*45)
    total_c, total_t = 0, 0
    for pid, acc, c, t in results:
        print(f"{pid:<25} {acc}%{'':<7} {c}/{t}")
        total_c+=c; total_t+=t
    print(f"\nOverall: {total_c}/{total_t} ({round(total_c/total_t*100)}%)")

asyncio.run(main())
