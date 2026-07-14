"""
Fast local embedder using DINOv2 ViT-S/14 + logistic regression.
Adapted from PressVisionLoop local_embed.py.
~50ms/image after first load. No cloud calls.
"""
import asyncio, hashlib, io, logging, pickle, os
from pathlib import Path
from collections import deque
import numpy as np
from PIL import Image as PILImage

log = logging.getLogger(__name__)

MODEL_NAME  = "dinov2_vits14"
CACHE_DIR   = Path("data/models/torch")
PATCH_SIZE  = 448
PATCH_GRID  = 32
BANK_MAX    = 10_000
PATCH_QUANT = 0.95  # lower = more sensitive, better localization with few images
MAX_BOXES   = 5

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DinoV2Embedder:
    """Lazy-loading DINOv2 with content-addressed embedding cache."""
    def __init__(self):
        self._model = None
        self._cache: dict[str,np.ndarray] = {}
        self._patch_cache: dict[str,np.ndarray] = {}

    def _load(self):
        if self._model is None:
            os.environ.setdefault("TORCH_HOME", str(CACHE_DIR))
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            import torch
            log.info("Loading %s via torch.hub...", MODEL_NAME)
            self._model = torch.hub.load("facebookresearch/dinov2",
                                          MODEL_NAME, verbose=False)
            self._model.eval()
            log.info("DINOv2 loaded OK")
        return self._model

    @staticmethod
    def _preprocess(data: bytes) -> np.ndarray:
        img = PILImage.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        scale = 256 / min(w, h)
        img = img.resize((max(224,round(w*scale)), max(224,round(h*scale))), PILImage.BICUBIC)
        w, h = img.size
        l, t = (w-224)//2, (h-224)//2
        img = img.crop((l, t, l+224, t+224))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return ((arr - _MEAN) / _STD).transpose(2,0,1)

    def embed(self, data: bytes) -> np.ndarray:
        sha = hashlib.sha256(data).hexdigest()
        if sha in self._cache: return self._cache[sha]
        import torch
        model = self._load()
        with torch.inference_mode():
            t = torch.from_numpy(self._preprocess(data)).unsqueeze(0)
            vec = model(t).squeeze(0).numpy().astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-12
        self._cache[sha] = vec
        return vec

    @staticmethod
    def _preprocess_sq(data: bytes) -> np.ndarray:
        img = PILImage.open(io.BytesIO(data)).convert("RGB")
        img = img.resize((PATCH_SIZE, PATCH_SIZE), PILImage.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return ((arr - _MEAN) / _STD).transpose(2,0,1)

    def embed_patches(self, data: bytes) -> np.ndarray:
        sha = hashlib.sha256(data).hexdigest()
        if sha in self._patch_cache: return self._patch_cache[sha]
        import torch
        model = self._load()
        with torch.inference_mode():
            t = torch.from_numpy(self._preprocess_sq(data)).unsqueeze(0)
            toks = model.forward_features(t)["x_norm_patchtokens"]
            pts = toks.squeeze(0).numpy().astype(np.float32)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12
        self._patch_cache[sha] = pts
        return pts


def _components(mask: np.ndarray):
    rows, cols = mask.shape
    seen = np.zeros((rows,cols), dtype=bool)
    comps = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r,c] or seen[r,c]: continue
            seen[r,c] = True
            q = deque([(r,c)]); comp = []
            while q:
                cr,cc = q.popleft(); comp.append((cr,cc))
                for nr,nc in ((cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)):
                    if 0<=nr<rows and 0<=nc<cols and mask[nr,nc] and not seen[nr,nc]:
                        seen[nr,nc]=True; q.append((nr,nc))
            comps.append(comp)
    return comps


def boxes_from_distances(dists: np.ndarray, threshold: float):
    grid = dists.reshape(PATCH_GRID, PATCH_GRID)
    mask = grid > threshold
    if not mask.any():
        mask = np.zeros_like(mask)
        top = np.argsort(grid, axis=None)[-MAX_BOXES:]
        mask[np.unravel_index(top, grid.shape)] = True
    boxes = []
    for comp in _components(mask):
        r0,r1 = min(r for r,_ in comp), max(r for r,_ in comp)
        c0,c1 = min(c for _,c in comp), max(c for _,c in comp)
        score = float(np.clip(max(float(grid[r,c]) for r,c in comp)/(2*threshold),0,1)) if threshold>0 else 1.0
        boxes.append({"y_min":r0/PATCH_GRID,"x_min":c0/PATCH_GRID,
                      "y_max":(r1+1)/PATCH_GRID,"x_max":(c1+1)/PATCH_GRID,"score":round(score,3)})
    boxes.sort(key=lambda b: b["score"], reverse=True)
    return boxes[:MAX_BOXES]


def train_classifier(embedder: DinoV2Embedder, good_dir: str, bad_dir: str, c: float = 1.0):
    """Train DINOv2 + logistic regression on labeled images. Returns model dict."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors

    good_dir, bad_dir = Path(good_dir), Path(bad_dir)
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

    good_files = [f for f in good_dir.iterdir() if f.suffix.lower() in exts]
    bad_files  = [f for f in bad_dir.iterdir()  if f.suffix.lower() in exts]
    log.info("Training: %d good + %d bad images", len(good_files), len(bad_files))

    X, y, good_patch_sets = [], [], []
    for f in good_files:
        data = f.read_bytes()
        X.append(embedder.embed(data))
        y.append("good")
        good_patch_sets.append(embedder.embed_patches(data))
    for f in bad_files:
        X.append(embedder.embed(f.read_bytes()))
        y.append("bad")

    X = np.stack(X)
    clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
    clf.fit(X, y)
    log.info("Classifier trained. Classes: %s", clf.classes_)

    # Build patch memory bank for localization
    bank = np.vstack(good_patch_sets).astype(np.float32)
    if bank.shape[0] > BANK_MAX:
        rng = np.random.default_rng(42)
        idx = rng.choice(bank.shape[0], BANK_MAX, replace=False)
        bank = bank[np.sort(idx)]

    # Calibrate patch threshold (LOO)
    patch_threshold = 0.5
    if len(good_patch_sets) >= 2:
        dists = []
        for i, held in enumerate(good_patch_sets):
            rest = np.vstack([p for j,p in enumerate(good_patch_sets) if j!=i])
            nn = NearestNeighbors(n_neighbors=1).fit(rest)
            d, _ = nn.kneighbors(held)
            dists.append(d.ravel())
        patch_threshold = float(np.quantile(np.concatenate(dists), PATCH_QUANT))
        log.info("Patch threshold calibrated: %.4f", patch_threshold)

    nn_bank = NearestNeighbors(n_neighbors=1).fit(bank)
    return {"clf": clf, "nn_bank": nn_bank, "patch_threshold": patch_threshold,
            "n_good": len(good_files), "n_bad": len(bad_files)}


def predict(model: dict, embedder: DinoV2Embedder, image_bytes: bytes):
    """Predict verdict + bounding boxes. Returns (verdict, confidence, boxes, reason)."""
    clf = model["clf"]
    vec = embedder.embed(image_bytes)
    idx_bad = list(clf.classes_).index("bad")
    prob_bad = float(clf.predict_proba(vec.reshape(1,-1))[0][idx_bad])
    threshold = 0.5
    detected = prob_bad >= threshold
    verdict = "ANOMALY" if detected else "GOOD"
    confidence = round(prob_bad if detected else 1.0-prob_bad, 3)
    boxes = []
    if detected:
        pts = embedder.embed_patches(image_bytes)
        d, _ = model["nn_bank"].kneighbors(pts)
        boxes = boxes_from_distances(d.ravel(), model["patch_threshold"])
    reason = f"DINOv2 anomaly probability {prob_bad:.0%}"
    if boxes: reason += f" — {len(boxes)} anomalous region(s) localized"
    return verdict, confidence, boxes, reason

