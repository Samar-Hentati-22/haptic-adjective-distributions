"""
Sparsh-X img-only checkpoint validation using two GENUINELY DIFFERENT real
frames (at ~25% and ~75% through each trial) stacked as 6 channels, instead
of duplicating a single frame (which previously collapsed embeddings to
~0.97 off-diagonal cosine similarity -- likely because duplicate frames give
no motion signal for a model that reads temporal difference).

Crop logic: reimplementation of data_prep/image_crop.py's
detect_circle + crop_and_resize (Otsu threshold + contour detection +
padded circular crop + resize to 224x224) -- not cross-imported since this
venv (.venv-sparshx) is isolated from the main project .venv.

Frames + pickles: reuses processed_pickle/<seq>/data.pickle produced by
validate_sparshx_real_pipeline.py's mcap->pickle conversion (already run
for these 10 trials).
"""
import io
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402
# Absolute paths: resolved before the os.chdir(SPARSH_REPO) below, which is kept
# because checkpoints/*.pth and the Hydra configs are loaded relative to it.
SPARSH_REPO = os.path.abspath(paths.sparsh_repo())
PICKLE_DIR = os.path.abspath(paths.sparshx_pickle_dir())

sys.path.insert(0, SPARSH_REPO)
os.chdir(SPARSH_REPO)

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

TEST_TRIALS = [
    ('obj_01', 'tap', 'trial_000'),
    ('obj_05', 'squeeze', 'trial_001'),
    ('obj_10', 'slide', 'trial_002'),
    ('obj_15', 'hold', 'trial_000'),
    ('obj_20', 'tap', 'trial_003'),
    ('obj_30', 'squeeze', 'trial_002'),
    ('obj_45', 'slide', 'trial_001'),
    ('obj_60', 'hold', 'trial_004'),
    ('obj_70', 'tap', 'trial_002'),
    ('obj_85', 'squeeze', 'trial_000'),
]

PADDING_FRAC = 0.08
MIN_CONTOUR_AREA_FRAC = 0.05


def detect_circle(gray_u8):
    _, mask = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas)
    kept = [c for c, a in zip(contours, areas) if a >= MIN_CONTOUR_AREA_FRAC * max_area]
    pts = np.vstack(kept)
    (cx, cy), r = cv2.minEnclosingCircle(pts)
    return cx, cy, r


def crop_and_resize(img_float, cx, cy, r, size=224):
    h, w = img_float.shape[:2]
    r_pad = r * (1.0 + PADDING_FRAC)
    x0, x1 = max(0, int(round(cx - r_pad))), min(w, int(round(cx + r_pad)))
    y0, y1 = max(0, int(round(cy - r_pad))), min(h, int(round(cy + r_pad)))
    crop = img_float[y0:y1, x0:x1]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def load_two_real_frames(obj_id, action, trial):
    seq = f"{obj_id}_{action}_{trial}"
    d = pd.read_pickle(os.path.join(PICKLE_DIR, seq, 'data.pickle'))
    frames_bytes = d['d360_0']['image_raw/compressed']['data']
    n = len(frames_bytes)
    STRIDE = 5  # 5 frames apart @ 30fps = 0.17s, matches d360_tactile.py's img_stride
    idx_a = max(0, (n - 1 - STRIDE) // 2)
    idx_b = idx_a + STRIDE
    idx_25, idx_75 = idx_a, idx_b  # reuse existing var names below

    crops = []
    for idx in (idx_25, idx_75):
        img = np.array(Image.open(io.BytesIO(frames_bytes[idx])).convert('RGB')).astype(np.float32) / 255.0
        gray_u8 = (np.clip(img.mean(axis=2), 0, 1) * 255).astype(np.uint8)
        cx, cy, r = detect_circle(gray_u8)
        cropped = crop_and_resize(img, cx, cy, r, size=224)  # (224,224,3) float
        crops.append(cropped)

    img6 = np.concatenate([crops[0], crops[1]], axis=2)  # (224,224,6)
    img_t = torch.from_numpy(np.ascontiguousarray(img6.transpose(2, 0, 1)))  # (6,224,224)
    return img_t.unsqueeze(0).float(), idx_25, idx_75, n


def main():
    ckpt_path = 'checkpoints/d360_sparshx_img_base.pth'
    with initialize_config_dir(version_base=None, config_dir=os.path.join(SPARSH_REPO, 'config'),
                                job_name='validate_realdiff'):
        cfg = compose(config_name='encoder/digit360_sparshx.yaml',
                       overrides=['model_size=base', 'encoder.use_mic=false',
                                  'encoder.use_imu=false', 'encoder.use_pressure=false',
                                  'fusion_type=vanilla'])
        OmegaConf.resolve(cfg)

    model = instantiate(cfg.encoder, _recursive_=True)
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    model.eval()

    embeddings = []
    labels = []
    for obj_id, action, trial in TEST_TRIALS:
        img6, idx_25, idx_75, n = load_two_real_frames(obj_id, action, trial)
        print(f"  {obj_id}/{action}/{trial}: n_frames={n}, using idx {idx_25} (25%) and {idx_75} (75%)")
        with torch.inference_mode():
            rep = model({'img': img6})
            emb = F.layer_norm(rep['img'], (rep['img'].shape[-1],))
        embeddings.append(emb.squeeze(0).mean(dim=0))
        labels.append(f'{obj_id}/{action}/{trial}')

    embeddings = torch.stack(embeddings)
    normed = F.normalize(embeddings, dim=1)
    cos_sim = (normed @ normed.mT).numpy()

    print("\nPairwise cosine similarity matrix:")
    print("      " + " ".join(f"{i:6d}" for i in range(len(labels))))
    for i in range(len(labels)):
        print(f"{i:3d}  " + " ".join(f"{cos_sim[i, j]:6.3f}" for j in range(len(labels))))
    for i, l in enumerate(labels):
        print(f"  [{i}] {l}")

    off_diag = cos_sim[~np.eye(len(labels), dtype=bool)]
    print(f"\nreal-diff-frames img-only ckpt -- Off-diagonal cosine similarity: "
          f"min={off_diag.min():.4f} max={off_diag.max():.4f} "
          f"mean={off_diag.mean():.4f} std={off_diag.std():.4f}")


if __name__ == '__main__':
    main()
