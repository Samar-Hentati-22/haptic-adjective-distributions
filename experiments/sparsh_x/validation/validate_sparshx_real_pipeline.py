"""
Sparsh-X validation using THEIR exact preprocessing pipeline (not our
approximations): raw .mcap -> scripts/d360/mcap2pickle.py's save_to_pickle
-> tactile_ssl/data/d360_tactile.py's D360SSLDataset -> Sparsh-X encoder.

Same 10 trials as validate_sparshx.py, pairwise cosine similarity on the
resulting embeddings.
"""
import os
import sys
import io
import shutil

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402
# Absolute paths: resolved before the os.chdir(SPARSH_REPO) below, which is kept
# because checkpoints/*.pth and the Hydra configs are loaded relative to it.
SPARSH_REPO = os.path.abspath(paths.sparsh_repo())
RAW_MCAP_DIR = os.path.abspath(paths.raw_mcap_test_dir())
PICKLE_DIR = os.path.abspath(paths.sparshx_pickle_dir())

sys.path.insert(0, SPARSH_REPO)
os.chdir(SPARSH_REPO)

from scripts.d360.mcap_utils import save_to_pickle  # noqa: E402
from tactile_ssl.data.d360_tactile import D360SSLDataset  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

OmegaConf.register_new_resolver("int_multiply", lambda a, b: int(a * b), replace=True)
OmegaConf.register_new_resolver("int_divide", lambda a, b: a // b, replace=True)

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


def seq_name(obj_id, action, trial):
    return f"{obj_id}_{action}_{trial}"


def convert_all_to_pickle():
    for obj_id, action, trial in TEST_TRIALS:
        name = seq_name(obj_id, action, trial)
        out_dir = os.path.join(PICKLE_DIR, name)
        mcap_dir = os.path.join(RAW_MCAP_DIR, obj_id, action, trial)
        mcap_file = os.path.join(mcap_dir, f"{trial}_0.mcap")  # only _0, matching their mcap2pickle.py exactly
        os.makedirs(out_dir, exist_ok=True)

        if os.path.isfile(os.path.join(out_dir, 'data.pickle')):
            print(f"  [skip, already converted] {name}")
        else:
            print(f"  Converting {name} ...")
            save_to_pickle(
                file=mcap_file, file_out=out_dir,
                image=True, audio=True, imu=True, pressure=True,
                allegro=None, franka=None, extra_topics=None,
            )

        meta_src = os.path.join(mcap_dir, 'metadata.yaml')
        meta_dst = os.path.join(out_dir, 'metadata.yaml')
        if os.path.isfile(meta_src) and not os.path.isfile(meta_dst):
            shutil.copy(meta_src, meta_dst)


# ── Missing calibration substitute (approved 2026-07-20) ────────────────
# load_image() requires background/{device}.txt: a crop rectangle isolating
# the D360 sensor's active area within the raw 1280x720 frame, normally
# calibrated once from a dedicated image on their physical rig -- we don't
# have that. Substitute: detect an equivalent rectangle ourselves using the
# same circular active-region detector already validated for pipeline_v3
# (data_prep/image_crop.py's detect_circle, reimplemented
# inline here), run on one representative raw frame. This is a proxy for
# their calibration, not identical to it -- flagged explicitly, not hidden.
PADDING_FRAC = 0.08
MIN_CONTOUR_AREA_FRAC = 0.05


def _detect_circle_rect(gray_u8):
    _, mask = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas)
    kept = [c for c, a in zip(contours, areas) if a >= MIN_CONTOUR_AREA_FRAC * max_area]
    pts = np.vstack(kept)
    (cx, cy), r = cv2.minEnclosingCircle(pts)
    h, w = gray_u8.shape
    r_pad = r * (1.0 + PADDING_FRAC)
    y0, y1 = max(0, int(round(cy - r_pad))), min(h, int(round(cy + r_pad)))
    x0, x1 = max(0, int(round(cx - r_pad))), min(w, int(round(cx + r_pad)))
    return y0, y1, x0, x1


def write_background_calibration():
    """Detects one crop rect from obj_01's first-converted sequence and
    writes it as background/d360_0.txt into every sequence folder -- one
    calibration per physical device (not per trial), matching what the
    real background/ file represents."""
    ref_seq = seq_name(*TEST_TRIALS[0])
    d = pd.read_pickle(os.path.join(PICKLE_DIR, ref_seq, 'data.pickle'))
    frames = d['d360_0']['image_raw/compressed']['data']
    img_bytes = frames[len(frames) // 2]
    img = np.array(Image.open(io.BytesIO(img_bytes)))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rect = _detect_circle_rect(gray)
    print(f"  Detected crop rect (y0,y1,x0,x1) = {rect} from {ref_seq}'s raw frame")

    for obj_id, action, trial in TEST_TRIALS:
        bg_dir = os.path.join(PICKLE_DIR, seq_name(obj_id, action, trial), 'background')
        os.makedirs(bg_dir, exist_ok=True)
        with open(os.path.join(bg_dir, 'd360_0.txt'), 'w') as f:
            f.write(' '.join(str(v) for v in rect) + '\n')


def build_dataset_node():
    with initialize_config_dir(version_base=None, config_dir=os.path.join(SPARSH_REPO, 'config'),
                                job_name='validate_real'):
        # hydra already nests config/data/d360.yaml's content under a "data"
        # key automatically (package inferred from its directory) -- raw.data
        # is the real root, no manual re-wrapping needed.
        full = compose(config_name='data/d360.yaml')

    # data/d360.yaml references ${img_size}/${mic_size}/${imu_size}/
    # ${pressure_size} as top-level interpolations that normally come from
    # the full experiment config (e.g. config/experiment/d360/dino.yaml),
    # which we're not composing here. Inject the same values used by
    # config/encoder/digit360_sparshx.yaml so the two stay consistent.
    OmegaConf.set_struct(full, False)
    full.img_size = [224, 224]
    full.mic_size = [224, 256]
    full.imu_size = 224
    full.pressure_size = 224

    full.data.sensor.d360.dataset.config.data_path = PICKLE_DIR

    # COMPATIBILITY PATCH (approved 2026-07-20): our sensor round-robins all
    # 4 IMU types over one shared channel, so isolating "acc" (type 6) alone
    # leaves it at ~109.5Hz effective (raw ~438Hz / 4), not the ~400Hz
    # preprocess_imu's defaults assume -- see the patch in
    # tactile_ssl/data/d360/utils.py's preprocess_imu. Override its
    # sample_rate/min_samples via the func partial's config (min_samples
    # scaled proportionally: 32 * 109.5/400 =~ 8.8 -> 8).
    imu_acc_cfg = full.data.sensor.d360.dataset.config.data.device.imu_acc
    imu_acc_cfg.load.func.sample_rate = 109.5
    imu_acc_cfg.load.func.min_samples = 8

    # Same root cause, same category of fix, different modality: our
    # pressure_topic's actual measured rate is ~60Hz (median gap ~16.6ms),
    # not the ~200Hz preprocess_pressure's defaults assume -- see the patch
    # in tactile_ssl/data/d360/utils.py's preprocess_pressure. min_samples
    # scaled proportionally: 25 * 60/200 = 7.5 -> 8. NOTE: the device-level
    # data.device.pressure.sample_rate=200 (used only for the high-pass
    # filter's cutoff-to-Nyquist ratio, not window length) is left
    # unchanged -- out of scope for this fix, flagged as a known remaining
    # inconsistency rather than silently corrected too.
    pressure_cfg = full.data.sensor.d360.dataset.config.data.device.pressure
    pressure_cfg.load.func.sample_rate = 60.0
    pressure_cfg.load.func.min_samples = 8

    OmegaConf.resolve(full)
    return full.data.sensor.d360.dataset  # NOT .config -- keep the _target_ node itself


def main():
    print("Step 1: mcap -> pickle")
    os.makedirs(PICKLE_DIR, exist_ok=True)
    convert_all_to_pickle()

    print("\nStep 1b: substitute background/ crop-rect calibration (see comment above "
          "write_background_calibration)")
    write_background_calibration()

    print("\nStep 2: build D360SSLDataset with their exact preprocessing config")
    dataset_node = build_dataset_node()
    sequences = [seq_name(*t) for t in TEST_TRIALS]
    # The dataset config's load/sync/filter/normalize entries are hydra
    # _target_/_partial_ specs (e.g. tactile_ssl.data.d360.utils.load_image) --
    # OmegaConf.resolve() only resolves ${} interpolations, not object
    # instantiation. instantiate() recursively turns those into real
    # partial callables (matching how their own training scripts build this
    # config), with sequences passed as an override kwarg.
    dataset = instantiate(dataset_node, sequences=sequences)
    print(f"  Dataset built, {len(dataset)} total windows across {len(sequences)} sequences")

    print("\nStep 3: build Sparsh-X encoder")
    with initialize_config_dir(version_base=None, config_dir=os.path.join(SPARSH_REPO, 'config'),
                                job_name='validate_real_encoder'):
        enc_cfg = compose(config_name='encoder/digit360_sparshx.yaml', overrides=['model_size=base'])
        OmegaConf.resolve(enc_cfg)
    model = instantiate(enc_cfg.encoder, _recursive_=True)
    ckpt_path = 'checkpoints/d360_sparshx_img_mic_imu_pressure_base.pth'
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print("  Model loaded OK")

    print("\nStep 4: one sample per sequence -> forward pass")
    # index by sequence: take the first window belonging to each of our 10 sequences
    seq_of_idx = [dataset.data_idxs[i][0][0] for i in range(len(dataset))]  # seg_idx per sample
    first_idx_per_seq = {}
    for i, seg_idx in enumerate(seq_of_idx):
        if seg_idx not in first_idx_per_seq:
            first_idx_per_seq[seg_idx] = i

    embeddings = []
    for seg_idx in range(len(sequences)):
        sample = dataset[first_idx_per_seq[seg_idx]]
        print(f"  {sequences[seg_idx]}: sample keys={list(sample.keys())}, "
              + ", ".join(f"{k}={tuple(v.shape)}" for k, v in sample.items() if hasattr(v, 'shape')))

        batch = {
            'img': torch.as_tensor(sample['img']).unsqueeze(0).float(),
            'mic': torch.as_tensor(sample['mic_fbank']).unsqueeze(0).float(),
            'imu': torch.as_tensor(sample['imu_acc']).unsqueeze(0).float(),
            'pressure': torch.as_tensor(sample['pressure']).unsqueeze(0).float(),
        }
        with torch.inference_mode():
            rep = model(batch)
            emb_parts = [F.layer_norm(v, (v.shape[-1],)) for v in rep.values()]
            emb = torch.cat(emb_parts, dim=1)
        pooled = emb.squeeze(0).mean(dim=0)
        embeddings.append(pooled)

    embeddings = torch.stack(embeddings)
    normed = F.normalize(embeddings, dim=1)
    cos_sim = (normed @ normed.mT).numpy()
    off_diag = cos_sim[~np.eye(len(sequences), dtype=bool)]
    print(f"\nReal-pipeline -- Off-diagonal cosine similarity: min={off_diag.min():.4f} "
          f"max={off_diag.max():.4f} mean={off_diag.mean():.4f} std={off_diag.std():.4f}")


if __name__ == '__main__':
    main()
