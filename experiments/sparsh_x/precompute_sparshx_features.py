import io
import os
import shutil
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402
# Absolute paths: resolved before the os.chdir(SPARSH_REPO) below, which is kept
# because checkpoints/*.pth and the Hydra configs are loaded relative to it.
SPARSH_REPO = os.path.abspath(paths.sparsh_repo())
RAW_MCAP_DIR = os.path.abspath(paths.raw_mcap_full_dir())
PICKLE_DIR = os.path.abspath(paths.sparshx_pickle_dir())
FEATURES_CACHE_DIR = os.path.abspath(paths.sparshx_features_dir())

sys.path.insert(0, SPARSH_REPO)
os.chdir(SPARSH_REPO)

from scripts.d360.mcap_utils import save_to_pickle, load_ros_data, get_timestamp  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

OmegaConf.register_new_resolver("int_multiply", lambda a, b: int(a * b), replace=True)
OmegaConf.register_new_resolver("int_divide", lambda a, b: a // b, replace=True)

ALL_OBJECTS = [f"obj_{i:02d}" for i in range(1, 86)]
PROCEDURES = ["squeeze", "hold"]
TRIALS = [f"trial_{i:03d}" for i in range(5)]
TEST_TRIALS = [(obj, proc, trial) for obj in ALL_OBJECTS for proc in PROCEDURES for trial in TRIALS]

PADDING_FRAC = 0.08
MIN_CONTOUR_AREA_FRAC = 0.05

def seq_name(obj_id, action, trial):
    return f"{obj_id}_{action}_{trial}"


MIN_FRAMES_NEEDED = (2 - 1) * 5 + 1  # length=2, stride=5 -> need indices 0 and 5
REQUIRED_TOPICS = ['image_raw/compressed', 'mic_0', 'mic_1', 'imu_raw_topic', 'imu_quat_topic', 'pressure_topic']


def _merge_additional_parts(name, mcap_dir, out_dir):
    """save_to_pickle()/load_ros_data() only read a single mcap file
    (matching mcap2pickle.py's "_0 only" convention, previously approved).
    Some recordings roll over into "_1"/"_2" parts when the first chunk
    hits a size limit -- for short first chunks that leaves too few image
    frames for length=2/stride=5, AND (discovered for obj_08) can leave some
    topics (imu/pressure/mic) with zero messages entirely in part _0, which
    check_topics() then treats as "device missing data" and drops the whole
    device regardless of image content. Fix: merge in later parts generically
    for every topic (via their own save_to_pickle(), unmodified, run against
    each additional part into a scratch dir, then dict-concatenated in) until
    both the frame-count and all-required-topics-present conditions hold."""
    pkl_path = os.path.join(out_dir, 'data.pickle')
    d = pd.read_pickle(pkl_path)
    dev_data = d['d360_0']

    def enough():
        if any(t not in dev_data for t in REQUIRED_TOPICS):
            return False
        return len(dev_data['image_raw/compressed']['data']) >= MIN_FRAMES_NEEDED

    if enough():
        return

    trial_token = os.path.basename(mcap_dir.rstrip('/'))
    part = 1
    while not enough():
        part_file = os.path.join(mcap_dir, f"{trial_token}_{part}.mcap")
        if not os.path.isfile(part_file):
            missing = [t for t in REQUIRED_TOPICS if t not in dev_data]
            print(f"    [warn] {name}: still incomplete (missing topics={missing}, "
                  f"n_img_frames={len(dev_data.get('image_raw/compressed', {}).get('data', []))}), "
                  f"no more mcap parts found (tried {part_file})")
            break

        print(f"    {name}: merging {part_file} ...")
        scratch_dir = os.path.join(out_dir, f'_scratch_part{part}')
        os.makedirs(scratch_dir, exist_ok=True)
        save_to_pickle(
            file=part_file, file_out=scratch_dir,
            image=True, audio=True, imu=True, pressure=True,
            allegro=None, franka=None, extra_topics=None,
        )
        part_pkl = pd.read_pickle(os.path.join(scratch_dir, 'data.pickle'))
        if 'd360_0' in part_pkl:
            for topic, entry in part_pkl['d360_0'].items():
                if topic not in dev_data:
                    dev_data[topic] = entry
                else:
                    for field, values in entry.items():
                        dev_data[topic][field] = dev_data[topic][field] + values
        shutil.rmtree(scratch_dir)
        part += 1

    # re-sort each topic by its own stamp, ascending
    for topic, entry in dev_data.items():
        if 'stamp' not in entry:
            continue
        order = np.argsort(entry['stamp'])
        for field in list(entry.keys()):
            entry[field] = [entry[field][i] for i in order]

    pd.to_pickle(d, pkl_path)

    if 'image_raw/compressed' not in dev_data:
        # Genuinely no image data in any available mcap part (not a
        # split-across-parts issue) -- nothing to rewrite to disk. Leave the
        # pickle as merged (still useful for whatever topics did come
        # through) and let the caller's per-trial error handling skip this
        # trial downstream instead of crashing the whole batch.
        print(f"    [warn] {name}: no image_raw/compressed data in any mcap part -- "
              f"trial has no usable images, skipping img rewrite")
        return

    # extract_image() reads pixels from disk files named by index (their
    # load_image()'s preload=False stores integer indices, not raw bytes,
    # in-memory -- see tactile_ssl/data/d360/utils.py preprocess_image()'s
    # `raw_imgs = ... np.arange(len(raw_data["data"]))` branch), where the
    # index must match position in the FINAL (merged, re-sorted) data list.
    # Rewrite the on-disk img/ dir so indices 0..n-1 match that order.
    img_dir = os.path.join(out_dir, 'd360_0', 'img')
    os.makedirs(img_dir, exist_ok=True)
    for i, img_bytes in enumerate(dev_data['image_raw/compressed']['data']):
        with open(os.path.join(img_dir, str(i)), 'wb') as f:
            f.write(img_bytes)

    n_frames = len(dev_data['image_raw/compressed']['data'])
    print(f"    {name}: merged to {n_frames} total image frames, "
          f"topics present={sorted(dev_data.keys())}, rewrote {n_frames} img files on disk")


def convert_all_to_pickle():
    failed = []
    for obj_id, action, trial in TEST_TRIALS:
        name = seq_name(obj_id, action, trial)
        out_dir = os.path.join(PICKLE_DIR, name)
        mcap_dir = os.path.join(RAW_MCAP_DIR, obj_id, action, trial)
        mcap_file = os.path.join(mcap_dir, f"{trial}_0.mcap")
        os.makedirs(out_dir, exist_ok=True)

        try:
            if os.path.isfile(os.path.join(out_dir, 'data.pickle')):
                print(f"  [skip, already converted] {name}")
            else:
                print(f"  Converting {name} ...")
                save_to_pickle(
                    file=mcap_file, file_out=out_dir,
                    image=True, audio=True, imu=True, pressure=True,
                    allegro=None, franka=None, extra_topics=None,
                )

            _merge_additional_parts(name, mcap_dir, out_dir)

            meta_src = os.path.join(mcap_dir, 'metadata.yaml')
            meta_dst = os.path.join(out_dir, 'metadata.yaml')
            if os.path.isfile(meta_src) and not os.path.isfile(meta_dst):
                shutil.copy(meta_src, meta_dst)
        except Exception as e:
            # One bad trial (e.g. a genuinely missing modality) shouldn't
            # halt conversion for the other ~170 trials -- same resilience
            # pattern as Step 4's per-trial try/except.
            print(f"  [FAILED, conversion] {name}: {e}")
            failed.append(name)
    return failed


def _detect_circle_rect_square(gray_u8):
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
    side = int(2 * r_pad)  # force square, extract_image()/get_img() assert h==w
    y0 = max(0, int(round(cy - side / 2)))
    x0 = max(0, int(round(cx - side / 2)))
    y1, x1 = min(h, y0 + side), min(w, x0 + side)
    side = min(y1 - y0, x1 - x0)  # re-clip to smallest, keep square after bounds clipping
    y1, x1 = y0 + side, x0 + side
    return y0, y1, x0, x1


def write_background_calibration_per_sequence():
    failed = []
    for obj_id, action, trial in TEST_TRIALS:
        name = seq_name(obj_id, action, trial)
        try:
            pkl_path = os.path.join(PICKLE_DIR, name, 'data.pickle')
            if not os.path.isfile(pkl_path):
                raise FileNotFoundError(f"{pkl_path} missing (conversion failed for this trial)")
            d = pd.read_pickle(pkl_path)
            frames = d['d360_0']['image_raw/compressed']['data']
            img_bytes = frames[len(frames) // 2]
            img = np.array(Image.open(io.BytesIO(img_bytes)))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            rect = _detect_circle_rect_square(gray)
            assert rect[1] - rect[0] == rect[3] - rect[2], f"{name}: rect not square {rect}"

            bg_dir = os.path.join(PICKLE_DIR, name, 'background')
            os.makedirs(bg_dir, exist_ok=True)
            with open(os.path.join(bg_dir, 'd360_0.txt'), 'w') as f:
                f.write(' '.join(str(v) for v in rect) + '\n')
            print(f"  {name}: rect (y0,y1,x0,x1)={rect}")
        except Exception as e:
            print(f"  [FAILED, calibration] {name}: {e}")
            failed.append(name)
    return failed


def build_dataset_node_img_only():
    with initialize_config_dir(version_base=None, config_dir=os.path.join(SPARSH_REPO, 'config'),
                                job_name='validate_squeeze'):
        full = compose(config_name='data/d360.yaml')

    OmegaConf.set_struct(full, False)
    full.img_size = [224, 224]
    full.mic_size = [224, 256]
    full.imu_size = 224
    full.pressure_size = 224
    full.data.sensor.d360.dataset.config.data_path = PICKLE_DIR

    cfg_device = full.data.sensor.d360.dataset.config.data.device
    for modality in ("mic_wave", "imu_acc", "pressure"):
        if "load" in cfg_device[modality]:
            del cfg_device[modality]["load"]
        if "sync" in cfg_device[modality]:
            del cfg_device[modality]["sync"]
    full.data.sensor.d360.dataset.config.msg.device = ["img"]
    full.data.sensor.d360.dataset.config.msg.label = []

    img_cfg = full.data.sensor.d360.dataset.config.data.device.img
    print(f"  img device config: length={img_cfg.length}, stride={img_cfg.stride}, "
          f"sample_rate={img_cfg.sample_rate}")

    OmegaConf.resolve(full)
    return full.data.sensor.d360.dataset


def extract_two_frames(obj_id, action, trial):
    name = seq_name(obj_id, action, trial)
    dataset_node = build_dataset_node_img_only()
    dataset = instantiate(dataset_node, sequences=[name])
    sample = dataset[0]
    img = sample['img']  # (2, 3, 224, 224), their real get_img()/extract_image() output
    assert img.shape == (2, 3, 224, 224), f"{name}: unexpected shape {img.shape}"
    return img

def main():
    print("Step 1: mcap -> pickle")
    os.makedirs(PICKLE_DIR, exist_ok=True)
    os.makedirs(FEATURES_CACHE_DIR, exist_ok=True)  # CHANGED
    failed_conversion = convert_all_to_pickle()
    if failed_conversion:
        print(f"  {len(failed_conversion)} trial(s) failed conversion: {failed_conversion}")

    print("\nStep 2: per-sequence background/crop-rect calibration (squared)")
    failed_calibration = write_background_calibration_per_sequence()
    if failed_calibration:
        print(f"  {len(failed_calibration)} trial(s) failed calibration: {failed_calibration}")

    print("\nStep 3: build Sparsh-X img-only encoder")
    with initialize_config_dir(version_base=None, config_dir=os.path.join(SPARSH_REPO, 'config'),
                                job_name='precompute_all'):
        cfg = compose(config_name='encoder/digit360_sparshx.yaml',
                       overrides=['model_size=base', 'encoder.use_mic=false',
                                  'encoder.use_imu=false', 'encoder.use_pressure=false',
                                  'fusion_type=vanilla'])
        OmegaConf.resolve(cfg)
    model = instantiate(cfg.encoder, _recursive_=True)
    sd = torch.load('checkpoints/d360_sparshx_img_base.pth', map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    model.eval()
    print("  Model loaded OK")

    print("\nStep 4: extract embeddings for all trials, save to disk")  # CHANGED
    n_success, n_failed = 0, 0  # CHANGED
    for obj_id, action, trial in TEST_TRIALS:
        out_path = os.path.join(FEATURES_CACHE_DIR, f"{obj_id}_{action}_{trial}.npy")  # CHANGED
        if os.path.isfile(out_path):  # CHANGED: skip already-done trials
            print(f"  {obj_id}/{action}/{trial}: already cached, skipping")
            n_success += 1
            continue
        try:  # CHANGED: don't crash the whole run on one bad trial
            img = extract_two_frames(obj_id, action, trial)
            img6 = torch.as_tensor(img).reshape(6, 224, 224).unsqueeze(0).float()
            with torch.inference_mode():
                rep = model({'img': img6})
                emb = F.layer_norm(rep['img'], (rep['img'].shape[-1],))
            emb_vec = emb.squeeze(0).mean(dim=0).numpy()  # CHANGED: numpy, ready to save
            np.save(out_path, emb_vec)  # CHANGED: save instead of printing similarity
            print(f"  {obj_id}/{action}/{trial}: saved, shape={emb_vec.shape}")
            n_success += 1
        except Exception as e:  # CHANGED
            print(f"  [FAILED] {obj_id}/{action}/{trial}: {e}")
            n_failed += 1

    print(f"\nDone. Success: {n_success}  Failed: {n_failed}")  # CHANGED


if __name__ == '__main__':
    main()