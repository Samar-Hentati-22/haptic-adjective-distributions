"""
Extracts the raw 2-frame image sequence for obj_01/tap/trial_000 using
THEIR actual D360SSLDataset.get_img()/extract_image() code path (not our
reimplementation), with their real config length/stride values, and saves
both frames as plain PNGs before any further (circular) cropping or model
forward pass.

Reuses build_dataset_node()/the mcap->pickle conversion + substitute
background calibration already produced by validate_sparshx_real_pipeline.py
for this trial (processed_pickle/obj_01_tap_trial_000/).
"""
import os
import sys

import numpy as np
from PIL import Image
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402
# Absolute paths: resolved before the os.chdir(SPARSH_REPO) below, which is kept
# because checkpoints/*.pth and the Hydra configs are loaded relative to it.
SPARSH_REPO = os.path.abspath(paths.sparsh_repo())
PICKLE_DIR = os.path.abspath(paths.sparshx_pickle_dir())
OUT_DIR = os.path.abspath(paths.output_dir(__file__))

sys.path.insert(0, SPARSH_REPO)
os.chdir(SPARSH_REPO)

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

OmegaConf.register_new_resolver("int_multiply", lambda a, b: int(a * b), replace=True)
OmegaConf.register_new_resolver("int_divide", lambda a, b: a // b, replace=True)

SEQ = "obj_01_tap_trial_000"


def build_dataset_node():
    with initialize_config_dir(version_base=None, config_dir=os.path.join(SPARSH_REPO, 'config'),
                                job_name='verify_raw_frames'):
        full = compose(config_name='data/d360.yaml')

    OmegaConf.set_struct(full, False)
    full.img_size = [224, 224]
    full.mic_size = [224, 256]
    full.imu_size = 224
    full.pressure_size = 224
    full.data.sensor.d360.dataset.config.data_path = PICKLE_DIR

    imu_acc_cfg = full.data.sensor.d360.dataset.config.data.device.imu_acc
    imu_acc_cfg.load.func.sample_rate = 109.5
    imu_acc_cfg.load.func.min_samples = 8

    pressure_cfg = full.data.sensor.d360.dataset.config.data.device.pressure
    pressure_cfg.load.func.sample_rate = 60.0
    pressure_cfg.load.func.min_samples = 8

    # print the literal length/stride config values actually used for img, as requested
    img_cfg = full.data.sensor.d360.dataset.config.data.device.img
    print(f"config/data/d360.yaml -- img device config: length={img_cfg.length}, "
          f"stride={img_cfg.stride}, sample_rate={img_cfg.sample_rate}")

    # This validation is image-only (img-only checkpoint, task scoped to img
    # extraction). D360.sync_time() intersects the time interval across
    # EVERY device_cfg entry that has a "load" key (mic_wave, imu_acc,
    # pressure), regardless of config.msg.device -- so with our real,
    # already-flagged mic/imu/pressure sample-rate approximations, that
    # cross-modal intersection comes back empty for these trials and
    # silently drops all image segments too (confirmed: preprocess_image()
    # called directly on the same raw data returns a valid 57-frame image
    # segment; only the full D360 object's img_time ends up empty).
    # _check_d360_data() only requires the *keys* mic_wave/mic_fbank/imu_acc
    # to exist in data.device, not that they have a "load" step -- so
    # dropping their "load" subkey (real device_cfg keys stay present,
    # satisfying that check) removes them from sync_time's intersection,
    # leaving only img's own interval. This is real code, unmodified; we are
    # just scoping the dataset to the img device_msg being tested, same as
    # the encoder's use_mic=false/use_imu=false/use_pressure=false override.
    cfg_device = full.data.sensor.d360.dataset.config.data.device
    for modality in ("mic_wave", "imu_acc", "pressure"):
        if "load" in cfg_device[modality]:
            del cfg_device[modality]["load"]
        if "sync" in cfg_device[modality]:
            del cfg_device[modality]["sync"]
    full.data.sensor.d360.dataset.config.msg.device = ["img"]
    full.data.sensor.d360.dataset.config.msg.label = []

    OmegaConf.resolve(full)
    return full.data.sensor.d360.dataset


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    dataset_node = build_dataset_node()
    dataset = instantiate(dataset_node, sequences=[SEQ])
    print(f"Dataset built, {len(dataset)} windows for {SEQ}")
    print(f"Confirmed dataset.img_length={dataset.img_length}, dataset.img_stride={dataset.img_stride}")

    # Real windowing now works (img-only scoped dataset) -- use their actual
    # __getitem__ -> get_img() -> extract_image() path unmodified.
    sample = dataset[0]
    img = sample['img']  # (length, C, H, W) via get_img's set_data
    print(f"sample['img'] shape: {img.shape}, dtype: {img.dtype}, "
          f"min/max: {img.min():.4f}/{img.max():.4f}")

    assert img.shape[0] == 2, f"expected length=2 frames, got {img.shape[0]}"

    for i in range(2):
        frame = img[i]  # (C,H,W), float, roughly in [0,1] (pre-model normalization)
        frame_hwc = np.transpose(frame, (1, 2, 0))
        frame_u8 = np.clip(frame_hwc * 255.0, 0, 255).astype(np.uint8)
        out_path = os.path.join(OUT_DIR, f"obj_01_frame{i + 1}.png")
        Image.fromarray(frame_u8).save(out_path)
        print(f"Saved {out_path} (shape {frame_u8.shape})")


if __name__ == '__main__':
    main()
