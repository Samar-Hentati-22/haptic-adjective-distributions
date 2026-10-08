"""
Sanity checks on the Sparsh-X img-only base checkpoint before trusting the
object-collapse result from validate_sparshx_realdiff_frames.py:

  1. Feed 3 obviously-different synthetic 6ch inputs (random noise, solid
     black, solid white) through the same model+preprocessing path used for
     the real trials -- confirm the model itself is capable of producing
     well-separated embeddings when the inputs are trivially different.
  2. Report the exact normalization buffers (img_avg/img_div) the model
     registered from config/encoder/digit360_sparshx.yaml's `normalization`
     block, and confirm our preprocessing (raw pixels scaled to [0,1], no
     manual mean/std subtraction) matches what tactile_ssl expects the model
     to receive (extract_image() in tactile_ssl/data/d360/d360.py:103-120
     produces img/255 when no background frame is available, and
     d360_transformer.py's pre_embed() applies (x - img_avg)/img_div
     internally -- so raw [0,1] pixels in, model normalizes, is correct).
  3. Report checkpoint parameter count and loading diagnostics.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402
# Absolute paths: resolved before the os.chdir(SPARSH_REPO) below, which is kept
# because checkpoints/*.pth and the Hydra configs are loaded relative to it.
SPARSH_REPO = os.path.abspath(paths.sparsh_repo())
sys.path.insert(0, SPARSH_REPO)
os.chdir(SPARSH_REPO)

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def main():
    ckpt_path = 'checkpoints/d360_sparshx_img_base.pth'
    with initialize_config_dir(version_base=None, config_dir=os.path.join(SPARSH_REPO, 'config'),
                                job_name='sanity_check'):
        cfg = compose(config_name='encoder/digit360_sparshx.yaml',
                       overrides=['model_size=base', 'encoder.use_mic=false',
                                  'encoder.use_imu=false', 'encoder.use_pressure=false',
                                  'fusion_type=vanilla'])
        OmegaConf.resolve(cfg)

    print("=" * 70)
    print("PART 2: normalization config as resolved from digit360_sparshx.yaml")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg.normalization.img))

    model = instantiate(cfg.encoder, _recursive_=True)

    print("Registered buffers actually used by the model at inference time:")
    print(f"  img_avg (per-channel, before tiling to {model.sensor_chans['img']} chans):")
    print(f"    {model.img_avg.flatten().tolist()}")
    print(f"  img_div (per-channel):")
    print(f"    {model.img_div.flatten().tolist()}")
    is_identity = torch.allclose(model.img_avg, torch.zeros_like(model.img_avg)) and \
        torch.allclose(model.img_div, torch.ones_like(model.img_div))
    print(f"  Fallback-to-identity (avg=0,div=1, i.e. normalization silently failed to load)? {is_identity}")

    print("\n" + "=" * 70)
    print("PART 3: checkpoint loading diagnostics")
    print("=" * 70)
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    n_ckpt_params = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
    print(f"  Checkpoint file: {ckpt_path}")
    print(f"  Checkpoint tensors: {len(sd)}, total checkpoint parameter count: {n_ckpt_params:,}")

    missing, unexpected = model.load_state_dict(sd, strict=True)
    print(f"  strict=True load_state_dict -> missing={missing}, unexpected={unexpected}")
    n_model_params = sum(p.numel() for p in model.parameters())
    n_model_params_loaded = sum(p.numel() for name, p in model.named_parameters() if name not in missing)
    print(f"  Model total parameter count (post-load): {n_model_params:,}")
    print(f"  Model parameters covered by checkpoint: {n_model_params_loaded:,}")
    model.eval()

    print("\n" + "=" * 70)
    print("PART 1: synthetic sanity inputs (noise / black / white), 6ch, same forward path")
    print("=" * 70)
    torch.manual_seed(0)
    synthetic = {
        'random_noise': torch.rand(1, 6, 224, 224),   # uniform [0,1], matches raw-pixel scale
        'solid_black': torch.zeros(1, 6, 224, 224),
        'solid_white': torch.ones(1, 6, 224, 224),
    }

    embeddings = []
    labels = []
    for name, img in synthetic.items():
        with torch.inference_mode():
            rep = model({'img': img})
            emb = F.layer_norm(rep['img'], (rep['img'].shape[-1],))
        pooled = emb.squeeze(0).mean(dim=0)
        embeddings.append(pooled)
        labels.append(name)
        print(f"  {name}: raw rep shape {tuple(rep['img'].shape)}, pooled norm={pooled.norm():.4f}")

    embeddings = torch.stack(embeddings)
    normed = F.normalize(embeddings, dim=1)
    cos_sim = (normed @ normed.mT).numpy()

    print("\nPairwise cosine similarity (synthetic sanity inputs):")
    print("               " + "  ".join(f"{l:>12s}" for l in labels))
    for i, l in enumerate(labels):
        print(f"  {l:>12s}  " + "  ".join(f"{cos_sim[i, j]:12.4f}" for j in range(len(labels))))

    off_diag = cos_sim[~np.eye(len(labels), dtype=bool)]
    print(f"\nSynthetic off-diagonal cosine similarity: min={off_diag.min():.4f} "
          f"max={off_diag.max():.4f} mean={off_diag.mean():.4f}")


if __name__ == '__main__':
    main()
