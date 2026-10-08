"""
Pipeline v3 — per-modality encoders + fusion head for TactileDatasetV3's
output (src/dataset.py).

REGULARIZED VARIANT (copy of model.py, 2026-07-20): adds nn.Dropout after
every ReLU in ImageEncoder's 4-layer MLP and in TactilePerceptionModelV3's
fusion head, at a configurable rate (default 0.3), to address the mild
train/test correlation gap observed for the 13 well-performing adjectives
(fold 1: mean train_r=0.908 vs test_r=0.730, gap~0.18 -- see
diagnose_train_test_gap.py). model.py itself is untouched; this is a
separate file for a separate experiment.

Self-contained: no imports from outside pipeline_v3/. The AudioEncoder
architecture is ported (not imported) from Training_Phase2/model.py's
AudioEncoder -- same 3-layer CNN + AdaptiveAvgPool2d pattern for handling
variable-T mel spectrograms.

pressure/ght are excluded, matching dataset.py's scope. image_input_dim
is fold-dependent (dataset.py drops dead ResNet channels per fold, see
compute_norm_stats's RESNET_DEAD_CHANNEL_STD_THRESH) and must be passed
into TactilePerceptionModelV3 explicitly -- read it off
len(norm_stats['image_resnet_alive_dims']) + GLCM_FEATURE_DIM at training
time, don't hardcode it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ADJECTIVES = 18
N_BINS = 7


class IMUEncoder(nn.Module):
    """(B, 10) -> (B, 32)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 32), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class PressureAPEncoder(nn.Module):
    """(B, 224, 50) -> (B, 64). Treated as a single-channel 2D "image"
    (time x channel): unsqueeze a channel dim, 3-layer 2D CNN, then
    AdaptiveAvgPool2d to a fixed 1x1 before the final linear projection."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, 64)

    def forward(self, x):
        x = x.unsqueeze(1)                            # (B, 1, 224, 50)
        feat = self.features(x).view(x.size(0), -1)   # (B, 64)
        return self.proj(feat)


class ImageEncoder(nn.Module):
    """(B, image_input_dim) -> (B, 128). image_input_dim is fold-dependent,
    passed in at construction (see module docstring).

    4-layer MLP (128, 256, 256, 128), matching Awan & Jeon's documented ANN
    baseline branch spec ("two parallel fully connected branches (layer
    sizes: 128, 256, 256, 128)") exactly. REGULARIZED: nn.Dropout(dropout)
    after each ReLU."""

    def __init__(self, image_input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_input_dim, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AudioEncoder(nn.Module):
    """(B, 2, 64, T) -> (B, 64), T variable. Ported from
    Training_Phase2/model.py's AudioEncoder: 3-layer CNN + AdaptiveAvgPool2d
    collapses the variable T (and frequency) axes to 1x1 before the final
    linear projection, so any T is accepted without resizing/padding
    assumptions baked into the architecture itself."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, 64)

    def forward(self, x):
        return self.proj(self.features(x).view(x.size(0), -1))


class TactilePerceptionModelV3(nn.Module):
    """
    Multimodal tactile perception model, pipeline_v3.

    Inputs (batched, matching TactileDatasetV3's __getitem__ keys):
      imu         : (B, 10)
      pressure_ap : (B, 224, 50)
      image       : (B, image_input_dim)  -- fold-dependent width
      audio       : (B, 2, 64, T)         -- T variable, see collate_batch
      has_image   : (B,)
      has_audio   : (B,)
    (pressure/ght excluded, per dataset.py's scope.)

    has_image/has_audio mask their embedding (multiply by 0/1) after
    encoding, same convention as Training_Phase2/model.py -- a dropped-out
    modality contributes nothing to the fused feature, rather than relying
    on the encoder's response to an all-zero input being itself zero
    (not guaranteed, e.g. BatchNorm bias).

    Output: (B, 18, 7), softmax over the last (rating-bin) dimension.

    REGULARIZED: dropout (default 0.3) after each ReLU in ImageEncoder and
    in the fusion head below, to address the mild train/test correlation
    gap seen for the 13 well-performing adjectives (see module docstring).
    """
    IMU_DIM = 32
    PRESSURE_AP_DIM = 64
    IMAGE_DIM = 128
    AUDIO_DIM = 64
    FUSED_DIM = IMU_DIM + PRESSURE_AP_DIM + IMAGE_DIM + AUDIO_DIM  # 288

    def __init__(self, image_input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.imu_enc = IMUEncoder()
        self.pressure_ap_enc = PressureAPEncoder()
        self.image_enc = ImageEncoder(image_input_dim, dropout=dropout)
        self.audio_enc = AudioEncoder()

        self.head = nn.Sequential(
            nn.Linear(self.FUSED_DIM, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, N_ADJECTIVES * N_BINS),
        )

    def forward(self, imu, pressure_ap, image, audio, has_image, has_audio):
        imu_emb = self.imu_enc(imu)                                 # (B, 32)
        pap_emb = self.pressure_ap_enc(pressure_ap)                 # (B, 64)
        img_emb = self.image_enc(image) * has_image.view(-1, 1)     # (B, 128)
        aud_emb = self.audio_enc(audio) * has_audio.view(-1, 1)     # (B, 64)

        fused = torch.cat([imu_emb, pap_emb, img_emb, aud_emb], dim=1)  # (B, 288)
        out = self.head(fused).view(-1, N_ADJECTIVES, N_BINS)
        return F.softmax(out, dim=2)


def collate_batch(samples):
    """
    Custom collate for a list of TactileDatasetV3 __getitem__ dicts.
    Needed because 'audio' has a variable time dimension T per trial
    (dataset.py deliberately doesn't pad it, see its module docstring) and
    can't be torch.stack'ed as-is -- zero-pads audio to the batch's max T
    (padding is safe: it's appended past each trial's real frames, and the
    AudioEncoder's AdaptiveAvgPool2d treats it like any other low-energy
    region). Every other field is already fixed-shape per fold and stacks
    directly.
    """
    out = {}
    for key in ('imu', 'pressure_ap', 'image', 'has_image', 'has_audio',
                'label_dist', 'label_mean', 'obj_idx'):
        out[key] = torch.stack([s[key] for s in samples])

    max_t = max(s['audio'].shape[-1] for s in samples)
    audio = torch.zeros(len(samples), 2, 64, max_t, dtype=samples[0]['audio'].dtype)
    for i, s in enumerate(samples):
        t = s['audio'].shape[-1]
        audio[i, :, :, :t] = s['audio']
    out['audio'] = audio

    return out
