"""
Pipeline v3 — standalone image feature extraction (frozen ResNet-50 + GLCM),
following Awan & Jeon 2025's hybrid deep/texture approach.

Self-contained: no imports from outside pipeline_v3/.

Input: cropped tactile images (224, 224, 3) float32 in [0, 1], from the
already-validated image_crop.py / DATA_Final/hdf5_cropped_images/. This
module only extracts features -- calling it from dataset.py/model.py is a
later step.

IMPORTANT normalization note: the ResNet branch standardizes with
ImageNet mean/std ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]), NOT the
plain [0,1] scaling used for images elsewhere in this pipeline (e.g. the
crop/resize output itself, or how other modalities are treated before
dataset.py's later per-channel z-score). This is a deliberate, different
convention specific to feeding a network pretrained on ImageNet-normalized
inputs -- flagging it explicitly so it doesn't get silently "fixed" to
match the rest of the pipeline later.

ResNet-50 and GLCM features end up on very different numeric scales (see
the module's test script) -- also worth carrying forward into any
downstream normalization decisions in dataset.py.
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from skimage.color import rgb2gray
from skimage.feature import graycomatrix

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESNET_FEATURE_DIM = 2048

# GLCM config, following Awan & Jeon's simplest/baseline setup: a single
# distance and angle (distance=1, angle=0 -- i.e. immediate horizontal
# neighbor), not the multi-distance/multi-angle co-occurrence sets some
# GLCM texture pipelines use. If their paper specifies more than one
# distance/angle combined (e.g. averaged over 0/45/90/135 degrees), this
# is a simplification of that -- note before treating this as literally
# equivalent to their reported results.
GLCM_GRAY_LEVELS = 16
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0]  # radians; 0 = horizontal
GLCM_FEATURE_DIM = GLCM_GRAY_LEVELS * GLCM_GRAY_LEVELS  # 256

_resnet_model = None
_resnet_device = None


def _get_resnet(device: str = 'cpu') -> nn.Module:
    """Lazily build and cache the frozen ResNet-50 feature extractor
    (module-level singleton -- loading ImageNet weights is expensive, do
    it once, not per call)."""
    global _resnet_model, _resnet_device
    if _resnet_model is None or _resnet_device != device:
        model = models.resnet50(weights='IMAGENET1K_V2')
        model.fc = nn.Identity()
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(device)
        _resnet_model = model
        _resnet_device = device
    return _resnet_model


def extract_resnet_features(image_batch: np.ndarray, device: str = 'cpu') -> np.ndarray:
    """
    image_batch: (B, 224, 224, 3) float array in [0, 1].

    Standardizes with ImageNet mean/std (NOT the [0,1] convention used
    elsewhere -- see module docstring), runs the frozen ResNet-50 (fc
    replaced with Identity, so this returns the 2048-dim pooled features
    directly), in eval mode under no_grad (frozen, inference only).

    Returns (B, 2048) float32.
    """
    model = _get_resnet(device)

    x = np.asarray(image_batch, dtype=np.float32)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 1, 3)
    x = (x - mean) / std                      # (B, 224, 224, 3), ImageNet-normalized
    x = np.transpose(x, (0, 3, 1, 2))          # (B, 3, 224, 224)

    tensor = torch.from_numpy(np.ascontiguousarray(x)).to(device)
    with torch.no_grad():
        feats = model(tensor)

    return feats.cpu().numpy().astype(np.float32)


def extract_glcm_features(image: np.ndarray) -> np.ndarray:
    """
    image: (224, 224, 3) float array in [0, 1], single image.

    Grayscale (skimage.color.rgb2gray, standard luminance weighting) ->
    quantize to 16 gray levels -> GLCM (distance=1, angle=0, symmetric +
    normed so it sums to 1) -> flatten the 16x16 matrix.

    Returns (256,) float32.
    """
    gray = rgb2gray(np.asarray(image, dtype=np.float32))          # (224, 224), [0, 1]
    quantized = np.clip(np.round(gray * (GLCM_GRAY_LEVELS - 1)), 0,
                         GLCM_GRAY_LEVELS - 1).astype(np.uint8)

    glcm = graycomatrix(quantized, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                         levels=GLCM_GRAY_LEVELS, symmetric=True, normed=True)
    return glcm[:, :, 0, 0].flatten().astype(np.float32)


def extract_image_features(image: np.ndarray, device: str = 'cpu') -> np.ndarray:
    """
    image: (224, 224, 3) float array in [0, 1], single image.

    Returns (2304,) float32: 2048 ResNet-50 features concatenated with
    256 GLCM features.
    """
    resnet_feats = extract_resnet_features(image[None, ...], device=device)[0]  # (2048,)
    glcm_feats = extract_glcm_features(image)                                    # (256,)
    return np.concatenate([resnet_feats, glcm_feats]).astype(np.float32)
