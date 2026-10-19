"""Dense per-patch DINOv3 features of each section, for searching a region in feature space."""

import os

import numpy as np
import timm
import torch
import torch.nn.functional as F

DEFAULT_MODEL = "vit_small_patch16_dinov3.lvd1689m"
PATCH = 16
# At 1x the hippocampus box (50x44 px) spans only ~3x3 patches, too few to carry its shape.
UPSCALE = 2
BACKGROUND = 255


def pick_device() -> torch.device:
    # REGION_SEARCH_DEVICE=cpu: on 2026-10-18 MPS command-buffer errors went to stderr only
    # and the search went on with corrupt scores (1.00 on blank sections); CPU is the check.
    if forced := os.environ.get("REGION_SEARCH_DEVICE"):
        return torch.device(forced)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(name: str, device: torch.device) -> torch.nn.Module:
    model = timm.create_model(name, pretrained=True, num_classes=0, dynamic_img_size=True)
    return model.eval().to(device)


def prepare(image: np.ndarray, mean: tuple, std: tuple) -> torch.Tensor:
    """Grayscale uint8 section -> normalised 1x3xHxW tensor.

    Upscaled by UPSCALE, then padded with background on the right and bottom up to a
    multiple of PATCH, so section pixel (x, y) lands in patch (x * UPSCALE // PATCH, ...).
    """
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(f"expected a 2D uint8 section, got {image.dtype} {image.shape}")
    tensor = torch.from_numpy(image).float()[None, None] / 255.0
    tensor = F.interpolate(tensor, scale_factor=UPSCALE, mode="bilinear", align_corners=False)
    height, width = tensor.shape[-2:]
    pad_h, pad_w = -height % PATCH, -width % PATCH
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=BACKGROUND / 255.0)
    tensor = tensor.expand(1, 3, -1, -1)
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    return (tensor - mean_t) / std_t


@torch.inference_mode()
def patch_features(model: torch.nn.Module, image: np.ndarray, device: torch.device) -> np.ndarray:
    """rows x cols x dim float16 grid of L2-normalised patch features."""
    config = model.pretrained_cfg
    batch = prepare(image, config["mean"], config["std"]).to(device)
    tokens = model.forward_features(batch)[0, model.num_prefix_tokens :]
    rows, cols = batch.shape[-2] // PATCH, batch.shape[-1] // PATCH
    tokens = F.normalize(tokens, dim=-1)
    return tokens.reshape(rows, cols, -1).to(torch.float16).cpu().numpy()
