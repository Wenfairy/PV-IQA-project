"""Image degradation generator for IQA degradation-ranking supervision.

The project trains the IQA model with pairs of degraded images:

    mild degradation should receive a higher quality score than severe degradation.

Degradations are organized into five groups:

1. Optical/dynamic: defocus blur, motion blur, radial distortion
2. Occlusion: random occlusion
3. Photometric: low light, overexposure
4. Digital: JPEG-like compression, low resolution
5. ROI geometry/localization: shift, crop-off, padding, rotation, scale error

The dataloader returns ImageNet-normalized tensors. Most operations are applied
in image space ([0, 1]) and then converted back to normalized space.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import gaussian_blur, rotate


DEGRADE_GROUPS: dict[str, list[str]] = {
    "optical_dynamic": [
        "defocus_blur",
        "motion_blur",
        "radial_distortion",
    ],
    "occlusion": [
        "occlude",
    ],
    "photometric": [
        "low_light",
        "overexpose",
    ],
    "digital": [
        "jpeg_compression",
        "low_resolution",
    ],
    "roi_geometry": [
        "roi_shift",
        "roi_crop_off",
        "roi_padding",
        "roi_rotate",
        "roi_scale_error",
    ],
}

DEGRADE_TYPES = [name for names in DEGRADE_GROUPS.values() for name in names]

DEFAULT_LEVELS: dict[str, tuple[float, float]] = {
    # Optical / dynamic degradation
    "gaussian_blur": (5, 11),
    "defocus_blur": (5, 11),
    "motion_blur": (5, 11),
    "radial_distortion": (0.08, 0.18),
    # Occlusion degradation
    "occlude": (0.15, 0.30),
    # Photometric degradation
    "low_light": (0.60, 0.25),
    "overexpose": (1.50, 3.00),
    # Digital degradation
    "jpeg_compression": (48, 14),
    "low_resolution": (0.50, 0.25),
    # ROI geometry / localization degradation
    "roi_shift": (0.06, 0.16),
    "roi_crop_off": (0.10, 0.25),
    "roi_padding": (0.08, 0.18),
    "roi_rotate": (8, 24),
    "roi_scale_error": (0.10, 0.25),
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _channel_stats(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    channels = images.shape[1]
    mean = torch.tensor(IMAGENET_MEAN[:channels], device=images.device, dtype=images.dtype)
    std = torch.tensor(IMAGENET_STD[:channels], device=images.device, dtype=images.dtype)
    return mean.view(1, channels, 1, 1), std.view(1, channels, 1, 1)


def _to_image_space(images: torch.Tensor) -> torch.Tensor:
    mean, std = _channel_stats(images)
    return torch.clamp(images * std + mean, 0.0, 1.0)


def _to_normalized_space(images: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    mean, std = _channel_stats(reference)
    return (torch.clamp(images, 0.0, 1.0) - mean) / std


def _odd_kernel_size(level: float) -> int:
    size = max(3, int(round(level)))
    return size if size % 2 == 1 else size + 1


def _make_motion_kernel(kernel_size: int, angle_deg: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    center = kernel_size // 2
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    kernel = torch.zeros(kernel_size, kernel_size, device=device, dtype=dtype)
    for y_idx in range(kernel_size):
        y = y_idx - center
        for x_idx in range(kernel_size):
            x = x_idx - center
            if abs(x * cos_a + y * sin_a) <= 0.5:
                kernel[y_idx, x_idx] = 1.0
    denom = kernel.sum().clamp_min(1.0)
    return kernel / denom


def _apply_motion_blur(images: torch.Tensor, kernel_size: int) -> torch.Tensor:
    batch, channels, _, _ = images.shape
    out = images.clone()
    pad = kernel_size // 2
    for idx in range(batch):
        angle = float(torch.randint(0, 180, (1,), device=images.device).item())
        kernel = _make_motion_kernel(kernel_size, angle, images.device, images.dtype)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        padded = F.pad(images[idx : idx + 1], [pad, pad, pad, pad], mode="reflect")
        out[idx : idx + 1] = F.conv2d(padded, kernel, groups=channels)
    return out


def _apply_radial_distortion(images: torch.Tensor, strength: float) -> torch.Tensor:
    batch, _, height, width = images.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height, device=images.device, dtype=images.dtype),
        torch.linspace(-1, 1, width, device=images.device, dtype=images.dtype),
        indexing="ij",
    )
    r2 = xx.square() + yy.square()
    factor = 1.0 + float(strength) * r2
    grid = torch.stack((xx * factor, yy * factor), dim=-1).unsqueeze(0)
    grid = grid.repeat(batch, 1, 1, 1)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def _apply_occlusion(images: torch.Tensor, frac: float) -> torch.Tensor:
    batch, _, height, width = images.shape
    out = images.clone()
    occ_h = max(1, int(height * min(float(frac), 0.6)))
    occ_w = max(1, int(width * min(float(frac), 0.6)))
    for idx in range(batch):
        max_y = max(1, height - occ_h + 1)
        max_x = max(1, width - occ_w + 1)
        y0 = int(torch.randint(0, max_y, (1,), device=images.device).item())
        x0 = int(torch.randint(0, max_x, (1,), device=images.device).item())
        out[idx, :, y0 : y0 + occ_h, x0 : x0 + occ_w] = 0.0
    return out


def _apply_low_resolution(images: torch.Tensor, scale: float) -> torch.Tensor:
    _, _, height, width = images.shape
    small_h = max(4, int(height * float(scale)))
    small_w = max(4, int(width * float(scale)))
    small = F.interpolate(images, size=(small_h, small_w), mode="bilinear", align_corners=False)
    return F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False)


def _apply_jpeg_like_compression(images: torch.Tensor, levels: float) -> torch.Tensor:
    quant_levels = max(4, int(round(levels)))
    quantized = torch.round(images * (quant_levels - 1)) / (quant_levels - 1)
    _, _, height, width = images.shape
    block_h = max(4, height // 16)
    block_w = max(4, width // 16)
    blocky = F.interpolate(
        F.interpolate(quantized, size=(block_h, block_w), mode="nearest"),
        size=(height, width),
        mode="nearest",
    )
    return torch.clamp(0.65 * quantized + 0.35 * blocky, 0.0, 1.0)


def _apply_random_shift(images: torch.Tensor, frac: float) -> torch.Tensor:
    batch, _, height, width = images.shape
    max_dy = max(1, int(height * float(frac)))
    max_dx = max(1, int(width * float(frac)))
    out = images.new_zeros(images.shape)
    for idx in range(batch):
        dy = int(torch.randint(-max_dy, max_dy + 1, (1,), device=images.device).item())
        dx = int(torch.randint(-max_dx, max_dx + 1, (1,), device=images.device).item())
        if dy == 0 and dx == 0:
            dy = max_dy

        src_y0 = max(0, -dy)
        src_y1 = min(height, height - dy)
        dst_y0 = max(0, dy)
        dst_y1 = min(height, height + dy)
        src_x0 = max(0, -dx)
        src_x1 = min(width, width - dx)
        dst_x0 = max(0, dx)
        dst_x1 = min(width, width + dx)

        out[idx, :, dst_y0:dst_y1, dst_x0:dst_x1] = images[idx, :, src_y0:src_y1, src_x0:src_x1]
    return out


def _apply_side_crop_off(images: torch.Tensor, frac: float) -> torch.Tensor:
    batch, _, height, width = images.shape
    out = images.clone()
    crop_h = max(1, int(height * float(frac)))
    crop_w = max(1, int(width * float(frac)))
    for idx in range(batch):
        side = int(torch.randint(0, 4, (1,), device=images.device).item())
        if side == 0:
            out[idx, :, :crop_h, :] = 0.0
        elif side == 1:
            out[idx, :, height - crop_h :, :] = 0.0
        elif side == 2:
            out[idx, :, :, :crop_w] = 0.0
        else:
            out[idx, :, :, width - crop_w :] = 0.0
    return out


def _apply_padding_context(images: torch.Tensor, frac: float) -> torch.Tensor:
    _, _, height, width = images.shape
    scale = max(0.2, 1.0 - 2.0 * float(frac))
    new_h = max(1, int(height * scale))
    new_w = max(1, int(width * scale))
    small = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)
    out = images.new_zeros(images.shape)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    out[:, :, y0 : y0 + new_h, x0 : x0 + new_w] = small
    return out


def _apply_random_rotate(images: torch.Tensor, max_angle: float) -> torch.Tensor:
    out = images.clone()
    for idx in range(images.shape[0]):
        angle = float(torch.empty(1).uniform_(-max_angle, max_angle).item())
        if abs(angle) < 1.0:
            angle = float(max_angle)
        out[idx] = rotate(
            images[idx],
            angle=angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
    return out


def _apply_scale_error(images: torch.Tensor, magnitude: float) -> torch.Tensor:
    _, _, height, width = images.shape
    zoom_in = bool(torch.randint(0, 2, (1,), device=images.device).item())
    if zoom_in:
        scale = 1.0 + float(magnitude)
        new_h = max(height + 1, int(height * scale))
        new_w = max(width + 1, int(width * scale))
        resized = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)
        y0 = (new_h - height) // 2
        x0 = (new_w - width) // 2
        return resized[:, :, y0 : y0 + height, x0 : x0 + width]

    scale = max(0.2, 1.0 - float(magnitude))
    new_h = max(1, int(height * scale))
    new_w = max(1, int(width * scale))
    resized = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)
    out = images.new_zeros(images.shape)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    out[:, :, y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return out


def apply_degradation(
    images: torch.Tensor,
    degrade_type: str,
    level: float,
) -> torch.Tensor:
    img = _to_image_space(images)

    if degrade_type in ("defocus_blur", "gaussian_blur"):
        kernel_size = _odd_kernel_size(level)
        sigma = max(0.1, kernel_size / 3.0)
        degraded = gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)

    elif degrade_type == "motion_blur":
        degraded = _apply_motion_blur(img, _odd_kernel_size(level))

    elif degrade_type == "radial_distortion":
        degraded = _apply_radial_distortion(img, float(level))

    elif degrade_type == "occlude":
        degraded = _apply_occlusion(img, float(level))

    elif degrade_type == "low_light":
        degraded = img * float(level)

    elif degrade_type == "overexpose":
        degraded = img * float(level)

    elif degrade_type == "jpeg_compression":
        degraded = _apply_jpeg_like_compression(img, float(level))

    elif degrade_type == "low_resolution":
        degraded = _apply_low_resolution(img, float(level))

    elif degrade_type == "roi_shift":
        degraded = _apply_random_shift(img, float(level))

    elif degrade_type == "roi_crop_off":
        degraded = _apply_side_crop_off(img, float(level))

    elif degrade_type == "roi_padding":
        degraded = _apply_padding_context(img, float(level))

    elif degrade_type == "roi_rotate":
        degraded = _apply_random_rotate(img, float(level))

    elif degrade_type == "roi_scale_error":
        degraded = _apply_scale_error(img, float(level))

    else:
        raise ValueError(f"Unknown degrade_type: {degrade_type}")

    return _to_normalized_space(degraded, images)


def sample_degradation_type(
    rng: torch.Generator | None = None,
) -> tuple[str, str]:
    groups = list(DEGRADE_GROUPS)
    group_idx = torch.randint(0, len(groups), (1,), generator=rng).item()
    group = groups[int(group_idx)]
    names = DEGRADE_GROUPS[group]
    type_idx = torch.randint(0, len(names), (1,), generator=rng).item()
    return group, names[int(type_idx)]


def generate_ranking_pair(
    images: torch.Tensor,
    rng: torch.Generator | None = None,
    return_info: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    group, degrade_type = sample_degradation_type(rng)
    mild_level, severe_level = DEFAULT_LEVELS[degrade_type]
    mild = apply_degradation(images, degrade_type, mild_level)
    severe = apply_degradation(images, degrade_type, severe_level)

    if return_info:
        return mild, severe, {
            "group": group,
            "degrade_type": degrade_type,
            "mild_level": mild_level,
            "severe_level": severe_level,
        }
    return mild, severe
