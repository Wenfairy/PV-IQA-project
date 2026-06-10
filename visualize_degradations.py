import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

os.environ.setdefault("WANDB_MODE", "offline")
sys.path.insert(0, "src")

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize all PV-IQA degradation types.")
    parser.add_argument("--input", required=True, help="Image file or folder. If folder, the first image is used.")
    parser.add_argument("--out", default="degradation_examples.png", help="Output image path.")
    return parser.parse_args()


def pick_image(path: Path) -> Path:
    if path.is_file():
        return path
    images = [p for p in sorted(path.rglob("*")) if p.suffix.lower() in EXTS]
    if not images:
        raise FileNotFoundError(f"No supported image found: {path}")
    return images[0]


def to_display_image(tensor: torch.Tensor):
    image = torch.clamp(tensor.cpu() * IMAGENET_STD + IMAGENET_MEAN, 0.0, 1.0)
    return image.permute(1, 2, 0).numpy()


def main() -> None:
    args = parse_args()

    from pv_iqa.config import Config
    from pv_iqa.utils.degradation import DEFAULT_LEVELS, DEGRADE_GROUPS, apply_degradation
    from pv_iqa.utils.transforms import build_transforms

    image_path = pick_image(Path(args.input))
    config = Config(num_workers=0).resolve()
    transform = build_transforms(image_size=config.image_size, is_train=False)

    image = Image.open(image_path).convert("L")
    if config.grayscale_to_rgb:
        image = image.convert("RGB")
    clean = transform(image).unsqueeze(0)

    degrade_items = [
        (group, degrade_type)
        for group, names in DEGRADE_GROUPS.items()
        for degrade_type in names
    ]

    fig_h = max(8, 2.15 * len(degrade_items))
    fig, axes = plt.subplots(len(degrade_items), 3, figsize=(8.5, fig_h))
    if len(degrade_items) == 1:
        axes = axes.reshape(1, 3)

    for row, (group, degrade_type) in enumerate(degrade_items):
        mild_level, severe_level = DEFAULT_LEVELS[degrade_type]
        mild = apply_degradation(clean, degrade_type, mild_level)
        severe = apply_degradation(clean, degrade_type, severe_level)

        titles = [
            f"{group}\n{degrade_type}\nclean",
            f"mild={mild_level}",
            f"severe={severe_level}",
        ]
        tensors = [clean[0], mild[0], severe[0]]
        for col in range(3):
            axes[row, col].imshow(to_display_image(tensors[col]), cmap="gray")
            axes[row, col].set_title(titles[col], fontsize=8)
            axes[row, col].axis("off")

    fig.suptitle(f"PV-IQA Degradation Examples\nsource: {image_path.name}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"source: {image_path}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
