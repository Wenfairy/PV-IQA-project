import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from PIL import Image

os.environ.setdefault("WANDB_MODE", "offline")
sys.path.insert(0, "src")

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def require_supported_python() -> None:
    if sys.version_info < (3, 10):
        version = ".".join(map(str, sys.version_info[:3]))
        raise SystemExit(
            f"Python {version} is too old for this project. "
            "Use the palm_iqa environment, for example: conda activate palm_iqa"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate degradation ranking accuracy by degradation group/type."
    )
    parser.add_argument("--ckpt", required=True, help="Path to IQA .pt checkpoint.")
    parser.add_argument("--input", required=True, help="Image file or folder.")
    parser.add_argument("--out", default="degradation_ranking_report.csv", help="Output CSV path.")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu.")
    parser.add_argument("--margin", type=float, default=1.0, help="Margin for margin_acc.")
    parser.add_argument("--limit", type=int, default=0, help="Max images to evaluate. 0 means all.")
    return parser.parse_args()


def iter_images(path: Path, limit: int) -> list[Path]:
    if path.is_file():
        files = [path] if path.suffix.lower() in EXTS else []
    else:
        files = [p for p in sorted(path.rglob("*")) if p.suffix.lower() in EXTS]
    if limit > 0:
        files = files[:limit]
    return files


def load_image_tensor(path: Path, config, transform) -> torch.Tensor:
    img = Image.open(path).convert("L")
    if config.grayscale_to_rgb:
        img = img.convert("RGB")
    return transform(img)


def score_batch(model, device: torch.device, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(images.to(device)).detach().cpu()


def main() -> None:
    require_supported_python()
    args = parse_args()

    from pv_iqa.config import Config
    from pv_iqa.eval import load_checkpoint
    from pv_iqa.utils.degradation import DEFAULT_LEVELS, DEGRADE_GROUPS, apply_degradation
    from pv_iqa.utils.transforms import build_transforms

    config = Config(device=args.device, num_workers=0).resolve()
    model, device = load_checkpoint(config, args.ckpt)
    transform = build_transforms(image_size=config.image_size, is_train=False)

    image_paths = iter_images(Path(args.input), args.limit)
    if not image_paths:
        raise SystemExit(f"No supported images found: {args.input}")

    rows = []
    group_totals: dict[str, dict[str, float]] = {}

    for group, degrade_types in DEGRADE_GROUPS.items():
        for degrade_type in degrade_types:
            mild_level, severe_level = DEFAULT_LEVELS[degrade_type]
            n = 0
            rank_ok = 0
            margin_ok = 0
            monotonic_ok = 0
            gap_sum = 0.0
            clean_mild_gap_sum = 0.0

            for image_path in image_paths:
                clean = load_image_tensor(image_path, config, transform).unsqueeze(0)
                mild = apply_degradation(clean, degrade_type, mild_level)
                severe = apply_degradation(clean, degrade_type, severe_level)
                batch = torch.cat([clean, mild, severe], dim=0)
                scores = score_batch(model, device, batch)
                clean_score = float(scores[0])
                mild_score = float(scores[1])
                severe_score = float(scores[2])

                gap = mild_score - severe_score
                clean_mild_gap = clean_score - mild_score
                n += 1
                rank_ok += int(mild_score > severe_score)
                margin_ok += int(gap > args.margin)
                monotonic_ok += int(clean_score > mild_score > severe_score)
                gap_sum += gap
                clean_mild_gap_sum += clean_mild_gap

            row = {
                "group": group,
                "degrade_type": degrade_type,
                "n": n,
                "rank_acc_mild_gt_severe": rank_ok / n if n else 0.0,
                "margin_acc": margin_ok / n if n else 0.0,
                "monotonic_acc_clean_gt_mild_gt_severe": monotonic_ok / n if n else 0.0,
                "mean_gap_mild_minus_severe": gap_sum / n if n else 0.0,
                "mean_gap_clean_minus_mild": clean_mild_gap_sum / n if n else 0.0,
                "mild_level": mild_level,
                "severe_level": severe_level,
            }
            rows.append(row)

            totals = group_totals.setdefault(
                group,
                {
                    "n": 0,
                    "rank_ok": 0,
                    "margin_ok": 0,
                    "monotonic_ok": 0,
                    "gap_sum": 0.0,
                    "clean_mild_gap_sum": 0.0,
                },
            )
            totals["n"] += n
            totals["rank_ok"] += rank_ok
            totals["margin_ok"] += margin_ok
            totals["monotonic_ok"] += monotonic_ok
            totals["gap_sum"] += gap_sum
            totals["clean_mild_gap_sum"] += clean_mild_gap_sum

    for group, totals in group_totals.items():
        n = int(totals["n"])
        rows.append(
            {
                "group": group,
                "degrade_type": "__group_average__",
                "n": n,
                "rank_acc_mild_gt_severe": totals["rank_ok"] / n if n else 0.0,
                "margin_acc": totals["margin_ok"] / n if n else 0.0,
                "monotonic_acc_clean_gt_mild_gt_severe": totals["monotonic_ok"] / n if n else 0.0,
                "mean_gap_mild_minus_severe": totals["gap_sum"] / n if n else 0.0,
                "mean_gap_clean_minus_mild": totals["clean_mild_gap_sum"] / n if n else 0.0,
                "mild_level": "",
                "severe_level": "",
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "degrade_type",
        "n",
        "rank_acc_mild_gt_severe",
        "margin_acc",
        "monotonic_acc_clean_gt_mild_gt_severe",
        "mean_gap_mild_minus_severe",
        "mean_gap_clean_minus_mild",
        "mild_level",
        "severe_level",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"evaluated {len(image_paths)} images")
    print(f"saved report: {out}")
    for row in rows:
        if row["degrade_type"] == "__group_average__":
            print(
                f'{row["group"]}: rank_acc={row["rank_acc_mild_gt_severe"]:.3f}, '
                f'monotonic={row["monotonic_acc_clean_gt_mild_gt_severe"]:.3f}, '
                f'gap={row["mean_gap_mild_minus_severe"]:.3f}'
            )


if __name__ == "__main__":
    main()
