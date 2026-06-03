import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
sys.path.insert(0, "src")

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def require_supported_python() -> None:
    if sys.version_info < (3, 10):
        version = ".".join(map(str, sys.version_info[:3]))
        raise SystemExit(
            f"Python {version} is too old for this project. "
            "Use the palm_iqa environment, for example: "
            "conda activate palm_iqa"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score images with a PV-IQA checkpoint.")
    parser.add_argument("--ckpt", required=True, help="Path to IQA .pt checkpoint.")
    parser.add_argument("--input", required=True, help="Image file or folder.")
    parser.add_argument("--out", default="scores.csv", help="Output CSV path.")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu.")
    parser.add_argument("--rename", action="store_true", help="Rename images to score.ext.")
    parser.add_argument("--digits", type=int, default=3, help="Digits after decimal for renamed files.")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "quality_score"])
        writer.writeheader()
        writer.writerows(rows)


def rename_by_score(rows: list[dict], digits: int) -> None:
    used: set[tuple[str, str]] = set()
    renames = []

    for idx, row in enumerate(rows, start=1):
        src = Path(row["image_path"])
        base = f'{float(row["quality_score"]):.{digits}f}'
        name_base = base
        suffix_idx = 2
        while (name_base.lower(), src.suffix.lower()) in used:
            name_base = f"{base}_{suffix_idx}"
            suffix_idx += 1
        used.add((name_base.lower(), src.suffix.lower()))

        tmp = src.with_name(f"__pv_iqa_tmp_{idx:06d}{src.suffix}")
        final = src.with_name(f"{name_base}{src.suffix}")
        renames.append((src, tmp, final))

    for src, tmp, _ in renames:
        src.rename(tmp)
    for _, tmp, final in renames:
        tmp.rename(final)

    for src, _, final in renames:
        print(f"{src.name} -> {final.name}")


def main() -> None:
    args = parse_args()
    require_supported_python()

    from pv_iqa.config import Config
    from pv_iqa.eval import score_folder, score_image

    config = Config(device=args.device, num_workers=0).resolve()
    input_path = Path(args.input)
    ckpt = Path(args.ckpt)
    out = Path(args.out)

    if input_path.is_file():
        if input_path.suffix.lower() not in EXTS:
            raise ValueError(f"Unsupported image type: {input_path}")
        rows = [score_image(config, ckpt, input_path)]
    else:
        rows = score_folder(config, ckpt, input_path)

    write_csv(out, rows)
    print(f"scored {len(rows)} images")
    print(f"saved csv: {out}")

    if args.rename:
        rename_by_score(rows, args.digits)


if __name__ == "__main__":
    main()
