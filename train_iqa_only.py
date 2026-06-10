import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("WANDB_MODE", "offline")
sys.path.insert(0, "src")


def require_supported_python() -> None:
    if sys.version_info < (3, 10):
        version = ".".join(map(str, sys.version_info[:3]))
        raise SystemExit(
            f"Python {version} is too old for this project. "
            "Use the palm_iqa environment, for example: conda activate palm_iqa"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train only the IQA model by reusing an existing metadata.csv with pseudo labels."
    )
    parser.add_argument("--source-run", required=True, help="Existing run name under checkpoints/.")
    parser.add_argument("--name", required=True, help="New run name under checkpoints/.")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers. Use 0 on Windows.")
    parser.add_argument("--iqa-epochs", type=int, default=None, help="IQA epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Evaluation batch size.")
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX export.")
    return parser.parse_args()


def copy_metadata_and_pseudo_labels(source_run: str, target_run: str) -> Path:
    source_root = Path("checkpoints") / source_run
    target_root = Path("checkpoints") / target_run
    source_metadata = source_root / "data" / "metadata.csv"
    target_metadata = target_root / "data" / "metadata.csv"

    if not source_metadata.exists():
        raise FileNotFoundError(f"metadata.csv not found: {source_metadata}")

    frame = pd.read_csv(source_metadata)
    if "quality_score" not in frame.columns:
        raise ValueError(f"quality_score not found in {source_metadata}")

    iqa_frame = frame[frame["split"].isin(["train", "val"])]
    if iqa_frame.empty:
        raise ValueError(f"No train/val rows found in {source_metadata}")
    if iqa_frame["quality_score"].isna().any():
        missing = int(iqa_frame["quality_score"].isna().sum())
        raise ValueError(
            f"metadata has {missing} train/val rows without quality_score: {source_metadata}"
        )

    target_metadata.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_metadata, target_metadata)

    source_pseudo = source_root / "pseudo_labels" / "pseudo_labels.csv"
    if source_pseudo.exists():
        target_pseudo = target_root / "pseudo_labels" / "pseudo_labels.csv"
        target_pseudo.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_pseudo, target_pseudo)

    return target_metadata


def main() -> None:
    require_supported_python()
    args = parse_args()

    from pv_iqa.config import Config
    from pv_iqa.train.iqa import train_iqa
    from pv_iqa.utils.common import set_seed
    from pv_iqa.utils.export_onnx import export_onnx

    metadata_path = copy_metadata_and_pseudo_labels(args.source_run, args.name)

    config = Config(
        name=args.name,
        metadata_path=str(metadata_path),
        device=args.device,
        num_workers=args.workers,
    ).resolve()

    if args.iqa_epochs is not None:
        config.iqa_epochs = args.iqa_epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.eval_batch_size is not None:
        config.eval_batch_size = args.eval_batch_size

    set_seed(config.seed)

    print(f"Source run: {args.source_run}", flush=True)
    print(f"Target run: {config.name}", flush=True)
    print(f"Reused metadata: {metadata_path}", flush=True)
    print("Train IQA only", flush=True)

    iqa_ckpt = train_iqa(config)

    if not args.skip_onnx:
        print("Export ONNX", flush=True)
        export_onnx(config, iqa_ckpt)

    print(f"Done: checkpoints/{config.name}", flush=True)
    print(f"IQA checkpoint: checkpoints/{config.name}/iqa/best.pt", flush=True)


if __name__ == "__main__":
    main()
