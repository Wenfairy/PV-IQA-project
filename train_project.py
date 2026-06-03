import argparse
import os
import sys

os.environ.setdefault("WANDB_MODE", "offline")
sys.path.insert(0, "src")


def require_supported_python() -> None:
    if sys.version_info < (3, 10):
        version = ".".join(map(str, sys.version_info[:3]))
        raise SystemExit(
            f"Python {version} is too old for this project. "
            "Use the palm_iqa environment, for example: "
            "conda activate palm_iqa"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the full PV-IQA pipeline.")
    parser.add_argument("--data", required=True, help="Training dataset root.")
    parser.add_argument("--name", default="pv_iqa_run", help="Run name under checkpoints/.")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers. Use 0 on Windows.")
    parser.add_argument("--recog-epochs", type=int, default=None, help="Recognizer epochs.")
    parser.add_argument("--iqa-epochs", type=int, default=None, help="IQA epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Evaluation batch size.")
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_supported_python()

    from pv_iqa.config import Config
    from pv_iqa.train.iqa import train_iqa
    from pv_iqa.train.pseudo_labels import generate_pseudo_labels
    from pv_iqa.train.recognition import export_features, train_recognizer
    from pv_iqa.utils.common import set_seed
    from pv_iqa.utils.datasets import build_metadata
    from pv_iqa.utils.export_onnx import export_onnx

    config = Config(
        data_root=args.data,
        name=args.name,
        device=args.device,
        num_workers=args.workers,
    ).resolve()

    if args.recog_epochs is not None:
        config.recog_epochs = args.recog_epochs
    if args.iqa_epochs is not None:
        config.iqa_epochs = args.iqa_epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.eval_batch_size is not None:
        config.eval_batch_size = args.eval_batch_size

    set_seed(config.seed)

    print(f"Run: {config.name}", flush=True)
    print(f"Data: {config.data_root}", flush=True)
    print(f"Device: {config.device}", flush=True)

    print("1 prepare data", flush=True)
    build_metadata(config)

    print("2 train recognizer", flush=True)
    recog_ckpt = train_recognizer(config)
    export_features(config, recog_ckpt)

    print("3 pseudo-labels", flush=True)
    generate_pseudo_labels(config)

    print("4 train iqa", flush=True)
    iqa_ckpt = train_iqa(config)

    if not args.skip_onnx:
        print("5 export onnx", flush=True)
        export_onnx(config, iqa_ckpt)

    print(f"Done: checkpoints/{config.name}", flush=True)
    print(f"IQA checkpoint: checkpoints/{config.name}/iqa/best.pt", flush=True)


if __name__ == "__main__":
    main()
