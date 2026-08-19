import argparse

from eval.data import load_dataset
from eval.pca import save_pca
from eval.probe import evaluate_probes
from models import MODEL_NAMES, load_model
from models.registry import download_checkpoints


def parser():
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-checkpoints")
    download.add_argument("models", nargs="*")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("model", choices=MODEL_NAMES)
    common.add_argument("dataset")
    common.add_argument("--config", default="configs/snellius.yaml")
    common.add_argument("--checkpoints", default="checkpoints")

    pca = commands.add_parser("pca", parents=[common])
    pca.add_argument("--split", default="train")
    pca.add_argument("--count", type=int, default=8)
    pca.add_argument("--output", default="outputs/pca")

    probe = commands.add_parser("probe", parents=[common])
    probe.add_argument("--train-split", default="train")
    probe.add_argument("--val-split", default="val")
    probe.add_argument("--batch-size", type=int, default=64)
    probe.add_argument("--workers", type=int, default=4)
    probe.add_argument("--train-limit", type=int, default=50000)
    probe.add_argument("--val-limit", type=int, default=10000)
    probe.add_argument("--output", default="outputs/probes.json")
    return root


def main():
    args = parser().parse_args()
    if args.command == "download-checkpoints":
        download_checkpoints(names=args.models or MODEL_NAMES)
        return
    model = load_model(args.model, args.checkpoints)
    if args.command == "pca":
        dataset = load_dataset(args.config, args.dataset, args.split)
        save_pca(model, dataset, args.output, args.count)
        return
    train = load_dataset(args.config, args.dataset, args.train_split)
    val = load_dataset(args.config, args.dataset, args.val_split)
    evaluate_probes(
        model, train, val, args.output, args.batch_size, args.workers,
        args.train_limit, args.val_limit,
    )


if __name__ == "__main__":
    main()
