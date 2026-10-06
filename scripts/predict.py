"""Classify new images with a saved stage checkpoint from `continual_dl.run`."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from continual_dl.data.transforms import build_eval_transform
from continual_dl.models import build_classifier
from continual_dl.strategies import build_strategy
from continual_dl.utils import resolve_device


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Stage checkpoint, e.g. outputs/ncm/seed_42/checkpoints/stage_3.pt",
    )
    parser.add_argument("images", nargs="+", type=Path, help="Image files or directories")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def collect_images(paths: list[Path]) -> list[Path]:
    images: list[Path] = []
    for path in paths:
        if path.is_dir():
            images.extend(
                sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)
            )
        elif path.is_file():
            images.append(path)
        else:
            raise FileNotFoundError(f"Image path not found: {path}")
    if not images:
        raise ValueError("No images found")
    return images


def seen_class_ids(config: dict, stage_id: int) -> list[int]:
    class_order = [str(value) for value in config["data"]["class_order"]]
    class_to_id = {name: index for index, name in enumerate(class_order)}
    seen: list[int] = []
    for stage in config["data"]["stages"][: stage_id + 1]:
        seen.extend(class_to_id[str(name)] for name in stage)
    return seen


def load_strategy(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = payload["config"]
    # All weights come from the checkpoint, so skip the ImageNet download and
    # any backbone initialization file used during training.
    model_config = {**config["model"], "pretrained": False, "init_checkpoint": None}
    model = build_classifier(model_config)
    strategy = build_strategy(model, device, {**config, "model": model_config})
    state = payload["strategy_state"]
    strategy.model.load_state_dict(state["model"])
    strategy.load_extra_state_dict(state.get("extra", {}))
    return strategy, config, int(payload["stage_id"])


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    strategy, config, stage_id = load_strategy(args.checkpoint, device)
    class_order = [str(value) for value in config["data"]["class_order"]]
    seen = seen_class_ids(config, stage_id)
    transform = build_eval_transform(int(config["data"].get("image_size", 224)))
    images = collect_images(args.images)

    print(
        f"strategy={strategy.name}, stage={stage_id}, "
        f"seen classes={[class_order[class_id] for class_id in seen]}"
    )
    for start in range(0, len(images), args.batch_size):
        batch_paths = images[start : start + args.batch_size]
        batch = torch.stack(
            [transform(Image.open(path).convert("RGB")) for path in batch_paths]
        )
        predictions = strategy.predict(batch, seen).cpu().tolist()
        for path, class_id in zip(batch_paths, predictions):
            print(f"{path}\t{class_order[int(class_id)]}")


if __name__ == "__main__":
    main()
