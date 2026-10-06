from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

from continual_dl.constants import CLASS_TO_ID
from continual_dl.data.scenario import build_experiences
from continual_dl.models import build_classifier
from continual_dl.strategies import build_strategy
from scripts.predict import collect_images, load_strategy, seen_class_ids

TRANSFORM = Compose([Resize((32, 32)), ToTensor()])
STAGES = [["dog", "cat"], ["car"], ["person"], ["building"]]


def make_config(strategy: str) -> dict:
    return {
        "seed": 42,
        "data": {
            "num_workers": 0,
            "image_size": 32,
            "class_order": ["dog", "cat", "car", "person", "building"],
            "stages": STAGES,
        },
        "model": {"backbone": "tiny_cnn", "pretrained": False, "num_classes": 5, "dropout": 0.0},
        "training": {
            "batch_size": 4,
            "epochs_stage0": 1,
            "epochs_incremental": 1,
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "amp": False,
            "reset_optimizer_each_stage": True,
        },
        "strategy": {"name": strategy, "memory_size": 8, "replay_ratio": 0.5},
    }


def save_stage_checkpoint(tiny_manifests, tmp_path: Path, strategy_name: str, stages: int):
    config = make_config(strategy_name)
    experiences = build_experiences(
        train_manifest=tiny_manifests["train"],
        val_manifest=tiny_manifests["val"],
        test_manifest=tiny_manifests["test"],
        stages=STAGES,
        class_to_id=CLASS_TO_ID,
        train_transform=TRANSFORM,
        eval_transform=TRANSFORM,
    )
    strategy = build_strategy(build_classifier(config["model"]), torch.device("cpu"), config)
    for experience in experiences[:stages]:
        strategy.fit(experience)
    checkpoint = tmp_path / f"{strategy_name}_stage_{stages - 1}.pt"
    torch.save(
        {"stage_id": stages - 1, "config": config, "strategy_state": strategy.state_dict()},
        checkpoint,
    )
    return strategy, checkpoint


def test_seen_class_ids_follow_stage_order() -> None:
    config = make_config("naive")
    assert seen_class_ids(config, 0) == [0, 1]
    assert seen_class_ids(config, 2) == [0, 1, 2, 3]


def test_collect_images_searches_folders(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(nested / "x.jpg")
    (nested / "notes.txt").write_text("skip", encoding="utf-8")
    assert collect_images([tmp_path]) == [nested / "x.jpg"]


def test_loaded_checkpoint_reproduces_predictions(tiny_manifests, tmp_path: Path) -> None:
    images = torch.rand(6, 3, 32, 32)
    for strategy_name in ("naive", "ncm", "replay_ncm_hybrid"):
        original, checkpoint = save_stage_checkpoint(
            tiny_manifests, tmp_path, strategy_name, stages=2
        )
        restored, config, stage_id = load_strategy(checkpoint, torch.device("cpu"))
        seen = seen_class_ids(config, stage_id)
        assert stage_id == 1
        assert restored.name == strategy_name
        expected = original.predict(images, seen)
        actual = restored.predict(images, seen)
        assert torch.equal(expected, actual)
        assert set(actual.tolist()) <= {0, 1, 2}
