from pathlib import Path
from urllib.request import urlretrieve

import torch

from models.common import load_weights
from models.dino import DINO
from models.lejepa import LeJEPA
from models.mae import MAE


CHECKPOINTS = {
    "dino": "https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_vits14_dino_2025-02-16_16-03-14/pretrain/version_0/checkpoints/epoch%3D99-step%3D1000900.ckpt",
    "lejepa": "https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_vits16_lejepa_2026-05-20_17-59-16/pretrain/version_0/checkpoints/epoch%3D099-val_online_cls_top1%3D0.5623.ckpt",
    "mae": "https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_vitb16_mae_2024-02-25_19-57-30/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt",
}
MODELS = {"dino": DINO, "lejepa": LeJEPA, "mae": MAE}
MODEL_NAMES = tuple(MODELS)


def checkpoint_path(name, root="checkpoints"):
    return Path(root) / name / "model.ckpt"


def download_checkpoints(root="checkpoints", names=MODEL_NAMES):
    for name in names:
        path = checkpoint_path(name, root)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            print(f"downloading {name}")
            urlretrieve(CHECKPOINTS[name], path)


def load_model(name, root="checkpoints", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_weights(MODELS[name](), checkpoint_path(name, root))
    return model.eval().to(device)
