from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


class SSLModel(nn.Module):
    backbone: nn.Module

    @property
    def vit(self):
        backbone = self.backbone
        return getattr(backbone, "vit", backbone)

    def activations(self, images):
        vit = self.vit
        x = vit.patch_embed(images)
        x = vit._pos_embed(x)
        x = vit.patch_drop(x)
        x = vit.norm_pre(x)
        outputs = {}
        for index, block in enumerate(vit.blocks):
            x = block(x)
            outputs[f"block_{index:02d}"] = vit.norm(x)
        return outputs


def load_weights(model, path: str | Path):
    for name, module in list(sys.modules.items()):
        if name == "numpy.core" or name.startswith("numpy.core."):
            sys.modules.setdefault(name.replace("numpy.core", "numpy._core"), module)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=False)
    return model
