from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid


def token_rgb(tokens, prefix_tokens, size):
    tokens = tokens[:, prefix_tokens:].float()
    side = int(tokens.shape[1] ** 0.5)
    maps = []
    for image in tokens:
        image = image - image.mean(0, keepdim=True)
        _, _, right = torch.pca_lowrank(image, q=3, center=False)
        rgb = image @ right[:, :3]
        rgb = (rgb - rgb.amin(0)) / (rgb.amax(0) - rgb.amin(0) + 1e-6)
        maps.append(rgb.T.reshape(3, side, side))
    return F.interpolate(torch.stack(maps), size=size, mode="bilinear", align_corners=False)


def save_grid(cells, rows, columns, output):
    grid = to_pil_image(make_grid(cells, nrow=columns, padding=4))
    canvas = Image.new("RGB", (grid.width + 72, grid.height), "white")
    canvas.paste(grid, (72, 0))
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(rows):
        draw.text((6, 111 + index * 228), name, fill="black")
    path = Path(output)
    path = path if path.suffix == ".png" else path / "pca_grid.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.inference_mode()
def save_pca(model, dataset, output, count=8):
    device = next(model.parameters()).device
    prefix_tokens = model.vit.num_prefix_tokens
    images = torch.stack([dataset[index][0] for index in range(count)])
    mean = images.new_tensor((0.485, 0.456, 0.406))[:, None, None]
    std = images.new_tensor((0.229, 0.224, 0.225))[:, None, None]
    originals = ((images * std + mean).clamp(0, 1)).unbind()
    activations = model.activations(images.to(device))
    maps = [token_rgb(tokens, prefix_tokens, images.shape[-2:]).cpu() for tokens in activations.values()]
    cells = list(originals) + [image for layer in maps for image in layer]
    save_grid(cells, ["input", *activations], count, output)
