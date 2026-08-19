from pathlib import Path

import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Images(Dataset):
    def __init__(self, root, transform):
        self.paths = sorted(
            path for path in Path(root).rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.transform(Image.open(self.paths[index]).convert("RGB")), -1


def image_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def load_dataset(config, name, split):
    spec = yaml.safe_load(Path(config).read_text())["datasets"][name]
    root = Path(spec["root"])
    transform = image_transform()
    if spec["type"] == "stl10":
        return datasets.STL10(root, split=spec["splits"][split], transform=transform)
    path = root / spec["splits"][split]
    if spec["type"] == "imagefolder":
        return datasets.ImageFolder(path, transform)
    return Images(path, transform)
