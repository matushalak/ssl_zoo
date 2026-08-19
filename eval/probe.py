import json
from pathlib import Path

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, Subset


@torch.inference_mode()
def extract(model, dataset, batch_size, workers, limit):
    if limit:
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))[:limit]
        dataset = Subset(dataset, indices)
    features = {}
    labels = []
    device = next(model.parameters()).device
    for images, target in DataLoader(dataset, batch_size=batch_size, num_workers=workers):
        for layer, tokens in model.activations(images.to(device)).items():
            features.setdefault(layer, []).append(tokens[:, 0].cpu())
        labels.append(target)
    return {key: torch.cat(value).numpy() for key, value in features.items()}, torch.cat(labels).numpy()


def evaluate_probes(model, train, val, output, batch_size=64, workers=4, train_limit=50000, val_limit=10000):
    train_x, train_y = extract(model, train, batch_size, workers, train_limit)
    val_x, val_y = extract(model, val, batch_size, workers, val_limit)
    scores = {}
    for layer in train_x:
        knn = KNeighborsClassifier(20, metric="cosine", n_jobs=-1).fit(train_x[layer], train_y)
        linear = LogisticRegression(max_iter=100).fit(train_x[layer], train_y)
        scores[layer] = {
            "knn": accuracy_score(val_y, knn.predict(val_x[layer])),
            "linear": accuracy_score(val_y, linear.predict(val_x[layer])),
        }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(scores, indent=2))
    return scores
