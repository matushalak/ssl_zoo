# SSL Zoo

Inference and layer-wise evaluation for Lightly's ImageNet-1k DINO ViT-S/16, LeJEPA ViT-S/16, and MAE ViT-B/16 checkpoints. Model code lives in `models/`; weights live in `checkpoints/<model>/`.

No training code is included. Evaluation datasets are read only from paths in YAML and are never downloaded.

## Setup on Snellius

```bash
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e .
ssl-zoo download-checkpoints
```

The default dataset paths in `configs/snellius.yaml` point to SURF's shared ImageNet-1k, COCO 2017, and STL-10 copies. Copy `configs/local.example.yaml` for another machine.

## PCA to RGB

```bash
ssl-zoo pca dino imagenet1k --split val --count 8 --output outputs/dino-imagenet
ssl-zoo pca lejepa coco2017 --count 8 --output outputs/lejepa-coco
ssl-zoo pca mae stl10 --split val --count 8 --output outputs/mae-stl10
```

Each command writes `pca_grid.png`. Images are columns, the model input is the top row, and successive rows are transformer blocks. PCA is fit independently for every image and layer, each component is min-max scaled, and the patch map is bilinearly upscaled to 224x224. `model.activations(images)` returns the unpooled token tensor for every block.

## kNN and linear probe

```bash
ssl-zoo probe dino imagenet1k --output outputs/dino-imagenet-probes.json
```

The command evaluates the class token at every transformer block. CUDA is used when available, otherwise CPU.

Architectures and checkpoints match the [Lightly ImageNet benchmarks](https://github.com/lightly-ai/lightly#imagenet1k). Dataset locations match the [SURF Snellius dataset catalogue](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/62227640/Available+datasets+and+models+on+Snellius).
