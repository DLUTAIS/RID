# RID

Official implementation of the paper:

**A Few-shot Sample Method for Source Camera Identification Based on Residual Information Distillation**  
Accepted by *Applied Intelligence*

This repository provides the source code for RID, a few-shot source camera identification method based on residual information distillation. The codebase contains the training and evaluation components used for the experiments reported in the paper, including classification backbones, residual information distillation modules, and supporting scripts.

## Overview

Source camera identification aims to determine the acquisition device of an image from camera-specific traces. In few-shot scenarios, only a limited number of labeled samples are available for each camera model or device, making robust feature learning and generalization especially challenging.

RID is designed to improve few-shot source camera identification by distilling residual information that is useful for discriminative camera trace representation. This repository is intended to support academic reproducibility and further research in multimedia forensics and source camera attribution.

## Repository Structure

```text
RID/
|-- CIFAR-100/                 # PyTorch training scripts, models, and distillation utilities
|-- Detection/                 # Detectron2-based model and configuration files
|-- ImageNet/                  # ImageNet training code and backbone definitions
|-- Paddle/Classification/     # PaddlePaddle classification implementation
|-- LICENSE                    # MIT License
`-- README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/DLUTAIS/RID.git
cd RID
```

The project contains multiple experimental components. Install the dependencies required by the part you want to run.

For the PaddlePaddle classification code:

```bash
cd Paddle/Classification
pip install -r requirements.txt
```

For the PyTorch and Detectron2 components, install compatible versions of Python, PyTorch, torchvision, CUDA, and Detectron2 according to your local environment.

## Data and Checkpoints

Datasets, pretrained weights, checkpoints, logs, and generated experimental results are not included in this repository.

Please prepare the required datasets and place local files under the corresponding working directories, for example:

```text
data/
dataset/
checkpoints/
pretrained/
output/
```

These paths are ignored by Git to avoid uploading private datasets, model weights, and experiment artifacts.

## Usage

### CIFAR/PyTorch Experiments

```bash
cd CIFAR-100
python train.py --model wrn-16-2 --suffix baseline1
```

Residual information distillation example:

```bash
python train.py \
  --model wrn-16-2 \
  --teacher wrn-40-2 \
  --teacher-weight checkpoints/cifar100_wrn-40-2__baseline1_best.pt \
  --kd-loss-weight 5.0 \
  --suffix reviewkd1
```

Additional examples are provided in:

```text
CIFAR-100/script/
```

### ImageNet Experiments

```bash
cd ImageNet
python imagenet_amp.py /path/to/imagenet --arch resnet18 --save_dir output/resnet18
```

### Detection Experiments

```bash
cd Detection
python train_net.py --config-file configs/ReviewKD-R18-R101.yaml
```

### PaddlePaddle Classification

```bash
cd Paddle/Classification
python tools/train.py -c configs/r50_mv1_reviewkd.yaml
python tools/eval.py -c configs/r50_mv1_reviewkd.yaml
```

## Citation

If this repository is useful for your research, please cite our paper.

```bibtex
@article{rid2026,
  title   = {A Few-shot Sample Method for Source Camera Identification Based on Residual Information Distillation},
  journal = {Applied Intelligence},
  year    = {2026}
}
```

The complete BibTeX entry will be updated after the final publication information is available.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgement

Parts of this repository build on widely used deep learning frameworks and open-source research code, including PyTorch, Detectron2, and PaddlePaddle. We thank the authors and maintainers of these projects.
