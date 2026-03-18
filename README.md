<div align="center">

# ProtoPNet — Prototype-Based Interpretable Image Recognition

### Skin Lesion Classification · HAM10000 · Prototype Pruning

*A neural network that says "this looks like that" — and can be pruned by what it recognises.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey?style=flat-square)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-ICW%20Report%20(DE)-8A2BE2?style=flat-square&logo=adobeacrobatreader&logoColor=white)](docs/ICW_report.pdf)
[![LRP Pruning Repo](https://img.shields.io/badge/Companion%20Repo-LRP__pruning__ICW1-24292e?style=flat-square&logo=github)](https://github.com/ilonae/LRP_pruning_ICW1)

**Independent Coursework (ICW 1) · M.Sc. Applied Computer Science · HTW Berlin**
*Ilona Eisenbraun · [ilonaeisenbraun@gmail.com](mailto:ilonaeisenbraun@gmail.com)*

</div>

## What is this?

Standard deep learning classifiers are black boxes — they output a class label with no human-interpretable explanation. **ProtoPNet** (Chen et al., NeurIPS 2019) changes this by building a network that reasons explicitly through *visual prototypes*: learned image patches that represent characteristic appearance patterns for each class.

At inference time the network says, in effect: *"This dermatoscopic image looks like prototype #12 (a distinctive `mel` lesion pattern) — therefore it is likely melanoma."* The decision is directly traceable to specific training image patches.

This repository adapts the original ProtoPNet codebase to:

- **HAM10000** — a 7-class dermoscopy dataset used as a real-world medical imaging benchmark
- **Prototype pruning** — removing prototypes that activate on non-discriminative regions, evaluated as part of the broader ICW comparison against LRP-guided pruning

The ProtoPNet companion lives alongside the LRP pruning implementation in:
[ilonae/LRP_pruning_ICW1](https://github.com/ilonae/LRP_pruning_ICW1)

> The original coursework report (written in German) is included in [`docs/ICW_report.pdf`](docs/ICW_report.pdf).

---

## Results

**Baseline:** ResNet18, 10 epochs → **90.21% test accuracy**

### Accuracy across pruning methods (from the ICW comparison)

| Pruning Criterion    | After pruning | After fine-tuning | Δ from baseline |
| -------------------- | :-----------: | :---------------: | :--------------: |
| LRP activations      |    72.16%     |     84.04%        |     −6.17%       |
| Weight magnitudes    |    72.16%     |     84.04%        |     −6.17%       |
| ProtoPNet prototypes |    81.32%     |   **90.11%**      |   **−0.10%**     |

### Key takeaway

> ProtoPNet prototype pruning achieves near-lossless compression because the pruning criterion is built into the architecture itself — prototypes that do not activate on class-specific regions are structurally identified and removed. This contrasts with post-hoc filter pruning (LRP / weight magnitude) which requires separate saliency computation.

---

## How ProtoPNet works

ProtoPNet trains in three phases:

```
Phase 1 — Joint training
  Input image
      │
      ▼
 ┌──────────────┐       ┌─────────────────────────┐
 │  CNN backbone│──────►│  Prototype layer        │
 │ (ResNet/VGG/ │       │  P = {p_1, ..., p_K}    │
 │  DenseNet)   │       │  distance to each patch │
 └──────────────┘       └────────────┬────────────┘
                                      │
                              Global min-pool
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │  Fully connected layer  │
                         │  (fixed class identity) │
                         └─────────────────────────┘

Phase 2 — Push (prototype projection)
  Replace each learned prototype vector with the
  nearest actual training image patch in feature space.
  Prototypes become human-interpretable image crops.

Phase 3 — Prune + fine-tune last layer
  Remove prototypes that never activate strongly on
  their assigned class (k-nearest-patch criterion).
  Fine-tune the last layer weights on remaining prototypes.
```

The prototype activation for a patch `z` and prototype `p_j` is:

```
similarity(z, p_j) = log((dist(z, p_j) + 1) / (dist(z, p_j) + epsilon))
```

This similarity is maximised (pooled globally), fed through the last layer, and trained with a combined loss that encourages prototypes to be:
- **Close** to at least one training patch of their class (cluster cost)
- **Far** from patches of other classes (separation cost)

---

## Dataset

**HAM10000** — 10,015 dermatoscopic images labelled by medical professionals across 7 classes:

| Code      | Class                         | Notes                                 |
| --------- | ----------------------------- | ------------------------------------- |
| `nv`    | Melanocytic nevi              | Largest class — oversampling applied  |
| `mel`   | Melanoma                      | Clinically critical                   |
| `bkl`   | Benign keratosis-like lesions |                                       |
| `bcc`   | Basal cell carcinoma          |                                       |
| `akiec` | Actinic keratoses             |                                       |
| `vasc`  | Vascular lesions              | Smallest class                        |
| `df`    | Dermatofibroma                |                                       |

Class imbalance is addressed via **per-class augmentation resampling** in the data loader.

**Download options:**

- [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) — original source, DOI: 10.7910/DVN/DBW86T
- [ISIC Archive](https://isic-archive.com) — ISIC 2018 Task 3

**Expected directory structure:**

```
datasets/ham10000/
├── HAM10000_metadata.csv
├── HAM10000_images_part1/
│   └── *.jpg
└── HAM10000_images_part2/
    └── *.jpg
```

---

## Modernisation — what changed from the original coursework

The original code targeted an older PyTorch / pandas API. The `update/modernize-dependencies` branch brings it fully up to date:

| # | File(s)                                                                       | What was broken                                              | Fix                                               |
| - | ----------------------------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------- |
| 1 | `main.py`, `run_pruning.py`                                                 | `df.append()` removed in pandas 2.0                        | Replaced with `pd.concat()`                     |
| 2 | `modules/resnet_features.py`, `densenet_features.py`, `vgg_features.py`    | `pretrained=True/False` parameter convention deprecated     | Updated to `weights='DEFAULT'` / `weights=None` |
| 3 | `modules/model.py`                                                           | `construct_PPNet` passed `pretrained=` to feature functions | Updated to translate to `weights=` internally   |
| 4 | All of the above                                                              | No `requirements.txt` in the original repo                  | Added `requirements.txt`                        |

---

## Setup

```bash
git clone https://github.com/ilonae/ProtoPNet_ICW1.git
cd ProtoPNet_ICW1

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **macOS SSL note:** Python from python.org ships without system certificates.
> If you get `CERTIFICATE_VERIFY_FAILED` when downloading model weights, run:
>
> ```bash
> open /Applications/Python\ 3.*/Install\ Certificates.command
> ```

---

## Usage

### 1 — Train from scratch

Edit `settings.py` to configure backbone architecture, prototype shape, number of classes, and experiment name, then:

```bash
python main.py -gpuid 0
```

Checkpoints are saved to `./saved_models/<arch>/<experiment_run>/`.

### 2 — Prune a trained model

After training and pushing prototypes, run pruning:

```bash
python run_pruning.py \
  -gpuid 0 \
  -modeldir ./saved_models/resnet50/001/ \
  -model 10push0.9011.pth
```

This will:
1. Load the pushed model
2. Prune prototypes with fewer than `k` (default 6) nearest training patches above `prune_threshold` (default 3)
3. Fine-tune the last layer for 100 iterations
4. Save the pruned model

### 3 — Analysis

```bash
# Global analysis — visualise the most activated prototype patches
python analysis/global_analysis.py

# Local analysis — explain a single image prediction
python analysis/local_analysis.py
```

---

## Project structure

```
ProtoPNet_ICW1/
│
├── main.py                         # Training entry point (warm → joint → push phases)
├── run_pruning.py                  # Pruning entry point
├── settings.py                     # Hyperparameters and experiment config
├── requirements.txt
│
├── modules/
│   ├── model.py                    # PPNet class and construct_PPNet factory
│   ├── train_and_test.py           # warm_only / joint / last_only training loops
│   ├── push.py                     # Prototype push (projection to training patches)
│   ├── prune.py                    # Prototype pruning logic
│   ├── save.py                     # Conditional model saving
│   ├── find_nearest.py             # k-NN search in prototype space
│   ├── receptive_field.py          # Receptive field calculation for prototypes
│   ├── resnet_features.py          # Custom ResNet feature extractor (no FC head)
│   ├── densenet_features.py        # Custom DenseNet feature extractor
│   ├── vgg_features.py             # Custom VGG feature extractor
│   ├── helpers.py                  # Utility functions (makedir, etc.)
│   ├── log.py                      # Logger setup
│   ├── preprocess.py               # ImageNet normalisation constants
│   ├── img_aug.py                  # Image augmentation helpers
│   ├── img_crop.py                 # Prototype crop visualisation
│   ├── oversampler.py              # Class imbalance oversampling
│   ├── measure_flops.py            # FLOPs measurement
│   └── preprocess_BCS_DBT.py       # BCS-DBT dataset preprocessing
│
├── analysis/
│   ├── global_analysis.py          # Global prototype visualisation
│   ├── local_analysis.py           # Per-image explanation
│   ├── script.py                   # Batch analysis script
│   ├── ham10000.py                 # HAM10000 dataset utilities
│   └── combine_models.ipynb        # Model comparison notebook
│
└── docs/
    ├── ICW_report.pdf              # Full coursework paper (German)
    ├── this_looks_like_that.pdf    # Chen et al. 2019 original paper
    ├── poster.pdf                  # Project poster
    ├── slides.pdf                  # Presentation slides
    └── rt.png                      # Prototype activation visualisation example
```

---

## Background & references

This work adapts and extends:

> **Chen et al. (2019)** — *This looks like that: deep learning for interpretable image recognition.*
> NeurIPS 2019

> **Tschandl et al. (2018)** — *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.*
> Scientific Data · [doi:10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)

```bibtex
@inproceedings{chen2019looks,
  title     = {This looks like that: deep learning for interpretable image recognition},
  author    = {Chen, Chaofan and Li, Oscar and Tao, Daniel and Barnett, Alina and
               Rudin, Cynthia and Su, Jonathan K},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {32},
  year      = {2019}
}

@article{tschandl2018ham10000,
  title   = {The {HAM10000} dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions},
  author  = {Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  journal = {Scientific Data},
  volume  = {5},
  pages   = {180161},
  year    = {2018}
}
```

---

## License

[Apache License 2.0](LICENSE)
Original ProtoPNet framework © Chen et al. (2019).
