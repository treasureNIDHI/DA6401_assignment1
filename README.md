# DA6401 Assignment 1 — Multi-Layer Perceptron for Image Classification

## Overview

This repository contains a NumPy-only implementation of a configurable Multi-Layer Perceptron (MLP) for image classification on MNIST and Fashion-MNIST.

The assignment requires:
- Modular neural network implementation (forward + backward pass)
- Multiple optimizers
- CLI-driven training/inference
- W&B-based experimentation and reporting (final report built in wandb.ai UI)

No deep learning frameworks (PyTorch, TensorFlow, JAX autograd, etc.) are used for model implementation.

---

## Repository Structure

```text
assignment_1/
├── .env/
├── data/
│   └── mnist.npz
├── images/
├── models/
│   ├── best_model.npy
│   └── ...
├── notebooks/
│   └── wandb_demo.ipynb
├── src/
│   ├── ann/
│   │   ├── activations.py
│   │   ├── __init__.py
│   │   ├── neural_layer.py
│   │   ├── neural_network.py
│   │   ├── objective_functions.py
│   │   └── optimizers.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── best_model.npy
│   ├── train.py
│   ├── inference.py
│   ├── error_analysis.py
│   ├── log_dataset_samples.py
│   ├── generate_error_analysis.py
│   ├── analyze_dead_relu_tanh.py
│   ├── analyze_global_performance.py
│   ├── analyze_global_performance_strict.py
│   └── generate_parallel_coordinates.py
├── wandb/
│   └── run-*/
├── best_config.json
├── pyrightconfig.json
├── report.md  (optional local notes; final report is on wandb.ai)
├── requirements.txt
├── sweep.yaml
├── wandb_export_2026-02-26T22_26_44.673+05_30.csv
└── README.md
```

---

## Environment Setup

```bash
python -m venv .env
.env\Scripts\activate
pip install -r requirements.txt
```

---

## CLI Contract (Autograder Compatibility)

`train.py` and `inference.py` support assignment-required arguments:

- `-d, --dataset` : `mnist` or `fashion` (`fashion_mnist` alias also accepted)
- `-e, --epochs`
- `-b, --batch_size`
- `-l, --loss` : `cross_entropy` or `mse`
- `-o, --optimizer` : `sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam`
- `-lr, --learning_rate`
- `-wd, --weight_decay`
- `-nhl, --num_layers`
- `-sz, --hidden_size` (list)
- `-a, --activation` : `sigmoid`, `tanh`, `relu`
- `-wi, --weight_init` : `random`, `xavier` (includes `zeros` for symmetry experiment)
- `-wp, --wandb_project, --wandb-project` : W&B project id/name

---

## Training

### Recommended final training run

```bash
.env\Scripts\python.exe src\train.py --config best_config.json
```

### Example explicit run

```bash
.env\Scripts\python.exe src\train.py -d mnist -e 15 -b 128 -lr 0.001 -nhl 2 -sz 128 64 -a relu -l cross_entropy -wi xavier -o rmsprop --wandb-project assignment_1-src
```

Notes:
- Training uses a validation split from the training set.
- Test set is reserved for final evaluation only.

---

## Inference

```bash
.env\Scripts\python.exe src\inference.py --model_path models\best_model.npy -d mnist -sz 128 64 -a relu -l cross_entropy
```

Inference reports:
- Accuracy
- Precision
- Recall
- F1-score

---

## Assignment Deliverables

Required submission artifacts:
- `models/best_model.npy`
- `best_config.json`
- Public GitHub repository
- Public W&B report link (created via wandb.ai UI)

### Links

- GitHub Repository: `<ADD_PUBLIC_GITHUB_REPO_URL_HERE>`
- W&B Report: [DA6401 Assignment 1 Report](https://wandb.ai/nidhi-jagatpura-iit-madras/assignment_1-src/reports/-DA6401-Assignment-1-Report--VmlldzoxNjA1NTQzMA?accessToken=onizvul3bhac2tgau1hds5klaaw1qw1yhkg005x4qumt0blhhcc729auhqd1qj5o)

### Latest Fashion-MNIST Results (March 2026)

| configuration | architecture | optimizer | activation | test accuracy |
|---|---|---|---|---|
| dataset=fashion_mnist, lr=0.001, batch_size=128, weight_init=xavier, epochs=10 (run cfjn0t0v) | 128-64 | adam | tanh | 0.8724 |
| dataset=fashion_mnist, lr=0.001, batch_size=128, weight_init=xavier, epochs=10 (run brhkoill) | 128-128-128 | rmsprop | relu | 0.8771 |
| dataset=fashion_mnist, lr=0.001, batch_size=128, weight_init=xavier, epochs=10 (run 5yasvfun) | 128-128-128 | adam | relu | 0.8738 |

---

## Reproducibility and Analysis Scripts

The following scripts were used to generate report artifacts:

- `src/log_dataset_samples.py` (Section 2.1)
- `src/generate_parallel_coordinates.py` (Section 2.2)
- `src/analyze_dead_relu_tanh.py` (Section 2.5)
- `src/analyze_global_performance.py` (historical/proxy analysis)
- `src/analyze_global_performance_strict.py` (strict train-vs-test analysis)
- `src/generate_error_analysis.py` (confusion matrix + failure gallery)

Generate confusion matrix and failure gallery:

```bash
.env\Scripts\python.exe src\generate_error_analysis.py best_config.json models\best_model.npy
```

---

