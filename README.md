# DA6401 Assignment 1 — Multi-Layer Perceptron for Image Classification

## Overview

This repository contains a NumPy-only implementation of a configurable Multi-Layer Perceptron (MLP) for image classification on MNIST and Fashion-MNIST.

The assignment requires:
- Modular neural network implementation (forward + backward pass)
- Multiple optimizers
- CLI-driven training/inference
- W&B-based experimentation and reporting

No deep learning frameworks (PyTorch, TensorFlow, JAX autograd, etc.) are used for model implementation.

---

## Repository Structure

```text
assignment_1/
├── src/
│   ├── ann/
│   │   ├── activations.py
│   │   ├── neural_layer.py
│   │   ├── neural_network.py
│   │   ├── objective_functions.py
│   │   └── optimizers.py
│   ├── utils/
│   │   └── data_loader.py
│   ├── train.py
│   ├── inference.py
│   ├── log_dataset_samples.py
│   ├── generate_error_analysis.py
│   ├── analyze_dead_relu_tanh.py
│   ├── analyze_global_performance.py
│   ├── analyze_global_performance_strict.py
│   └── generate_parallel_coordinates.py
├── images/
├── models/
│   └── best_model.npy
├── best_config.json
├── report.md
├── requirements.txt
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

---

## Training

### Recommended final training run

```bash
.env\Scripts\python.exe src\train.py --config best_config.json
```

### Example explicit run

```bash
.env\Scripts\python.exe src\train.py -d mnist -e 10 -b 64 -lr 0.001 -nhl 2 -sz 128 64 -a tanh -l cross_entropy -wi xavier -o adam
```

Notes:
- Training uses a validation split from the training set.
- Test set is reserved for final evaluation only.

---

## Inference

```bash
.env\Scripts\python.exe src\inference.py --model_path models\best_model.npy -d mnist -sz 128 -a tanh -l cross_entropy
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
- Public W&B report link

### Links

- GitHub Repository: `<ADD_PUBLIC_GITHUB_REPO_URL_HERE>`
- W&B Report: `<ADD_PUBLIC_WANDB_REPORT_URL_HERE>`

---

## Reproducibility and Analysis Scripts

The following scripts were used to generate report artifacts:

- `src/log_dataset_samples.py` (Section 2.1)
- `src/generate_parallel_coordinates.py` (Section 2.2)
- `src/analyze_dead_relu_tanh.py` (Section 2.5)
- `src/analyze_global_performance.py` (historical/proxy analysis)
- `src/analyze_global_performance_strict.py` (strict train-vs-test analysis)
- `src/generate_error_analysis.py` (confusion matrix + failure gallery)

---

## Notes

- This is coursework for DA6401 (Introduction to Deep Learning).
- The repository is structured to satisfy assignment grading constraints and reproducible experimentation.