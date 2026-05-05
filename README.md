# Crack Detection Classifier

Binary image classification using :contentReference[oaicite:1]{index=1} ResNet18.

## Classes
- crack
- normal

## Pipeline

Dataset → Resize → CNN → Train → Evaluate → Inference

## Features
- Binary classification
- Loss curve
- Confusion matrix
- Precision / Recall / F1
- Model checkpoint
- Inference demo

## Run

```bash
python train.py
```

Inference:

```bash
python inference.py
```

## Outputs

- models/best_model.pth
- outputs/loss_curve.png
- outputs/confusion_matrix.png
