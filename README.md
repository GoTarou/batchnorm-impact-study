# Batch Normalization Impact Study

This project investigates the effect of Batch Normalization on deep feedforward neural networks using the Fashion-MNIST dataset. The goal is to understand how BatchNorm influences training stability, convergence speed, and overall performance.

## Models

The following model variants are implemented and compared:

- Baseline MLP
- MLP + BatchNorm
- MLP + Dropout
- MLP + BatchNorm + Dropout

## Objectives
- Analyze training stability
- measure convergence speed
- compare validation annd test performance
- study the efect of different learning rates

## Run
```bash
pip install -r requirements.txt
python src/experiment.py
```
