# Batch Normalization Impact Study

This project compares deep feedforward neural networks with and without Batch Normalization on the Fashion-MNIST dataset.

## Models
- Baseline MLP
- MLP + BatchNorm
- MLP + Dropout
- MLP + BatchNorm + Dropout

## Main goals
- study training stability
- measure convergence speed
- compare validation/test performance
- analyze interaction with learning rate

## Run
```bash
pip install -r requirements.txt
python src/experiment.py
```
