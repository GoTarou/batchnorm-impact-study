# Batch Normalization Impact Study

This project investigates the effect of Batch Normalization on deep feedforward neural networks using the Fashion-MNIST dataset. The focus is on understanding how BatchNorm influences training stability, convergence behavior, and robustness under different optimization settings.

## Models

The following model variants are implemented and compared:

- Baseline MLP  
- MLP + BatchNorm  
- MLP + Dropout  
- MLP + BatchNorm + Dropout  

## Objectives

- Analyze training stability  
- Evaluate convergence speed  
- Compare validation and test performance  
- Study behavior under different learning rates, including high learning rate settings  
- Examine the effect of different optimization methods  

## Experiments

The project includes:

- Comparison of model variants (Baseline, BatchNorm, Dropout, combined)
- Learning rate sensitivity analysis (0.001, 0.01, 0.05)
- Optimizer comparison (SGD, Momentum, Nesterov, Adam)

## Run

```bash
pip install -r requirements.txt
python src/experiment.py
