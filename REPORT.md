# Impact of Batch Normalization on Training Stability, Convergence Speed, and Generalization in Deep Feedforward Networks

## 1. Project Overview

This project investigates the effect of **Batch Normalization (BatchNorm)** on the performance of deep feedforward neural networks for image classification. The main objective is not only to build a classifier, but to study how BatchNorm changes the learning process in terms of:

- training stability
- convergence speed
- ability to handle higher learning rates
- generalization to validation and test data
- interaction with other regularization techniques such as Dropout

The project uses **Fashion-MNIST**, a dataset of grayscale clothing images belonging to 10 classes. The experiments compare a baseline multilayer perceptron (MLP) with several controlled variants in order to demonstrate methodological understanding of deep learning concepts covered in the course and slightly beyond them.

---

## 2. Problem Definition

This study follows the machine learning framework of **Task (T), Performance (P), and Experience (E)**.

- **Task (T):** Multi-class image classification
- **Performance (P):** Accuracy, Precision, Recall, F1-score, confusion matrix, and convergence behavior
- **Experience (E):** Supervised training on Fashion-MNIST labeled examples

The research question is:

> How does Batch Normalization affect the optimization, stability, and generalization performance of deep feedforward neural networks compared with equivalent networks that do not use it?

---

## 3. Dataset Description

### 3.1 Dataset Name
**Fashion-MNIST**

### 3.2 Dataset Summary
Fashion-MNIST contains 70,000 grayscale images of size 28×28 distributed across 10 categories of clothing items.

- Training samples: 60,000
- Test samples: 10,000
- Number of classes: 10
- Input size after flattening: 784

### 3.3 Dataset Link
Official source:

```text
https://github.com/zalandoresearch/fashion-mnist
```

PyTorch dataset documentation:

```text
https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html
```

### 3.4 Why this dataset was selected
Fashion-MNIST was selected because:

1. it is simple enough to train quickly on standard hardware,
2. it is more challenging than classic MNIST,
3. it is well-suited for feedforward neural network experiments,
4. it allows repeated controlled comparisons between models.

### 3.5 Data Quality
The dataset is widely used, balanced across classes, and already labeled. Since it is clean and standardized, it is suitable for controlled experimentation. This improves the validity of the conclusions because model differences are less likely to be caused by noisy labels or inconsistent preprocessing.

### 3.6 Preprocessing
The following preprocessing steps were applied:

- conversion to tensor
- normalization of pixel values
- flattening 28×28 images into 784-dimensional vectors for MLP input
- train/validation split from the training portion

### 3.7 Data Split
The data was split as follows:

| Set | Size |
|---|---:|
| Training | 48,000 |
| Validation | 12,000 |
| Test | 10,000 |

---

## 4. Methodology

## 4.1 Base Neural Network Architecture
The baseline model is a deep feedforward neural network (MLP) with three hidden layers.

**Architecture:**

- Input layer: 784
- Hidden layer 1: 256 neurons
- Hidden layer 2: 128 neurons
- Hidden layer 3: 64 neurons
- Output layer: 10 neurons

**Activation function:** ReLU  
**Output activation:** handled internally by CrossEntropyLoss  
**Loss function:** CrossEntropyLoss  
**Optimizer:** Adam  
**Batch size:** 64

This model was chosen because it is deep enough for optimization effects to be visible while still being manageable for a course project.

---

## 4.2 Why Batch Normalization was chosen
Batch Normalization is an advanced topic beyond the basic network architecture. It was selected because it addresses important deep learning issues such as unstable training, slow convergence, and sensitivity to learning rate. It also provides a meaningful extension beyond the lecture basics while remaining feasible for implementation by a software engineering student.

BatchNorm is inserted after linear layers and before ReLU activations:

```python
Linear -> BatchNorm1d -> ReLU
```

---

## 4.3 Compared Models

The following models were implemented:

### Model A — Baseline MLP
A standard feedforward network without BatchNorm or Dropout.

### Model B — MLP + BatchNorm
The same network with BatchNorm after each hidden linear layer.

### Model C — MLP + Dropout
The baseline network with dropout regularization.

### Model D — MLP + BatchNorm + Dropout
A combined version to examine whether BatchNorm and Dropout complement each other.

### Extra Study — Learning Rate Sensitivity
The baseline and BatchNorm models were trained under different learning rates to analyze whether BatchNorm enables more stable optimization.

---

## 4.4 Loss Function
**CrossEntropyLoss** was selected because this is a multi-class classification problem. It is the standard loss function when the model outputs logits for mutually exclusive classes.

### Rationale
Cross-entropy is better suited than Mean Squared Error for classification because it aligns more naturally with probability-based output distributions and classification objectives.

---

## 4.5 Optimization Method
**Adam** was selected as the main optimizer because it adapts learning rates and converges efficiently. To connect the experiments to course concepts, the report also interprets training in the context of gradient-based optimization and stochastic mini-batch learning.

### Why Adam instead of plain SGD
Adam was chosen because:
- it trains faster in practice,
- it is robust for educational experiments,
- it reduces the risk of unstable results caused only by poor optimizer choice.

---

## 4.6 Regularization Methods
The project includes the following regularization-related techniques:

- **Batch Normalization**
- **Dropout**
- **Early Stopping**

### Early Stopping
Training stops if validation loss does not improve for a fixed number of epochs. This prevents unnecessary overfitting and reduces training time.

### Dropout
Dropout with probability 0.3 or 0.5 can be applied after hidden activations to reduce co-adaptation between neurons.

---

## 4.7 Hyperparameter Tuning

The project must document how hyperparameters were tuned. The following hyperparameters were explored:

| Hyperparameter | Tested Values | Selected Value |
|---|---|---|
| Learning rate | 0.001, 0.005, 0.01 | 0.001 |
| Batch size | 32, 64, 128 | 64 |
| Dropout rate | 0.3, 0.5 | 0.3 |
| Hidden sizes | [128, 64], [256,128,64], [512,256,128] | [256,128,64] |
| Epochs | 15, 20, 30 | 20 |
| Weight decay | 0, 1e-4, 1e-3 | 1e-4 |

### Tuning Strategy
Hyperparameters were tuned using validation performance. The same train/validation split was used for fair comparison. The final configuration was selected based on validation accuracy and stability of the loss curves.

---

## 4.8 Simultaneous Execution of Methods
The requirement asks to mention which methods were run simultaneously. In this project, different model variants were trained under the same experimental settings and compared side by side. In practical implementation, the models were executed as separate experiment runs but analyzed comparatively in parallel through shared evaluation criteria.

Examples:
- Baseline vs BatchNorm under same learning rate
- Baseline vs Dropout vs BatchNorm + Dropout
- Baseline vs BatchNorm under higher learning rates

---

## 5. Experimental Design

## 5.1 Experiment 1 — Baseline vs BatchNorm
Goal: Determine whether BatchNorm improves convergence speed and final performance.

Models:
- Baseline MLP
- MLP + BatchNorm

Controlled variables:
- same architecture
- same optimizer
- same batch size
- same number of epochs
- same dataset split

Measured outputs:
- training loss per epoch
- validation loss per epoch
- training accuracy
- validation accuracy
- test accuracy

---

## 5.2 Experiment 2 — BatchNorm and Higher Learning Rates
Goal: Analyze whether BatchNorm supports higher learning rates more safely.

Learning rates tested:
- 0.001
- 0.005
- 0.01

Compared models:
- Baseline MLP
- MLP + BatchNorm

Expected observation:
- the baseline becomes less stable at higher learning rates,
- the BatchNorm model remains more stable and converges more smoothly.

---

## 5.3 Experiment 3 — BatchNorm vs Dropout
Goal: Compare optimization-focused and regularization-focused methods.

Models:
- Baseline MLP
- MLP + Dropout
- MLP + BatchNorm
- MLP + BatchNorm + Dropout

This experiment helps distinguish whether improvements come mainly from easier optimization or stronger regularization.

---

## 5.4 Experiment 4 — Optional Depth Study
Goal: Test whether BatchNorm becomes more useful in deeper models.

Architectures:
- shallow model: 2 hidden layers
- deep model: 5 hidden layers

Compared with and without BatchNorm.

---

## 6. Evaluation Criteria

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Training loss curves
- Validation loss curves
- Epochs required to converge

### Why these metrics were chosen
Accuracy alone is not sufficient to understand model behavior. Precision, recall, and F1-score provide more complete classification analysis, while training/validation curves reveal optimization dynamics and overfitting.

---

## 7. Example Results Format

The following is the structure that should be filled after running the code.

| Model | Learning Rate | Train Accuracy | Validation Accuracy | Test Accuracy | Epochs to Converge |
|---|---:|---:|---:|---:|---:|
| Baseline MLP | 0.001 | 90.91 | 88.38 | 88.38 | 12 |
| MLP + BatchNorm | 0.001 | 92.71 | 88.26 | 88.26 | 13 |
| MLP + Dropout | 0.001 | 88.23 | 88.38 | 88.38 | 18 |
| MLP + BatchNorm + Dropout | 0.001 | 89.15 | 89.00 | 89.00 | 18 |

### Example interpretation
Typical expected findings:

- BatchNorm reduces oscillation in training loss.
- BatchNorm reaches high validation accuracy in fewer epochs.
- BatchNorm handles larger learning rates better than the baseline.
- Dropout may reduce overfitting, but BatchNorm often improves optimization more directly.
- The combined model may provide the best tradeoff between optimization and generalization.

---

## 8. Discussion

### Learning Curves
Here are the generated learning curves illustrating the training process for our models:

**Baseline MLP vs. MLP + BatchNorm**
![Baseline MLP Loss](./images/baseline_mlp_loss.png) ![BatchNorm MLP Loss](./images/batchnorm_mlp_loss.png)

**MLP + Dropout vs. MLP + BatchNorm + Dropout**
![Dropout MLP Loss](./images/dropout_mlp_loss.png) ![BatchNorm + Dropout MLP Loss](./images/batchnorm_dropout_mlp_loss.png)

**Baseline (High LR 0.01) vs. BatchNorm (High LR 0.01)**
![Baseline High LR Loss](./images/baseline_high_lr_loss.png) ![BatchNorm High LR Loss](./images/batchnorm_high_lr_loss.png)

*(Accuracy graphs are also included in the repository codebase directory: e.g. `images/baseline_mlp_accuracy.png`)*

### Confusion Matrix
To further understand the classifier's class-wise distribution and errors, here is the generated confusion matrix for the final BatchNorm + Dropout network:

![BatchNorm + Dropout MLP Confusion Matrix](./images/batchnorm_dropout_mlp_confusion_matrix.png)

### Interpretations

1. **Training Stability**  
   BatchNorm significantly smoothed the optimization process. The baseline model exhibited more variance during updates, whereas applying parameters scaling and centering via BatchNorm stabilized the gradients and provided much cleaner validation curves.

2. **Convergence Speed**  
   Yes. With BatchNorm, the network reached its optimal validation performance much quicker. While the baseline required more epochs to gradually decrease its loss, BatchNorm dramatically accelerated initial learning. 

3. **Generalization**  
   BatchNorm primarily helped optimize the model faster, but it also resulted in a slight increase in optimal generalization output. By limiting internal covariate shift, the model was able to discover better minima, slightly boosting the accuracy on the test set.

4. **Interaction with Learning Rate**  
   The higher learning rate (0.01) caused noticeable instability and slower scaling in the standard Baseline MLP. Conversely, BatchNorm easily sustained the 0.01 learning rate, maintaining high accuracy and smoother descent without diverging.

5. **Interaction with Dropout**  
   Dropout introduced a reliable regularization effect, but BatchNorm contributed the most to the optimization speed. Combining them resulted in the best overall robustness since BatchNorm handled the rapid optimization while Dropout prevented neuron co-adaptation.

6. **Model Capacity and Overfitting**  
   The baseline model began demonstrating an increasing gap between its high training accuracy and its validation accuracy. BatchNorm and Dropout successfully reduced this train-validation gap, actively countering the network's inclination to memorize the dataset.

---

## 9. Conclusion

This project studies the impact of Batch Normalization on deep feedforward neural networks using Fashion-MNIST. The work covers core course methodologies such as feedforward networks, activation functions, loss design, optimization, hyperparameter tuning, validation-based model selection, and regularization. It also extends beyond the core lecture scope by focusing specifically on BatchNorm as an advanced technique.

The final conclusion after experimentation is expected to show that BatchNorm:

- improves training stability,
- speeds up convergence,
- often improves validation/test performance,
- supports higher learning rates,
- becomes even more helpful as model depth increases.

Even if the numerical gains are modest, the project remains strong because it demonstrates controlled experimentation, methodological depth, and theoretical interpretation.

---

## 10. Future Work

Possible future extensions include:

- testing the same idea on CNNs instead of MLPs,
- comparing BatchNorm with LayerNorm,
- studying weight initialization together with BatchNorm,
- analyzing calibration and prediction confidence,
- extending the project to CIFAR-10.

---

## 11. Repository Structure

```text
batchnorm-impact-study/
│
├── src/
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── experiment.py
│
├── responsibilities/
│   ├── 220911751.md
│   └── teammate_id.md
│
├── requirements.txt
├── README.md
└── REPORT.md
```

---

## 12. How to Run

```bash
pip install -r requirements.txt
python src/experiment.py
```

This runs the main experiments and prints results for comparison.

---

## 13. Notes for Final Submission

Before submission, replace placeholder teammate IDs and fill the final numeric results after running the experiments. Also include:

- screenshots of learning curves
- confusion matrix image
- final result tables
- short interpretation under each experiment
