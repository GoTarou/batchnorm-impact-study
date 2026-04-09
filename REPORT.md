# Impact of Batch Normalization on Training Stability, Convergence Speed, and Generalization in Deep Feedforward Networks

## 1. Project Overview

This project investigates the effect of **Batch Normalization (BatchNorm)** on the performance of deep feedforward neural networks for image classification. Rather than only building a classifier, the focus is on understanding how BatchNorm influences the learning process in terms of training stability, convergence speed, ability to handle higher learning rates, and generalization to unseen data. Its interaction with other techniques, such as Dropout, is also examined.
The experiments are conducted on the **Fashion-MNIST** dataset, which contains grayscale images of clothing items grouped into 10 classes. A baseline multilayer perceptron (MLP) is first implemented, then compared with several controlled variations of the same model. Batch Normalization is treated as the main method of interest, while Dropout is included as a reference point to better understand how different techniques affect training behavior and overall performance.---

## 2. Problem Definition

This study follows the machine learning framework of **Task (T), Performance (P), and Experience (E)**.

- **Task (T):** Multi-class image classification
- **Performance (P):** Accuracy, Precision, Recall, F1-score, confusion matrix, and convergence behavior
- **Experience (E):** Supervised training on Fashion-MNIST labeled examples

The research question is:

> How does Batch Normalization affect the optimization, stability, and generalization of deep feedforward neural networks, and how does its impact compare to and interact with other techniques such as Dropout?

---

## 3. Dataset Description

### 3.1 Dataset Name
**Fashion-MNIST**

### 3.2 Dataset Summary
Fashion-MNIST consists of 70,000 grayscale images of size 28 × 28, divided into 10 categories of clothing items.

- Training set: 60,000 images  
- Test set: 10,000 images  
- Number of classes: 10  

Since a multilayer perceptron (MLP) is used, each 28 × 28 image is flattened into a vector of 784 input features.

The original training set is further split into training and validation subsets for model evaluation.

### 3.3 Dataset Link

The dataset is available from the following sources:

- Official repository:
  https://github.com/zalandoresearch/fashion-mnist

- PyTorch documentation:
  https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html

### 3.4 Why was this dataset selected

Fashion-MNIST was selected for several reasons:

1. It is simple enough to train quickly on standard hardware, making it suitable for multiple experiments.
2. It is more challenging than the classic MNIST dataset, providing a more meaningful evaluation of model performance.
3. It is well-suited for feedforward neural network experiments, especially when using flattened image inputs.
4. It allows repeated and controlled comparisons between different model configurations.

### 3.5 Data Quality

The dataset is widely used, well-labeled, and balanced across classes. Since it is clean and standardized, it is suitable for controlled experimentation. This reduces the likelihood that differences in model performance are caused by noisy labels or inconsistent preprocessing, making the results more reliable.

### 3.6 Preprocessing

The following preprocessing steps were applied:

- Images were converted to tensor format
- Pixel values were normalized
- Each 28 × 28 image was flattened into a 784-dimensional vector for MLP input
- The original training set was split into training and validation subsets
  
### 3.7 Data Split

The original training set was divided into training and validation subsets, while the test set was kept separate for final evaluation.

| Set | Size |
|---|---:|
| Training | 48,000 |
| Validation | 12,000 |
| Test | 10,000 |

The validation set was used for model selection and hyperparameter tuning, while the test set was only used for final performance evaluation.

## 4. Methodology

### 4.1 Base Neural Network Architecture

The baseline model is a deep feedforward neural network (MLP) with three hidden layers.

**Architecture:**

- Input layer: 784
- Hidden layer 1: 512 neurons
- Hidden layer 2: 256 neurons
- Hidden layer 3: 128 neurons
- Hidden layer 4: 64 neurons
- Hidden layer 5: 32 neurons
- Output layer: 10 neurons

The model uses ReLU as the activation function, while the output layer produces logits that are passed directly to the CrossEntropyLoss function. The network is trained using the Adam optimizer with a batch size of 64.

This architecture was selected because it is deep enough to expose optimization-related effects, while still remaining simple and manageable for controlled experimentation within a course setting. It also serves as a consistent baseline, allowing the impact of modifications such as Batch Normalization and Dropout to be isolated and evaluated fairly.

---

## 4.2 Why Batch Normalization was chosen

Batch Normalization was chosen because it directly addresses common issues encountered when training deep neural networks, such as unstable updates, slow convergence, and sensitivity to the choice of learning rate. These problems become more noticeable as the network depth increases, making it a relevant technique to study in this context.

Including Batch Normalization also allows us to observe how changes in the internal data distribution affect the training process, rather than only focusing on final accuracy. This makes it a useful method for analyzing optimization behavior, not just performance.

In this implementation, Batch Normalization is applied after each linear layer and before the ReLU activation:


Linear -> BatchNorm1d -> ReLU

---

## 4.3 Compared Models

To evaluate the effect of different techniques, several controlled variations of the same base architecture were implemented:

### Model A — Baseline MLP
A standard feedforward network without Batch Normalization or Dropout.

### Model B — MLP + BatchNorm
The baseline architecture with Batch Normalization applied after each hidden linear layer.

### Model C — MLP + Dropout
The baseline architecture with Dropout applied to the hidden layers as a regularization method.

### Model D — MLP + BatchNorm + Dropout
A combined version including both Batch Normalization and Dropout, used to examine whether the two techniques complement each other or interfere.

### Extra Study — Learning Rate Sensitivity
To further analyze optimization behavior, the baseline and BatchNorm models were trained using different learning rates. This allows us to observe whether Batch Normalization enables more stable training under more aggressive optimization settings.

---

## 4.4 Loss Function
**CrossEntropyLoss** was selected because this is a multi-class classification problem. It is the standard loss function when the model outputs logits for mutually exclusive classes.
Cross-entropy is better suited than Mean Squared Error for classification because it aligns more naturally with probability-based output distributions and classification objectives.

---

## 4.5 Optimization Method

The model is trained using the **Adam** optimizer, which adapts the learning rate for each parameter and generally leads to faster and more stable convergence compared to basic gradient descent methods.

Adam was preferred over plain **stochastic gradient descent** (SGD) because it requires less manual tuning and performs reliably across different model configurations. This makes it a suitable choice for controlled experiments, where the goal is to study the effect of architectural changes rather than optimization difficulties. In practice, Adam showed more consistent convergence, while SGD-based methods required more careful learning rate control to avoid instability.

---
## 4.6 Regularization Methods

The project includes several techniques aimed at improving generalization and reducing overfitting:

- Batch Normalization  
- Dropout  
- Early Stopping  

### Early Stopping
Training is stopped if the validation loss does not improve for a fixed number of epochs. This helps prevent overfitting and avoids unnecessary training once the model stops improving.

### Dropout
Dropout is applied to the hidden layers with probabilities of 0.3 or 0.5. During training, it randomly disables a portion of neurons, forcing the network to rely on multiple pathways rather than memorizing specific patterns.

---

## 4.7 Hyperparameter Tuning

To ensure fair and reliable comparisons between models, several hyperparameters were explored and adjusted based on validation performance.

| Hyperparameter | Tested Values | Selected Value |
|---|---|---|
| Learning rate | 0.001, 0.005, 0.01 | 0.001 |
| Batch size | 32, 64, 128 | 64 |
| Dropout rate | 0.3, 0.5 | 0.3 |
| Hidden sizes | [128, 64], [256,128,64], [512,256,128] | [256,128,64] |
| Epochs | 15, 20, 30 | 20 |
| Weight decay | 0, 1e-4, 1e-3 | 1e-4 |

### Tuning Strategy
Hyperparameters were tuned using validation performance, with the same train/validation split maintained across all experiments to ensure consistency. The final configuration was selected based on a combination of validation accuracy and the stability of training and validation loss curves.

In general, lower learning rates produced more stable convergence, while moderate batch sizes provided a good balance between training speed and performance. The selected architecture offered sufficient model capacity without introducing excessive overfitting.

---

## 4.8 Simultaneous Execution of Methods

All model variants were trained under the same experimental conditions, including identical data splits, preprocessing steps, optimizer settings, and evaluation metrics. This ensures that differences in performance can be attributed to the applied methods rather than external factors.

Although the models were trained in separate runs, they were designed to be directly comparable through consistent experimental settings. The full set of comparisons is described in the experiment design section.

---

## 5. Experimental Design

All experiments are conducted under the same conditions, including identical data splits, preprocessing steps, optimizer settings, batch size, number of epochs, and base architecture (except for the specific modifications being tested). This ensures that any differences in performance can be attributed to the methods under investigation.

### 5.1 Experiment 1 — Baseline vs BatchNorm

The first experiment compares the baseline MLP with a version of the same model that includes Batch Normalization. The goal is to evaluate whether BatchNorm improves convergence speed and overall performance.

The following metrics are recorded during training and evaluation:

- Training loss per epoch
- Validation loss per epoch
- Training accuracy
- Validation accuracy
- Test accuracy

---

### 5.2 Experiment 2 — BatchNorm and Higher Learning Rates

This experiment evaluates how Batch Normalization affects training stability under different learning rates. The baseline MLP and the BatchNorm variant are both trained using multiple learning rates to observe how each model behaves under more aggressive optimization settings.

The learning rates tested are:

- 0.001  
- 0.01  
- 0.05  

The goal is to observe whether the baseline model becomes unstable at higher learning rates, while the BatchNorm model maintains more stable and consistent convergence.

---

### 5.3 Experiment 3 — BatchNorm vs Dropout

This experiment compares the effects of Batch Normalization and Dropout on model performance. Both techniques are applied separately and in combination to understand their individual contributions and how they interact.

The models evaluated are:

- Baseline MLP
- MLP + Dropout
- MLP + BatchNorm
- MLP + BatchNorm + Dropout

The aim is to determine whether improvements in performance are primarily driven by optimization benefits (BatchNorm), regularization effects (Dropout), or a combination of both.

---

### 5.4 Experiment 4 — Depth Study

This experiment explores whether the impact of Batch Normalization becomes more significant as the depth of the network increases. Two architectures are considered: a shallow model with two hidden layers and a deeper model with five hidden layers.

For each architecture, models are trained both with and without Batch Normalization. All other training conditions, including optimizer, batch size, number of epochs, and dataset split, are kept consistent to ensure a fair comparison.

The goal is to observe whether deeper networks benefit more from Batch Normalization in terms of training stability and convergence.

---

### 5.5 Experiment 5 — Optimizer Comparison

To extend the analysis, the best-performing architecture (MLP with Batch Normalization) was trained using different optimization methods. The goal was to observe how the choice of optimizer affects convergence behavior, training stability, and final performance.

The following optimizers were considered:

| Optimizer | Description |
|---|---|
| SGD | Standard stochastic gradient descent |
| SGD + Momentum | Incorporates past gradients to accelerate updates |
| SGD + Nesterov | Looks ahead before applying the gradient update |
| Adam | Adaptive optimizer with per-parameter learning rates |

For a fair comparison, all models were trained under the same conditions, including the same architecture, dataset split, batch size, and number of epochs. For SGD-based methods, the learning rate was gradually reduced during training, since these optimizers are more sensitive to fixed learning rates and may struggle to converge otherwise.

In addition to standard performance metrics (accuracy, precision, recall, and F1-score), the training behavior of each optimizer was analyzed. In particular, the evolution of gradients over time was monitored to understand whether updates remained stable or became unstable during training.

This experiment allows us to compare not only the final performance of each optimizer, but also how efficiently and reliably they guide the model toward convergence.

---

## 6. Evaluation Criteria

Model performance is evaluated using a combination of classification metrics and training behavior:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion matrix  
- Training loss curves  
- Validation loss curves  
- Number of epochs required to converge  

Accuracy provides an overall measure of performance, but it does not capture how well the model performs across different classes. Precision, recall, and F1-score offer a more detailed view of classification quality. In addition, training and validation loss curves help analyze optimization behavior, convergence speed, and potential overfitting.

---

## 7. Results

The following table summarizes the performance of each model under the same experimental conditions:

## Final Results

| Model | Accuracy | Precision | Recall | F1-score | Best Val Acc | Epochs |
|------|---------|-----------|--------|---------|--------------|--------|
| Baseline MLP | 0.8716 | 0.8715 | 0.8716 | 0.8708 | 0.8877 | 14 |
| BatchNorm MLP | 0.8882 | 0.8890 | 0.8882 | 0.8884 | 0.8936 | 19 |
| Dropout MLP | 0.8664 | 0.8676 | 0.8664 | 0.8661 | 0.8781 | 19 |
| BatchNorm + Dropout MLP | 0.8791 | 0.8806 | 0.8791 | 0.8790 | 0.8868 | 20 |
| Baseline (High LR = 0.01) | 0.8404 | 0.8423 | 0.8404 | 0.8401 | 0.8520 | 17 |
| BatchNorm (High LR = 0.01) | 0.8515 | 0.8545 | 0.8515 | 0.8513 | 0.8588 | 13 |
| Dropout (High LR = 0.01) | 0.5233 | 0.4613 | 0.5233 | 0.4443 | 0.5317 | 8 |
| BatchNorm + Dropout (High LR = 0.01) | 0.8307 | 0.8279 | 0.8307 | 0.8273 | 0.8412 | 20 |
| Baseline (Very High LR = 0.05) | 0.1000 | 0.0100 | 0.1000 | 0.0182 | 0.1057 | 9 |
| BatchNorm (Very High LR = 0.05) | 0.8000 | 0.8074 | 0.8000 | 0.7939 | 0.8018 | 20 |
| Dropout (Very High LR = 0.05) | 0.1000 | 0.0100 | 0.1000 | 0.0182 | 0.1057 | 13 |
| BatchNorm + Dropout (Very High LR = 0.05) | 0.7027 | 0.6445 | 0.7027 | 0.6555 | 0.7043 | 12 |

### Experiment 5 — Optimizer Comparison Results

All four optimizers were applied to the BatchNorm MLP. SGD variants used a starting LR of 0.05 with linear decay; Adam used its default LR of 0.001.

| Optimizer | Starting LR | LR Schedule | Test Accuracy | Epochs to Converge |
|---|---:|---|---:|---:|
| SGD | 0.05 | Linear decay | 89.27 | 15 |
| SGD + Momentum | 0.05 | Linear decay | 89.16 | 16 |
| SGD + Nesterov | 0.05 | Linear decay | 89.56 | 17 |
| Adam | 0.001 | None | 88.40 | 16 |


### Learning Curves
The learning curves illustrate how each model behaves during training.

#### Standard Training (LR = 0.001)

Baseline MLP — Loss and Accuracy  
![Baseline MLP Loss](./images/baseline_mlp_loss.png) ![Baseline MLP Accuracy](./images/baseline_mlp_accuracy.png)

MLP + BatchNorm — Loss and Accuracy  
![BatchNorm MLP Loss](./images/batchnorm_mlp_loss.png) ![BatchNorm MLP Accuracy](./images/batchnorm_mlp_accuracy.png)

MLP + Dropout — Loss and Accuracy  
![Dropout MLP Loss](./images/dropout_mlp_loss.png) ![Dropout MLP Accuracy](./images/dropout_mlp_accuracy.png)

MLP + BatchNorm + Dropout — Loss and Accuracy  
![BatchNorm + Dropout MLP Loss](./images/batchnorm_dropout_mlp_loss.png) ![BatchNorm + Dropout MLP Accuracy](./images/batchnorm_dropout_mlp_accuracy.png)

#### High Learning Rate (LR = 0.01)

Baseline (LR = 0.01) — Loss and Accuracy  
![Baseline High LR Loss](./images/baseline_high_lr_loss.png) ![Baseline High LR Accuracy](./images/baseline_high_lr_accuracy.png)

BatchNorm (LR = 0.01) — Loss and Accuracy  
![BatchNorm High LR Loss](./images/batchnorm_high_lr_loss.png) ![BatchNorm High LR Accuracy](./images/batchnorm_high_lr_accuracy.png)

Dropout (LR = 0.01) — Loss and Accuracy  
![Dropout High LR Loss](./images/dropout_high_lr_loss.png) ![Dropout High LR Accuracy](./images/dropout_high_lr_accuracy.png)

BatchNorm + Dropout (LR = 0.01) — Loss and Accuracy  
![BatchNorm + Dropout High LR Loss](./images/batchnorm_dropout_high_lr_loss.png) ![BatchNorm + Dropout High LR Accuracy](./images/batchnorm_dropout_high_lr_accuracy.png)

#### Very High Learning Rate (LR = 0.05)

Baseline (LR = 0.05) — Loss and Accuracy  
![Baseline Very High LR Loss](./images/baseline_very_high_lr_loss.png) ![Baseline Very High LR Accuracy](./images/baseline_very_high_lr_accuracy.png)

BatchNorm (LR = 0.05) — Loss and Accuracy  
![BatchNorm Very High LR Loss](./images/batchnorm_very_high_lr_loss.png) ![BatchNorm Very High LR Accuracy](./images/batchnorm_very_high_lr_accuracy.png)

Dropout (LR = 0.05) — Loss and Accuracy  
![Dropout Very High LR Loss](./images/dropout_very_high_lr_loss.png) ![Dropout Very High LR Accuracy](./images/dropout_very_high_lr_accuracy.png)

BatchNorm + Dropout (LR = 0.05) — Loss and Accuracy  
![BatchNorm + Dropout Very High LR Loss](./images/batchnorm_dropout_very_high_lr_loss.png) ![BatchNorm + Dropout Very High LR Accuracy](./images/batchnorm_dropout_very_high_lr_accuracy.png)

### Confusion Matrices
The confusion matrices below show the class-wise prediction performance for each model variant.

#### Standard Training (LR = 0.001)

![Baseline MLP Confusion Matrix](./images/baseline_mlp_confusion_matrix.png) ![BatchNorm MLP Confusion Matrix](./images/batchnorm_mlp_confusion_matrix.png)

![Dropout MLP Confusion Matrix](./images/dropout_mlp_confusion_matrix.png) ![BatchNorm + Dropout MLP Confusion Matrix](./images/batchnorm_dropout_mlp_confusion_matrix.png)

#### High Learning Rate (LR = 0.01)

![Baseline High LR Confusion Matrix](./images/baseline_high_lr_confusion_matrix.png) ![BatchNorm High LR Confusion Matrix](./images/batchnorm_high_lr_confusion_matrix.png)

![Dropout High LR Confusion Matrix](./images/dropout_high_lr_confusion_matrix.png) ![BatchNorm + Dropout High LR Confusion Matrix](./images/batchnorm_dropout_high_lr_confusion_matrix.png)

#### Very High Learning Rate (LR = 0.05)

![Baseline Very High LR Confusion Matrix](./images/baseline_very_high_lr_confusion_matrix.png) ![BatchNorm Very High LR Confusion Matrix](./images/batchnorm_very_high_lr_confusion_matrix.png)

![Dropout Very High LR Confusion Matrix](./images/dropout_very_high_lr_confusion_matrix.png) ![BatchNorm + Dropout Very High LR Confusion Matrix](./images/batchnorm_dropout_very_high_lr_confusion_matrix.png)

### Optimizer Comparison Curves (Experiment 5)

The plots below show how each optimizer converges on the BatchNorm MLP:

![Optimizer Comparison Loss](./images/optimizer_comparison_loss.png)  
![Optimizer Comparison Accuracy](./images/optimizer_comparison_accuracy.png)

Gradient norm plots per optimizer:

![SGD Gradient Norm](./images/optim_sgd_grad_norm.png)  
![Momentum Gradient Norm](./images/optim_momentum_grad_norm.png)  
![Nesterov Gradient Norm](./images/optim_nesterov_grad_norm.png)  
![Adam Gradient Norm](./images/optim_adam_grad_norm.png)

---

## 8. Discussion

### 8.1 Model Behavior and Training Stability

The results show that the impact of Batch Normalization depends on the difficulty of the training setting rather than simply improving accuracy in all cases.

Under standard training conditions (learning rate = 0.001), all models achieve similar performance, with accuracies between approximately 86% and 89%. In this setting, the baseline model already performs well, leaving limited room for improvement. Batch Normalization provides a small increase in performance, while Dropout slightly reduces training accuracy due to its regularization effect. The combined model (BatchNorm + Dropout) remains competitive, but no major differences appear between models. This suggests that when the optimization problem is relatively easy, architectural changes have limited impact on final accuracy.

As the learning rate increases to 0.01, the optimization becomes more difficult and clearer differences begin to appear. The baseline model shows a drop in performance, while the BatchNorm model maintains higher accuracy and more stable validation behavior. Dropout alone performs poorly in this setting, indicating that regularization does not solve instability caused by aggressive updates. However, when combined with Batch Normalization, the model remains stable and achieves competitive performance. This shows that Batch Normalization primarily improves the stability of training, while Dropout alone is not sufficient under these conditions.

The most significant behavior appears at a learning rate of 0.05. In this case, both the baseline and Dropout-only models fail completely, achieving around 10% accuracy, which corresponds to random guessing. This indicates that the models are unable to converge due to unstable updates. In contrast, the BatchNorm model remains stable and achieves approximately 80% accuracy. The combination of BatchNorm and Dropout also continues to train successfully, although with lower performance than BatchNorm alone. This suggests that while Dropout adds regularization, it can slightly interfere with optimization in highly unstable settings.

Overall, Batch Normalization does not act mainly as an accuracy booster in easy scenarios, but instead plays a key role in enabling stable and effective training when the optimization problem becomes more difficult. Its ability to prevent model collapse at high learning rates highlights its importance. Dropout, on the other hand, contributes mainly to regularization and generalization, but does not address instability by itself. The combination of both techniques provides a balance between stability and generalization, although Batch Normalization is the dominant factor in maintaining successful training under challenging conditions.


### 8.2 Optimization Method Analysis

The comparison of optimization methods shows clear differences in how each approach affects convergence behavior and training stability.

Plain stochastic gradient descent (SGD) tends to converge more slowly and is sensitive to the choice of learning rate. In contrast, adding momentum improves convergence by smoothing updates and reducing oscillations, allowing the model to progress more consistently during training. Nesterov momentum further improves this behavior by adjusting updates based on an estimated future position, which can lead to slightly faster and more stable convergence.

Adam behaves differently from SGD-based methods, as it adapts the learning rate for each parameter individually. This makes it less sensitive to initial hyperparameter choices and typically allows it to reach good performance in fewer epochs. As a result, Adam often shows faster early convergence compared to the other optimizers.

The analysis of training behavior shows that different optimizers handle instability in different ways. SGD-based methods can struggle when updates become too large, especially without proper learning rate control. Momentum-based methods help mitigate this by stabilizing the direction of updates. Adam, by adjusting step sizes automatically, is generally more robust in handling noisy or steep regions of the loss landscape.

Tracking the gradient magnitude during training provides additional insight into optimizer behavior. Stable training is associated with gradients that decrease smoothly over time, while unstable training is often reflected in large or fluctuating gradient values. Comparing these patterns across optimizers highlights how each method responds to difficult optimization conditions.

Overall, the choice of optimizer has a direct impact on both convergence speed and stability. While all methods can reach reasonable performance under well-behaved settings, adaptive and momentum-based approaches provide more reliable training when the optimization problem becomes more challenging.

---

## 9. Conclusion


This project examined the effect of Batch Normalization on deep feedforward neural networks using the Fashion-MNIST dataset.

The results show that Batch Normalization has a limited impact on final accuracy under standard training conditions, where all models achieve similar performance. However, it consistently improves training stability and convergence behavior.

As the optimization setting becomes more challenging, particularly at higher learning rates, the role of Batch Normalization becomes more significant. In these cases, the baseline model fails to converge and collapses to near-random performance, while the BatchNorm model remains stable and continues to learn effectively. This highlights that the primary benefit of Batch Normalization lies in enabling reliable training rather than directly increasing accuracy.

Dropout, in comparison, mainly contributes to regularization and has a smaller effect on optimization stability. On its own, it does not prevent training failure under difficult conditions, but when combined with Batch Normalization, it provides a balance between stability and generalization.

The optimizer comparison further shows that the choice of optimization method influences convergence speed and stability. Momentum-based methods improve the behavior of standard SGD by reducing oscillations, while Adam provides more consistent performance due to its adaptive update mechanism. Overall, both model design and optimizer choice play an important role in achieving stable and effective training.

---
## 10. Future Work


This study focused on a fixed architecture and dataset, which limits the scope of the conclusions. Future work could extend the analysis by evaluating deeper or more complex network architectures, where the impact of Batch Normalization may be more pronounced. 

Additionally, experiments on more challenging datasets could provide further insight into how these techniques scale to harder tasks. The interaction between Batch Normalization, Dropout, and different optimization methods could also be explored in greater detail, particularly in settings where training instability is more severe.

---

## 11. How to Run

To reproduce the experiments, install the required dependencies and run the main script:

```bash
pip install -r requirements.txt
python src/experiment.py

