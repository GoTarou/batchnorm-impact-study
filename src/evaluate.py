import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def print_classification_results(y_true, y_pred):
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def save_confusion_matrix(y_true, y_pred, model_name="model"):
    os.makedirs("images", exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"images/{model_name}_confusion_matrix.png")
    plt.close()


def plot_history(history, model_name="model"):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title(f"Loss Curves - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/{model_name}_loss.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.title(f"Accuracy Curves - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/{model_name}_accuracy.png")
    plt.close()


def plot_grad_norm(history, model_name="model"):
    """
    Plot the average gradient norm over training epochs.
    Mirrors Figure 8.1 from §8.2.1 (Ill-Conditioning).
    Helps visualize whether gradients grow, shrink, or stay stable.
    """
    if "grad_norm" not in history or not history["grad_norm"]:
        return
    os.makedirs("images", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history["grad_norm"], label="Avg Gradient Norm", color="darkorange")
    plt.title(f"Gradient Norm over Training - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/{model_name}_grad_norm.png")
    plt.close()


def plot_optimizer_comparison(histories, labels, filename="optimizer_comparison"):
    """
    Overlay validation loss curves for multiple optimizers on one figure.
    Used for Experiment 5 (§8.3 / §8.5 optimizer comparison).

    Args:
        histories: list of history dicts
        labels:    list of string labels matching histories
        filename:  output filename (without extension)
    """
    os.makedirs("images", exist_ok=True)

    plt.figure(figsize=(9, 5))
    for history, label in zip(histories, labels):
        plt.plot(history["val_loss"], label=label)
    plt.title("Optimizer Comparison — Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/{filename}_loss.png")
    plt.close()

    plt.figure(figsize=(9, 5))
    for history, label in zip(histories, labels):
        plt.plot(history["val_acc"], label=label)
    plt.title("Optimizer Comparison — Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/{filename}_accuracy.png")
    plt.close()

