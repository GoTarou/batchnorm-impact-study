import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from dataset import get_dataloaders
from models import MLP
from train import train_model
from evaluate import plot_history, save_confusion_matrix


def run_single_experiment(name, use_batchnorm=False, use_dropout=False, lr=0.001):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)

    model = MLP(use_batchnorm=use_batchnorm, use_dropout=use_dropout)
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=20,
        lr=lr,
        patience=5,
        weight_decay=1e-4,
    )

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    plot_history(history, model_name=name)
    save_confusion_matrix(y_true, y_pred, model_name=name)

    return {
        "model": name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_val_acc": max(history["val_acc"]),
        "epochs_ran": len(history["val_acc"]),
    }


def main():
    experiments = [
        ("baseline_mlp", False, False, 0.001),
        ("batchnorm_mlp", True, False, 0.001),
        ("dropout_mlp", False, True, 0.001),
        ("batchnorm_dropout_mlp", True, True, 0.001),
        ("baseline_high_lr", False, False, 0.01),
        ("batchnorm_high_lr", True, False, 0.01),
        ("baseline_very_high_lr", False, False, 0.05),
        ("batchnorm_very_high_lr", True, False, 0.05),
    ]

    results = []

    for name, use_bn, use_do, lr in experiments:
        print(f"\\nRunning experiment: {name}")
        result = run_single_experiment(
            name,
            use_batchnorm=use_bn,
            use_dropout=use_do,
            lr=lr
        )
        results.append(result)

    print("\\nFinal Results")
    print("-" * 80)
    for r in results:
        print(
            f"{r['model']}: "
            f"Acc={r['accuracy']:.4f}, "
            f"Precision={r['precision']:.4f}, "
            f"Recall={r['recall']:.4f}, "
            f"F1={r['f1']:.4f}, "
            f"Best Val Acc={r['best_val_acc']:.4f}, "
            f"Epochs={r['epochs_ran']}"
        )


if __name__ == "__main__":
    main()
