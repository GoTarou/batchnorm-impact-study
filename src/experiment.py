import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from dataset import get_dataloaders
from models import MLP
from train import train_model
from evaluate import plot_history, save_confusion_matrix, plot_grad_norm, plot_optimizer_comparison


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
        patience=7,
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
    plot_grad_norm(history, model_name=name)
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


def run_optimizer_experiment():
    """
    Experiment 5 — Optimizer Comparison (Chapter 8: §8.3.1–8.3.3, §8.5.3).

    Trains the BatchNorm MLP using four optimizers:
      - SGD (plain)                  §8.3.1
      - SGD + Momentum (α=0.9)       §8.3.2
      - SGD + Nesterov (α=0.9)       §8.3.3
      - Adam                         §8.5.3

    SGD variants use linear LR decay (§8.3.1, eq. 8.14) to ensure convergence.
    Grad norm curves are saved for each run (§8.2.1, Fig 8.1).
    A combined comparison plot is also generated.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # SGD-based optimizers need a higher LR and linear schedule to converge well
    optimizer_configs = [
        ("SGD",              "sgd",       0.05, True),
        ("SGD + Momentum",   "momentum",  0.05, True),
        ("SGD + Nesterov",   "nesterov",  0.05, True),
        ("Adam",             "adam",      0.001, False),
    ]

    histories = []
    labels = []
    results = []

    for display_name, opt_name, lr, use_sched in optimizer_configs:
        safe_name = f"optim_{opt_name}"
        print(f"\nRunning Experiment 5 — Optimizer: {display_name}")

        train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)
        model = MLP(use_batchnorm=True, use_dropout=False)

        model, history = train_model(
            model,
            train_loader,
            val_loader,
            device=device,
            epochs=20,
            lr=lr,
            patience=5,
            weight_decay=1e-4,
            optimizer_name=opt_name,
            use_lr_schedule=use_sched,
        )

        # Per-optimizer plots
        plot_history(history, model_name=safe_name)
        plot_grad_norm(history, model_name=safe_name)

        # Evaluate on test set
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels_batch in test_loader:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                y_true.extend(labels_batch.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        histories.append(history)
        labels.append(display_name)
        results.append({
            "model": display_name,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "best_val_acc": max(history["val_acc"]),
            "epochs_ran": len(history["val_acc"]),
        })

    # Combined comparison plot
    plot_optimizer_comparison(histories, labels, filename="optimizer_comparison")

    return results


def main():
    experiments = [
        ("baseline_mlp", False, False, 0.001),
    ("batchnorm_mlp", True, False, 0.001),
    ("dropout_mlp", False, True, 0.001),
    ("batchnorm_dropout_mlp", True, True, 0.001),

    ("baseline_high_lr", False, False, 0.01),
    ("batchnorm_high_lr", True, False, 0.01),
    ("dropout_high_lr", False, True, 0.01),
    ("batchnorm_dropout_high_lr", True, True, 0.01),

    ("baseline_very_high_lr", False, False, 0.05),
    ("batchnorm_very_high_lr", True, False, 0.05),
    ("dropout_very_high_lr", False, True, 0.05),
    ("batchnorm_dropout_very_high_lr", True, True, 0.05),
    ]

    results = []

    for name, use_bn, use_do, lr in experiments:
        print(f"\nRunning experiment: {name}")
        result = run_single_experiment(
            name,
            use_batchnorm=use_bn,
            use_dropout=use_do,
            lr=lr
        )
        results.append(result)

    # --- Experiment 5: Optimizer Comparison (Chapter 8) ---
    print("\n" + "=" * 80)
    print("Experiment 5 — Optimizer Comparison (Chapter 8: §8.3, §8.5)")
    print("=" * 80)
    optimizer_results = run_optimizer_experiment()
    results.extend(optimizer_results)

    print("\nFinal Results")
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

