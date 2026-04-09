import copy
import torch
import torch.nn as nn


def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    return avg_loss, acc


def build_optimizer(model, optimizer_name, lr, weight_decay):
    """
    Build an optimizer by name.
    Supported: 'adam', 'sgd', 'momentum', 'nesterov'
    Covers §8.3.1 (SGD), §8.3.2 (Momentum), §8.3.3 (Nesterov), §8.5.3 (Adam).
    """
    name = optimizer_name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "momentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == "nesterov":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Choose from: adam, sgd, momentum, nesterov")


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=20,
    lr=0.001,
    patience=5,
    weight_decay=1e-4,
    optimizer_name="adam",
    use_lr_schedule=False,
):
    """
    Train a model with the specified optimizer and optional linear LR decay.

    Args:
        optimizer_name: One of 'adam', 'sgd', 'momentum', 'nesterov' (§8.3–8.5)
        use_lr_schedule: If True, apply linear LR decay from lr to lr*0.01 (§8.3.1, eq. 8.14)

    History includes grad_norm per epoch for ill-conditioning analysis (§8.2.1, Fig 8.1).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, optimizer_name, lr, weight_decay)

    # Linear LR schedule: decay from lr to lr*0.01 over all epochs (§8.3.1, eq. 8.14)
    if use_lr_schedule:
        lr_end = lr * 0.01
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=lr_end / lr,
            total_iters=epochs,
        )
    else:
        scheduler = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "grad_norm": [],   # gradient norm tracking (§8.2.1)
    }

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        epoch_grad_norm = 0.0
        num_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Compute gradient norm before stepping (§8.2.1)
            total_norm_sq = torch.tensor(0.0, device=device)
            for p in model.parameters():
                if p.grad is not None:
                    total_norm_sq += p.grad.data.norm(2) ** 2
            epoch_grad_norm += total_norm_sq.sqrt().item()
            num_batches += 1

            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)
        avg_grad_norm = epoch_grad_norm / num_batches if num_batches > 0 else 0.0

        val_loss, val_acc = evaluate_model(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["grad_norm"].append(avg_grad_norm)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Grad Norm: {avg_grad_norm:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_state)
    return model, history

