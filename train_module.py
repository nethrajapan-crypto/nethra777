"""Training utilities and loop for PCB defect classification.

This module is meant to be imported by other scripts (e.g., ``train.py``) to run
training/validation loops and save best checkpoints + metrics.

Example:
    from train_module import run_training

    run_training(num_epochs=10)
"""

import json
import time

import torch

from load_dataset import make_dataloader
from model import create_model
from training_utils import get_loss_and_optimizer


DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(model: torch.nn.Module, loader: torch.utils.data.DataLoader, criterion, optimizer, device: torch.device = DEFAULT_DEVICE):
    """Train the model for a single epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, criterion, device: torch.device = DEFAULT_DEVICE):
    """Evaluate the model on a dataset."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def run_training(
    num_epochs: int = 1,
    checkpoint_path: str = './best_model.pth',
    metrics_path: str = './training_metrics.json',
    device: torch.device = DEFAULT_DEVICE,
):
    """Run the full training loop.

    Returns:
        dict: metrics logged during training (train/val loss/accuracy).
    """
    loaders, sizes = make_dataloader()
    print("Dataset sizes:", sizes)

    model = create_model().to(device)
    criterion, optimizer = get_loss_and_optimizer(model)

    best_val_acc = 0.0
    metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, num_epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, loaders['train'], criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, loaders['val'], criterion, device)

        elapsed = time.time() - start
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)

        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f} "
            f"({elapsed:.1f}s)"
        )

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print("  ✓ new best model saved")

    # save metrics for visualization
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # final test
    test_loss, test_acc = evaluate(model, loaders['test'], criterion, device)
    print(f"Test - loss: {test_loss:.4f}, acc: {test_acc:.4f}")

    return metrics


if __name__ == '__main__':
    run_training()
