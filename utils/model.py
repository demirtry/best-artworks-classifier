import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.dataset import get_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_one_epoch(model: nn.Module,
                    data_loader: torch.utils.data.DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> tuple[float, float]:
    """
    Выполняет один цикл обучения по всем батчам.
    Возвращает средний loss и accuracy за эпоху.
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, labels in tqdm(data_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == labels).sum().item()
        total_samples += inputs.size(0)

    torch.cuda.empty_cache()

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc


def validate_one_epoch(model: nn.Module,
                       data_loader: torch.utils.data.DataLoader,
                       criterion: nn.Module,
                       device: torch.device) -> tuple[float, float]:
    """
    Один проход валидации: усредненный loss и accuracy.
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()
            total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    logger.info(f"[Valid] epoch_loss: {epoch_loss:.4f}  epoch_acc: {epoch_acc:.4f}")

    return epoch_loss, epoch_acc


def save_checkpoint(state: dict, filename: str) -> None:
    """
    Сохраняет checkpoint модели на диск.
    """
    torch.save(state, filename)
    logger.info(f"Checkpoint saved to {filename}")


def start_training(
        model: nn.Module,
        device: torch.device,
        train_path: str,
        test_path: str,
        epochs: int = 2,
        batch_size: int = 32,
        num_classes: int = 50,
        save_path: str = 'best.pth'
) -> None:
    """
    Основной цикл дообучения модели с разделением на train/val,
    использованием выделенных функций и сохранением лучшей модели.
    """
    train_loader, val_loader = get_loaders(
        train_dir=train_path,
        test_dir=test_path,
        batch_size=batch_size
    )

    if hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    model = model.to(device)

    try:
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.cpu().numpy())

        class_counts = np.bincount(all_labels, minlength=num_classes)
        class_weights = 1. / (class_counts + 1e-8)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    except:
        logger.warning("Не удалось рассчитать веса классов, используем стандартные веса")
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        logger.info(f"Epoch {epoch}/{epochs}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        logger.info(f"[Train] Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )
        logger.info(f"[Test] Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }
            save_checkpoint(checkpoint, save_path)

    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
