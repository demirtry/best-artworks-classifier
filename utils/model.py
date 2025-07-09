import logging

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.dataset import get_loaders, calculate_class_weights
from utils.losses import LabelSmoothingLoss, TripletLoss
from utils.visualization_utils import plot_training_history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super(CustomModel, self).__init__()
        self.base = base_model
        if hasattr(base_model, 'fc'):
            self.feature_dim = base_model.fc.in_features
            self.base.fc = nn.Identity()
        else:
            self.feature_dim = base_model.classifier.in_features
            self.base.classifier = nn.Identity()
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        features = self.base(x)
        logits = self.classifier(features)
        return logits, features  # Возвращаем и логиты, и эмбеддинги


def train_one_epoch(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        criterion_ce: nn.Module,
        criterion_triplet: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        alpha: float = 0.5
) -> tuple[float, float]:
    """
    Выполняет один цикл обучения по всем батчам.
    Возвращает средний loss и accuracy за эпоху.

    Args:
        model: Обучаемая модель.
        data_loader: DataLoader для тренировочных данных.
        criterion_ce: Функция потерь для классификации (например, LabelSmoothingLoss).
        criterion_triplet: Функция триплетного лосса.
        optimizer: Оптимизатор.
        device: Устройство ('cuda' или 'cpu').
        alpha: Коэффициент для объединения двух лоссов (class + triplet).

    Returns:
        epoch_loss: float — среднее значение лосса за эпоху.
        epoch_acc: float — доля правильных предсказаний.
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, labels in tqdm(data_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        logits, embeddings = model(inputs)
        loss_ce = criterion_ce(logits, labels)
        loss_triplet = criterion_triplet(embeddings, labels)
        loss = loss_ce + alpha * loss_triplet

        loss.backward()
        optimizer.step()

        _, preds = torch.max(logits, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == labels).sum().item()
        total_samples += inputs.size(0)

    torch.cuda.empty_cache()

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc


def validate_one_epoch(model: nn.Module,
                       data_loader: torch.utils.data.DataLoader,
                       criterion_ce: nn.Module,
                       criterion_triplet: nn.Module,
                       device: torch.device,
                       alpha: float = 0.5) -> tuple[float, float]:
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
            logits, embeddings = model(inputs)
            loss_ce = criterion_ce(logits, labels)
            loss_triplet = criterion_triplet(embeddings, labels)
            loss = loss_ce + alpha * loss_triplet

            _, preds = torch.max(logits, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()
            total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc


def save_checkpoint(state: dict, filename: str) -> None:
    """
    Сохраняет checkpoint модели на диск.
    """
    torch.save(state, filename)
    logger.info(f"Checkpoint saved to {filename}")


def start_training(
        base_model: nn.Module,
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

    model = CustomModel(base_model, num_classes=num_classes)
    model = model.to(device)

    class_weights = calculate_class_weights(train_path, device=device)
    criterion_ce = LabelSmoothingLoss(num_classes=num_classes, smoothing=0.1, weights=class_weights)
    criterion_triplet = TripletLoss(margin=1.0)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    logger.info(f"start training")
    for epoch in range(1, epochs + 1):
        logger.info(f"Epoch {epoch}/{epochs}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion_ce, criterion_triplet, optimizer, device
        )
        logger.info(f"train_Loss: {train_loss:.4f} train_Acc: {train_acc:.4f}")

        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion_ce, criterion_triplet, device
        )
        logger.info(f"test_Loss: {val_loss:.4f} test_Acc: {val_acc:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': base_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }
            save_checkpoint(checkpoint, save_path)

    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")

    plot_training_history(
        epochs=epochs,
        train_losses=train_losses,
        val_losses=val_losses,
        train_accuracies=train_accs,
        val_accuracies=val_accs,
        path='plots/training_history.png'
    )
