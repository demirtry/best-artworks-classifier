import csv
import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import get_loaders, calculate_class_weights
from utils.visualization_utils import plot_confusion_matrix
from utils.losses import SoftF1Loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_one_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        use_combined_loss: bool = False,
        alpha: float = 0.7
) -> Tuple[float, float, float]:
    """
    Trains the model for one epoch.
    :param model: Model to train
    :param data_loader: Training data loader
    :param criterion: Main loss function
    :param optimizer: Optimizer for model parameters
    :param device: Device to use for training (cuda or cpu)
    :param use_combined_loss: Whether to combine main loss with CrossEntropy loss
    :param alpha: Weight for CrossEntropy loss when using combined loss
    :return: Tuple of (average loss, accuracy, confidence) for the epoch
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    total_confidence = 0.0
    ce_loss_fn = nn.CrossEntropyLoss()

    for inputs, labels in tqdm(data_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        if use_combined_loss:
            main_loss = criterion(outputs, labels)

            ce_loss = ce_loss_fn(outputs, labels)

            loss = alpha * ce_loss + (1 - alpha) * main_loss
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        probs = torch.softmax(outputs, dim=1)
        confidence = probs.max(dim=1).values
        batch_confidence = confidence.sum().item()

        _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == labels).sum().item()
        total_samples += inputs.size(0)
        total_confidence += batch_confidence

    torch.cuda.empty_cache()

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    epoch_confidence = total_confidence / total_samples

    tqdm.write(f"train_Loss: {epoch_loss:.4f} train_Acc: {epoch_acc:.4f} train_Conf: {epoch_confidence:.4f}")

    return epoch_loss, epoch_acc, epoch_confidence


def validate_one_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        use_combined_loss: bool = False,
        alpha: float = 0.7
) -> Tuple[float, float, float]:
    """
    Validates the model for one epoch.
    :param model: Model to validate
    :param data_loader: Validation data loader
    :param criterion: Main loss function
    :param device: Device to use for validation (cuda or cpu)
    :param use_combined_loss: Whether to combine main loss with CrossEntropy loss
    :param alpha: Weight for CrossEntropy loss when using combined loss
    :return: Tuple of (average loss, accuracy, confidence) for the epoch
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    total_confidence = 0.0
    ce_loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if use_combined_loss:
                main_loss = criterion(outputs, labels)

                ce_loss = ce_loss_fn(outputs, labels)

                loss = alpha * ce_loss + (1 - alpha) * main_loss
            else:
                loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            confidence = probs.max(dim=1).values
            batch_confidence = confidence.sum().item()

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()
            total_samples += inputs.size(0)
            total_confidence += batch_confidence

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    epoch_confidence = total_confidence / total_samples

    tqdm.write(f"test_Loss: {epoch_loss:.4f} test_Acc: {epoch_acc:.4f} test_Conf: {epoch_confidence:.4f}")

    return epoch_loss, epoch_acc, epoch_confidence


def save_checkpoint(state: Dict, filename: str) -> None:
    """
    Saves a model checkpoint to disk.
    :param state: Dictionary containing model and training state
    :param filename: Path to save the checkpoint
    """
    torch.save(state, filename)
    logger.info(f"Checkpoint saved to {filename}")


def start_training(
        model: nn.Module,
        device: torch.device,
        train_path: str,
        test_path: str,
        pic_size: Tuple[int, int] = (512, 512),
        epochs: int = 2,
        batch_size: int = 32,
        num_classes: int = 50,
        augmentation: bool = True,
        save_path: str = 'best.pth'
) -> None:
    """
    Main training loop that trains and validates the model, saves checkpoints,
    and logs metrics.
    :param model: Model to train
    :param device: Device to use for training (cuda or cpu)
    :param train_path: Path to training dataset
    :param test_path: Path to test dataset
    :param pic_size: Image size for resizing
    :param epochs: Number of training epochs
    :param batch_size: Batch size for training and validation
    :param num_classes: Number of output classes
    :param augmentation: Whether to apply augmentation to minority classes
    :param save_path: Path to save the best model checkpoint
    """
    minority_classes = []
    if augmentation:
        minority_classes = [
            3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25,
            26, 27, 28, 29, 31, 32, 34, 37, 39, 40, 41, 44, 47, 49
        ]

    train_loader, val_loader = get_loaders(
        train_dir=train_path,
        test_dir=test_path,
        batch_size=batch_size,
        pic_size=pic_size,
        classes_to_augment=minority_classes,
    )

    if hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    model = model.to(device)

    class_weights = calculate_class_weights(train_path, device=device)
    # criterion = SoftF1Loss(class_weights=class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    metrics_file = "training_metrics.csv"
    best_metrics = {
        'epoch': 0,
        'train_loss': float('inf'),
        'val_loss': float('inf'),
        'train_acc': 0.0,
        'val_acc': 0.0,
        'train_conf': 0.0,
        'val_conf': 0.0,
        'best_model_path': save_path
    }

    with open(metrics_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_conf', 'val_conf'])

    for epoch in range(1, epochs + 1):
        tqdm.write(f"Epoch {epoch}/{epochs}")
        train_loss, train_acc, train_conf = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_combined_loss=False
        )

        val_loss, val_acc, val_conf = validate_one_epoch(
            model, val_loader, criterion, device, use_combined_loss=False
        )

        with open(metrics_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc, train_conf, val_conf])

        if val_acc > best_metrics['val_acc']:
            best_metrics.update({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_conf': train_conf,
                'val_conf': val_conf
            })

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': val_acc,
                'val_conf': val_conf
            }
            save_checkpoint(checkpoint, save_path)

    best_metrics_file = "best_metrics.csv"
    file_exists = os.path.isfile(best_metrics_file)

    with open(best_metrics_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=best_metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(best_metrics)

    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    y_pred, y_true = evaluate_model(model, val_loader, device)
    cm_path = f"/kaggle/working/confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, cm_path)


def evaluate_model(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates the model on the given dataset.
    :param model: Trained model to evaluate
    :param dataloader: Data loader for evaluation
    :param device: Device to use for evaluation
    :return: Tuple of (predictions, true labels)
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            all_preds.append(preds.argmax(dim=1).cpu())
            all_targets.append(y.cpu())

    return torch.cat(all_preds), torch.cat(all_targets)
