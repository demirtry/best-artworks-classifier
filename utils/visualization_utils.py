from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def dataset_count_visualization(results: List[Dict], path: str) -> None:
    """
    Plots a bar chart showing the number of images per class.
    :param results: (List[Dict]) List of dictionaries, each containing at least a 'class' key.
    :param path: (str) Path where the plot image will be saved.
    :return: None
    """
    df = pd.DataFrame(results)
    class_counts = df['class'].value_counts().sort_index()

    class_indices = range(len(class_counts))

    plt.figure(figsize=(12, 6))
    plt.bar(class_indices, class_counts.values, color='orange', edgecolor='black')
    plt.title("Количество изображений по классам")
    plt.xticks(class_indices)
    plt.xlabel("Классы (по индексу)")
    plt.ylabel("Количество")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def dataset_count_statistics_to_csv(results: List[Dict], path: str) -> None:
    """
    Computes class distribution statistics and saves them to a CSV file.

    The statistics include:
    - min count
    - max count
    - mean count
    - total count
    :param results: (List[Dict]) List of dictionaries, each containing at least a 'class' key.
    :param path: (str) Path where the CSV will be saved.
    :return: None
    """
    df = pd.DataFrame(results)

    class_counts = df['class'].value_counts()

    stats = {
        'min': [class_counts.min()],
        'max': [class_counts.max()],
        'mean': [class_counts.mean()],
        'total': [class_counts.sum()]
    }

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(path, index=False)


def dataset_size_visualization(results: List[Dict], path: str) -> None:
    """
    Plots a histogram showing the distribution of image areas.

    Assumes a key 'area' exists in each result dictionary.
    :param results: (List[Dict]) List of dictionaries, each containing at least an 'area' key.
    :param path: (str) Path where the histogram image will be saved.
    :return: None
    """
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 4))
    plt.plot()
    plt.hist(df['area'], bins=30, color='skyblue', edgecolor='black')
    plt.title("распределение размеров изображений")
    plt.xlabel("размер (width × height)")
    plt.ylabel("Количество")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_training_history(
        epochs: int,
        train_losses: List[float],
        val_losses: List[float],
        train_accuracies: List[float],
        val_accuracies: List[float],
        path: str
) -> None:
    """
    Plots training and validation loss and accuracy curves.
    :param epochs: (int) Total number of training epochs.
    :param train_losses: (List[float]) Training loss values per epoch.
    :param val_losses: (List[float]) Validation loss values per epoch.
    :param train_accuracies: (List[float]) Training accuracy values per epoch.
    :param val_accuracies: (List[float]) Validation accuracy values per epoch.
    :param path: (str) Path where the plot image will be saved.
    :return: None
    """
    epochs = range(1, epochs + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Acc', marker='o')
    plt.plot(epochs, val_accuracies, label='Val Acc', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_training_curves_csv(csv_path: str, save_path: str) -> None:
    """
    Plots training curves from a CSV file containing per-epoch metrics.

    Expects columns:
        - 'epoch'
        - 'train_loss'
        - 'val_loss'
        - 'train_acc'
        - 'val_acc'
        - 'train_conf'
        - 'val_conf'
    :param csv_path: (str) Path to the CSV file with training metrics.
    :param save_path: (str) Path where the plot image will be saved.
    :return: None
    """
    metrics = pd.read_csv(csv_path)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(metrics['epoch'], metrics['train_loss'], label='Train Loss', marker='o')
    plt.plot(metrics['epoch'], metrics['val_loss'], label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(metrics['epoch'], metrics['train_acc'], label='Train Acc', marker='o')
    plt.plot(metrics['epoch'], metrics['val_acc'], label='Val Acc', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(metrics['epoch'], metrics['train_conf'], label='Train Conf', marker='o')
    plt.plot(metrics['epoch'], metrics['val_conf'], label='Val Conf', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Confidence')
    plt.title('Training & Validation Confidence')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)


def plot_confusion_matrix(
        y_true: List[int],
        y_pred: List[int],
        path: str,
        title: str = 'Confusion matrix'
) -> None:
    """
    Plots a confusion matrix normalized by columns, so that each column sums to 100%.

    For columns with zero predictions, displays 0.0% instead of empty values.
    :param y_true: (List[int]) True class labels.
    :param y_pred: (List[int]) Predicted class labels.
    :param path: (str) Path where the confusion matrix plot will be saved.
    :param title: (str, optional) Title of the plot. Defaults to 'Confusion matrix'.
    :return: None
    """
    classes_names = [str(i) for i in range(50)]
    num_classes = len(classes_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    col_sums = cm.sum(axis=0, keepdims=True)
    col_sums_fixed = np.where(col_sums == 0, 1, col_sums)

    cm_percent = cm / col_sums_fixed * 100

    fig, ax = plt.subplots(figsize=(40, 30))
    im = ax.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes_names, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes_names)

    thresh = cm_percent.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            display = f"{cm_percent[i, j]:.1f}"
            ax.text(j, i, display,
                    ha='center', va='center',
                    color='white' if cm_percent[i, j] > thresh else 'black')

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
