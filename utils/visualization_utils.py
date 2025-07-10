import matplotlib.pyplot as plt
import pandas as pd


def dataset_count_visualization(results: list[dict], path: str):
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


def dataset_count_statistics_to_csv(results: list[dict], path: str):
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


def dataset_size_visualization(results: list[dict], path: str):
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
        train_losses: list[float],
        val_losses: list[float],
        train_accuracies: list[float],
        val_accuracies: list[float],
        path: str
):
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


def plot_training_curves_csv(csv_path, save_path):
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


def plot_confusion_matrix(y_true, y_pred, path, title: str = 'Confusion matrix'):
    """
    Строит матрицу ошибок, нормализованную по столбцам (каждый столбец суммируется в 100%).
    Для столбцов без предсказаний выводит 0.0% вместо прочерка.
    """
    import numpy as np
    import matplotlib.pyplot as plt

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
