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


def dataset__size_visualization(results: list[dict], path: str):
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
        epochs: int, train_losses: list[float], val_losses: list[float],
        train_accuracies: list[float], val_accuracies: list[float], path: str
):
    epochs = range(1, epochs + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Acc')
    plt.plot(epochs, val_accuracies, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()