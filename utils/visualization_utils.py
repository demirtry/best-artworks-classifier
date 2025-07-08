import matplotlib.pyplot as plt
import pandas as pd


def dataset_count_visualization(results: list[dict], path: str):
    df = pd.DataFrame(results)
    class_counts = df['class'].value_counts().sort_index()

    class_indices = range(len(class_counts))

    plt.figure(figsize=(12, 6))
    plt.bar(class_indices, class_counts.values, color='orange', edgecolor='black')
    plt.title("Количество изображений по классам")
    plt.xticks(class_indices)  # подписываем только цифры
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
