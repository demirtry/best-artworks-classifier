import shutil
from pathlib import Path

from PIL import Image

from utils.visualization_utils import (
    dataset_count_visualization,
    dataset_size_visualization,
    dataset_count_statistics_to_csv
)


def analyze_dataset(
        data_dir: str = 'data/images/images',
        count_save_path: str = 'plots/dataset_count_analysis.png',
        size_save_path: str = 'plots/dataset_size_analysis.png',
        csv_path: str = 'plots/dataset_count_statistics.csv'
) -> None:
    """
    Analyzes a dataset of images and generates plots and statistics.

    - Computes statistics of image counts per class and saves them to a CSV.
    - Plots the count of images per class.
    - Plots the distribution of image sizes.
    :param data_dir: (str) Directory containing subfolders for each class with images.
    :param count_save_path: (str) Path to save the plot of image count per class.
    :param size_save_path: (str) Path to save the histogram of image sizes.
    :param csv_path: (str) Path to save statistics as a CSV file.
    :return: None
    """

    data_dir = Path(data_dir)
    image_info = []

    classes = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    for cls in classes:
        for img_path in cls.glob('*.*'):
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    image_info.append({
                        'class': cls.name,
                        'width': width,
                        'height': height,
                        'area': width * height
                    })
            except Exception as e:
                print(f"Ошибка при обработке {img_path}: {e}")

    dataset_count_statistics_to_csv(image_info, csv_path)
    dataset_count_visualization(image_info, path=count_save_path)
    dataset_size_visualization(image_info, path=size_save_path)


def organize_by_class(source_dir: Path, target_base: Path) -> None:
    """
    Splits a flat dataset of images into subfolders by class.

    Images must be named with the format: <class>_<unique_id>.jpg
    :param source_dir: (Path) Directory containing images (no subfolders).
    :param target_base: (Path) Directory where subfolders per class will be created.
    :return: None
    """
    for img_path in source_dir.glob('*.jpg'):
        stem = img_path.stem
        parts = stem.rsplit('_', 1)
        if len(parts) == 2 and parts[0]:
            class_name = parts[0]
        else:
            class_name = 'unknown'

        class_dir = target_base / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        dest_path = class_dir / img_path.name
        shutil.copy2(img_path, dest_path)


if __name__ == '__main__':
    analyze_dataset()

    src = Path('data/resized/resized')
    dst = Path('data/resized/resized_class')
    organize_by_class(src, dst)

    analyze_dataset(
        data_dir="data/resized/resized_class",
        count_save_path="plots/resized_dataset_count_analysis.png",
        size_save_path="plots/resized_dataset_size_analysis.png"
    )
