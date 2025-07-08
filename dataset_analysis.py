from pathlib import Path
from PIL import Image
from utils.visualization_utils import dataset_count_visualization, dataset__size_visualization
import shutil


def analyze_dataset(
        data_dir: str = 'data/images/images',
        count_save_path: str = 'plots/dataset_count_analysis.png',
        size_save_path: str = 'plots/dataset_size_analysis.png'
):
    """
    Строит графики распределения количества и размеров изображений
    :param data_dir: Директория с изображениями
    :param count_save_path: Путь для сохранения графика количества изображений по классам
    :param size_save_path: Путь для сохранения графика размеров изображений
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


    dataset_count_visualization(image_info, path=count_save_path)
    dataset__size_visualization(image_info, path=size_save_path)


def organize_by_class(source_dir: Path, target_base: Path):
    """разделяет resized датасет на папки по классам"""
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
