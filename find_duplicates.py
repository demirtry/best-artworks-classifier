import logging
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.functional import normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_similar_artworks(artist_name, threshold_low=0.9, threshold_up=1., max_images_per_group=10):
    """
    Находит и визуализирует группы похожих изображений для заданного художника

    Параметры:
    artist_name (str): Имя папки художника
    threshold (float): Порог косинусной схожести (0.0-1.0)
    max_images_per_group (int): Макс. изображений в одной визуализации
    max_groups_to_show (int): Макс. количество групп для отображения
    """
    data_root = "data/images/images"
    artist_path = os.path.join(data_root, artist_name)

    if not os.path.exists(artist_path):
        logger.error(f"Папка художника не найдена: {artist_path}")
        return

    image_files = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    for file in os.listdir(artist_path):
        if any(file.lower().endswith(ext) for ext in valid_extensions):
            image_files.append(file)

    if not image_files:
        logger.error(f"Не найдено изображений для художника: {artist_name}")
        return

    logger.info(f"Найдено {len(image_files)} изображений для художника: {artist_name}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используемое устройство: {device}")

    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model = model.to(device).eval()
    logger.info("Модель ResNet50 загружена и переведена в режим оценки")

    features = []
    valid_images = []

    with torch.no_grad():
        for i, file in enumerate(image_files):
            try:
                img_path = os.path.join(artist_path, file)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                feature = model(img_tensor).flatten()
                feature = normalize(feature.unsqueeze(0)).squeeze()
                features.append(feature.cpu())
                valid_images.append((i, file))
                if (i + 1) % 50 == 0:
                    logger.info(f"Обработано {i + 1}/{len(image_files)} изображений")
            except Exception as e:
                logger.warning(f"Ошибка обработки {file}: {e}")

    if not features:
        logger.error("Не удалось извлечь признаки для изображений")
        return

    features = torch.stack(features)
    logger.info(f"Извлечены признаки для {len(features)} изображений")

    logger.info("Вычисление матрицы схожести...")
    similarity_matrix = torch.mm(features, features.T)

    logger.info("Нормализация матрицы схожести...")
    visited = set()
    groups = []
    similarity_dict = defaultdict(list)

    for i in range(similarity_matrix.shape[0]):
        if i in visited:
            continue

        group = [j for j in range(similarity_matrix.shape[0])
                 if threshold_low <= similarity_matrix[i, j] <= threshold_up]

        if len(group) > 1:
            groups.append(group)
            visited.update(group)

            for j in group:
                if i != j:
                    similarity_dict[i].append((j, similarity_matrix[i, j].item()))

    logger.info(f"\nРезультаты анализа для {artist_name}:\n"
                f"Всего изображений: {len(image_files)}\n"
                f"Найдено групп схожих изображений: {len(groups)}\n"
                f"Порог схожести: {threshold_low}")

    if not groups:
        logger.info("Не найдено групп схожих изображений")
        return

    groups.sort(key=len, reverse=True)

    logger.info("\nДетали групп:")
    for group_idx, group in enumerate(groups):
        group_size = len(group)
        avg_similarity = sum(similarity_matrix[group[0], j].item() for j in group[1:]) / (group_size - 1)

        logger.info(f"Группа {group_idx + 1}:")
        logger.info(f"  Размер: {group_size} изображений")
        logger.info(f"  Средняя схожесть с центром: {avg_similarity:.4f}")
        logger.info(f"  Индексы изображений: {group}")
        logger.info(f"  Файлы изображений:")
        for img_idx in group:
            logger.info(f"    [{img_idx}] {valid_images[img_idx][1]}")

    logger.info("\nВизуализация групп...")
    for group_idx, group in enumerate(groups):
        group_size = len(group)
        plt.figure(figsize=(15, 5))
        plt.suptitle(f"Группа {group_idx + 1} | Художник: {artist_name} | Изображений: {group_size}", fontsize=14)

        display_group = group[:max_images_per_group]

        for i, img_idx in enumerate(display_group):
            img_path = os.path.join(artist_path, valid_images[img_idx][1])
            img_idx_to_plot = int(img_path.split('_')[-1].split('.')[0])
            img = Image.open(img_path)
            plt.subplot(1, len(display_group), i + 1)
            plt.imshow(img)
            plt.title(f"Индекс: {img_idx_to_plot}\nСхожесть: {similarity_matrix[group[0], img_idx]:.2f}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    vincent = 'Vincent_van_Gogh'
    degah = 'Edgar_Degas'
    find_similar_artworks(vincent, threshold_low=0.85, threshold_up=1., max_images_per_group=100)
