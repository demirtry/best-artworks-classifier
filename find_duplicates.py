import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import normalize
from collections import defaultdict


def find_similar_artworks(artist_name, threshold_low=0.9, threshold_up=1., max_images_per_group=10):
    """
    Находит и визуализирует группы похожих изображений для заданного художника

    Параметры:
    artist_name (str): Имя папки художника
    threshold (float): Порог косинусной схожести (0.0-1.0)
    max_images_per_group (int): Макс. изображений в одной визуализации
    max_groups_to_show (int): Макс. количество групп для отображения
    """
    # Путь к данным
    data_root = "data/images/images"
    artist_path = os.path.join(data_root, artist_name)

    if not os.path.exists(artist_path):
        print(f"Папка художника не найдена: {artist_path}")
        return

    # Загрузка изображений
    image_files = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    for file in os.listdir(artist_path):
        if any(file.lower().endswith(ext) for ext in valid_extensions):
            image_files.append(file)

    if not image_files:
        print(f"Не найдено изображений для художника: {artist_name}")
        return

    print(f"Найдено {len(image_files)} изображений для художника: {artist_name}")

    # Трансформации для изображений
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Загрузка предобученной модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])  # Удаляем последний слой
    model = model.to(device).eval()
    print("Модель ResNet50 загружена и переведена в режим оценки")

    # Извлечение признаков
    features = []
    valid_images = []

    print("\nИзвлечение признаков изображений...")
    with torch.no_grad():
        for i, file in enumerate(image_files):
            try:
                img_path = os.path.join(artist_path, file)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                feature = model(img_tensor).flatten()
                feature = normalize(feature.unsqueeze(0)).squeeze()
                features.append(feature.cpu())
                valid_images.append((i, file))  # Сохраняем индекс и имя файла
                if (i + 1) % 50 == 0:
                    print(f"Обработано {i + 1}/{len(image_files)} изображений")
            except Exception as e:
                print(f"Ошибка обработки {file}: {e}")

    if not features:
        print("Не удалось извлечь признаки для изображений")
        return

    features = torch.stack(features)
    print(f"Успешно извлечены признаки для {len(features)} изображений")

    # Вычисление матрицы схожести
    print("\nВычисление матрицы схожести...")
    similarity_matrix = torch.mm(features, features.T)

    # Поиск групп схожих изображений
    print("\nПоиск групп схожих изображений...")
    visited = set()
    groups = []
    similarity_dict = defaultdict(list)

    for i in range(similarity_matrix.shape[0]):
        if i in visited:
            continue

        # Находим все изображения, схожие с текущим
        group = [j for j in range(similarity_matrix.shape[0])
                 if threshold_low <= similarity_matrix[i, j] <= threshold_up]

        if len(group) > 1:  # Только группы с 2+ изображениями
            groups.append(group)
            visited.update(group)

            # Сохраняем информацию для вывода
            for j in group:
                if i != j:
                    similarity_dict[i].append((j, similarity_matrix[i, j].item()))

    print(f"\nРезультаты анализа для {artist_name}:")
    print(f"Всего изображений: {len(image_files)}")
    print(f"Найдено групп схожих изображений: {len(groups)}")
    print(f"Порог схожести: {threshold_low}")

    # Выводим информацию о группах
    if not groups:
        print("Не найдено групп схожих изображений")
        return

    # Сортируем группы по размеру (от больших к маленьким)
    groups.sort(key=len, reverse=True)

    print("\nДетали групп:")
    for group_idx, group in enumerate(groups):
        group_size = len(group)
        avg_similarity = sum(similarity_matrix[group[0], j].item() for j in group[1:]) / (group_size - 1)

        print(f"\nГруппа {group_idx + 1}:")
        print(f"  Размер: {group_size} изображений")
        print(f"  Средняя схожесть с центром: {avg_similarity:.4f}")
        print(f"  Индексы изображений: {group}")
        print("  Файлы изображений:")
        for img_idx in group:
            print(f"    [{img_idx}] {valid_images[img_idx][1]}")

    # Визуализация групп
    print("\nВизуализация групп...")
    for group_idx, group in enumerate(groups):
        group_size = len(group)
        plt.figure(figsize=(15, 5))
        plt.suptitle(f"Группа {group_idx + 1} | Художник: {artist_name} | Изображений: {group_size}", fontsize=14)

        # Ограничиваем количество изображений в группе
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


vincent = 'Vincent_van_Gogh'
degah = 'Edgar_Degas'
# Пример использования
find_similar_artworks('Albrecht_Du╠Иrer', threshold_low=0.90, threshold_up=1., max_images_per_group=100)
