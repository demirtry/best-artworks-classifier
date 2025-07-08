import logging
import os
import random
import shutil
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from tqdm import tqdm


class CustomImageDataset(Dataset):
    """Кастомный датасет для работы с папками классов"""

    def __init__(self, root_dir, transform=None, target_size=(224, 224)):
        """
        :param root_dir: корневая директория
        :param transform: трансформация
        :param target_size: нужный размер изображений
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size

        self.classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.images = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_names(self):
        """Возвращает список имен классов"""
        return self.classes


def get_loaders(train_dir: str, test_dir: str, pic_size=(512, 512), batch_size=64, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize(pic_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    logging.info('Initializing datasets')
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    logging.info('Train initialized')
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    logging.info('Test initialized')

    logging.info('Initializing dataloaders')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    logging.info('Train dataloader initialized')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logging.info('Test dataloader initialized')

    return train_loader, test_loader


def split_dataset(
        source_dir: str,
        train_dir: str,
        test_dir: str,
        split_ratio: float = 0.8,
        seed: int = 42
):
    """
    Разделяет датасет на train и test, сохраняя структуру классов.

    Args:
        source_dir (str): Путь к исходной директории с классами.
        train_dir (str): Путь для сохранения train части.
        test_dir (str): Путь для сохранения test части.
        split_ratio (float): Доля данных для train (от 0 до 1).
        seed (int): Сид для воспроизводимости случайного разделения.
    """
    random.seed(seed)

    source_path = Path(source_dir)
    train_path = Path(train_dir)
    test_path = Path(test_dir)

    if train_path.exists():
        shutil.rmtree(train_path)
    if test_path.exists():
        shutil.rmtree(test_path)

    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    classes = [d.name for d in source_path.iterdir() if d.is_dir()]

    print(f"Найдено классов: {len(classes)}")
    print(f"Разделение данных (train: {split_ratio * 100}%, test: {(1 - split_ratio) * 100}%)")

    for cls in tqdm(classes, desc="Обработка классов", total=len(classes)):
        cls_source = source_path / cls
        cls_train = train_path / cls
        cls_test = test_path / cls

        cls_train.mkdir(parents=True, exist_ok=True)
        cls_test.mkdir(parents=True, exist_ok=True)

        files = [f for f in cls_source.iterdir() if f.is_file()]
        random.shuffle(files)

        split_idx = int(len(files) * split_ratio)
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        for f in train_files:
            shutil.copy2(f, cls_train)

        for f in test_files:
            shutil.copy2(f, cls_test)

    print("Данные успешно разделены на train и test.")


def calculate_class_weights(dir_path: str, device: torch.device):

    class_counts = []
    for cls in sorted(os.listdir(dir_path)):
        cls_dir = os.path.join(dir_path, cls)
        n = len(os.listdir(cls_dir))
        class_counts.append(n)

    class_counts = torch.tensor(class_counts, dtype=torch.float, device=device)
    class_weights = 1.0 / (class_counts + 1e-8)

    return class_weights