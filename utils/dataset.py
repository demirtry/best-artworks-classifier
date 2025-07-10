import logging
import os
import random
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from tqdm import tqdm


class ClassConditionalDataset(Dataset):
    def __init__(self,
                 root: str,
                 base_transform,
                 augment_transform,
                 classes_to_augment: set[int]):
        """
        :param root: Путь к корню ImageFolder
        :param base_transform: Трансформации, общие для всех экземпляров
        :param augment_transform: Дополнительно применимые трансформации для выбранных классов
        :param classes_to_augment: Список именклассов, к которым нужно добавлять augment_transform
        """
        self.folder = datasets.ImageFolder(root, transform=None)
        self.base_tf = base_transform
        self.aug_tf = augment_transform
        self.aug_cls = classes_to_augment

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        path, label = self.folder.samples[idx]
        img = self.folder.loader(path)

        img = self.base_tf(img)
        if label in self.aug_cls:
            img = self.aug_tf(img)

        return img, label


def get_loaders(
        train_dir: str,
        test_dir: str,
        pic_size=(512,512),
        batch_size=64,
        num_workers=4,
        classes_to_augment=None
):
    base_tf = transforms.Compose([
        transforms.Resize(pic_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    aug_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    if classes_to_augment is None:
        classes_to_augment = set()
    else:
        classes_to_augment = set(classes_to_augment)

    train_ds = ClassConditionalDataset(
        root=train_dir,
        base_transform=base_tf,
        augment_transform=aug_tf,
        classes_to_augment=classes_to_augment
    )
    test_ds  = datasets.ImageFolder(
        root=test_dir,
        transform=transforms.Compose([
            transforms.Resize(pic_size),
            transforms.CenterCrop(pic_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406),
                                 std=(0.229,0.224,0.225))
        ])
    )

    logging.info('Initializing dataloaders')
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    logging.info('Train dataloader initialized')
    test_loader = DataLoader(test_ds,  batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)
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
