import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


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


def get_loaders(train_dir: str, test_dir: str, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    target_size = (512, 512)
    train_dataset = CustomImageDataset(root_dir=train_dir, transform=transform, target_size=target_size)
    test_dataset = CustomImageDataset(root_dir=test_dir, transform=transform, target_size=target_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader