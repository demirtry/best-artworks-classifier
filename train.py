import torch
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

from utils.model import start_training

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_res_net_50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model_res_net_18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    start_training(
        model=model_res_net_18,
        device=device,
        pic_size=(512, 512),
        batch_size=64,
        train_path='data/data/train',
        test_path='data/data/test',
        save_path='models/best_resnet18.pth'
    )
