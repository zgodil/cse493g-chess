"""Image transforms for training and validation."""
from torchvision import transforms

# ImageNet statistics used for ViT-B/16 pretraining
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])
