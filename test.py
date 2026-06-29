"""
galaxy_models.py

Reusable models and utilities for galaxy morphology regression/classification.

Contains:
- Center weight mask generation
- GalaxyConvNeXt (regression head, ugriz magnitudes)
- GalaxyEfficientNet (morphology classification)
- Image preprocessing
- Prediction utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Utility: Center Weight Mask
def generate_center_weight_mask(size: int = 7, sigma: float = 1.5) -> torch.Tensor:
    """
    Creates a Gaussian mask emphasizing the center of the feature map.

    Args:
        size (int): size of the mask (size x size)
        sigma (float): standard deviation of Gaussian

    Returns:
        torch.Tensor: mask of shape (size, size)
    """
    x = np.arange(size)
    y = np.arange(size)
    x_grid, y_grid = np.meshgrid(x, y)
    center = (size - 1) / 2
    gaussian = np.exp(-((x_grid - center)**2 + (y_grid - center)**2) / (2 * sigma**2))
    return torch.tensor(gaussian, dtype=torch.float32)


class GalaxyConvNeXt(nn.Module):
    """
    ConvNeXt-based regression model for ugriz magnitudes.
    """

    def __init__(self, device: torch.device, base_mask_size: int = 7, sigma: float = 1.5):
        super().__init__()
        base = models.convnext_base(pretrained=False)
        self.device = device

        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.regression_head = nn.Sequential(
            nn.Linear(base.classifier[2].in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 5)
        )

        # Register center-weight mask
        mask = generate_center_weight_mask(base_mask_size, sigma).unsqueeze(0).unsqueeze(0)
        self.register_buffer("center_mask", mask.to(device))

    def forward(self, x):
        features = self.features(x)
        H, W = features.shape[2], features.shape[3]
        mask_resized = F.interpolate(self.center_mask, size=(H, W), mode='bilinear', align_corners=False)
        weighted = features * mask_resized
        pooled = self.avgpool(weighted)
        pooled = torch.flatten(pooled, 1)
        return self.regression_head(pooled)


class GalaxyEfficientNet4(nn.Module):
    """EfficientNet-B0 classifier for 4 galaxy morphology outputs"""
    def __init__(self, device: torch.device, base_mask_size: int = 7, sigma: float = 1.5):
        super().__init__()
        base = models.efficientnet_b0(pretrained=True)
        self.device = device

        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classification_head = nn.Sequential(
            nn.Linear(base.classifier[1].in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )

        mask = generate_center_weight_mask(base_mask_size, sigma).unsqueeze(0).unsqueeze(0)
        self.register_buffer("center_mask", mask.to(device))

    def forward(self, x):
        features = self.features(x)
        H, W = features.shape[2], features.shape[3]
        mask_resized = F.interpolate(self.center_mask, size=(H, W), mode='bilinear', align_corners=False)
        weighted = features * mask_resized
        pooled = self.avgpool(weighted)
        pooled = torch.flatten(pooled, 1)
        return self.classification_head(pooled)


class GalaxyEfficientNet6(nn.Module):
    """EfficientNet-B0 classifier for 6 galaxy morphology outputs"""
    def __init__(self, device: torch.device, base_mask_size: int = 7, sigma: float = 1.5):
        super().__init__()
        base = models.efficientnet_b0(pretrained=True)
        self.device = device

        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classification_head = nn.Sequential(
            nn.Linear(base.classifier[1].in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 6),
            nn.Sigmoid()
        )

        mask = generate_center_weight_mask(base_mask_size, sigma).unsqueeze(0).unsqueeze(0)
        self.register_buffer("center_mask", mask.to(device))

    def forward(self, x):
        features = self.features(x)
        H, W = features.shape[2], features.shape[3]
        mask_resized = F.interpolate(self.center_mask, size=(H, W), mode='bilinear', align_corners=False)
        weighted = features * mask_resized
        pooled = self.avgpool(weighted)
        pooled = torch.flatten(pooled, 1)
        return self.classification_head(pooled)


def load_and_preprocess_image(image_path: str, size: int = 256) -> torch.Tensor:
    """
    Loads an image and applies standard preprocessing.

    Args:
        image_path (str): path to image
        size (int): output size for resizing

    Returns:
        torch.Tensor: preprocessed image tensor of shape (1, 3, size, size)
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)
    

class StarGalaxyEffNet(nn.Module):
    """EfficientNet-B0 classifier for Star vs Galaxy"""
    def __init__(self, num_outputs=2):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(base.classifier[1].in_features, num_outputs)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Prediction Utility
def predict_galaxies(checkpoint_path: str, img_paths: list[str], device: torch.device = None, model_class=None, **model_kwargs):
    """
    Load a trained model and predict galaxy outputs for a list of images.

    Args:
        checkpoint_path (str): path to model checkpoint
        img_paths (list[str]): list of image paths
        device (torch.device, optional): device for inference
        model_class (nn.Module): model class to instantiate
        model_kwargs (dict): additional args for the model (e.g., num_outputs)

    Returns:
        list[dict]: predictions per image
    """
    if device is None:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Initialize model
    model = model_class(device=device, **model_kwargs).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Filter checkpoint keys to match current model
    state_dict = checkpoint['model_state_dict']
    filtered_state = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict({**model.state_dict(), **filtered_state})

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    results = []
    with torch.inference_mode():
        for img_path in img_paths:
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            output = model(img).squeeze(0).cpu().numpy()
            results.append({'image_path': img_path, **{f'out_{i}': float(v) for i, v in enumerate(output)}})

    return results
