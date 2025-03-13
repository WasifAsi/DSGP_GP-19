import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model architecture classes
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super().__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # Initial conv layers with reduced stride
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # ResNet-like blocks with modified strides
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=1)

        # ASPP
        self.aspp = ASPP(512, [6, 12, 18])  # Reduced dilation rates

        # Low-level features conversion
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        # Auxiliary decoder
        self.aux_decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 1, 1)
        )

        # Main decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        input_size = x.size()[-2:]

        # Initial convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # Backbone
        low_level_feat = self.layer1(x)
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        x = self.layer4(x)

        # Auxiliary output
        aux_out = self.aux_decoder(x)
        aux_out = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=False)

        # ASPP
        x = self.aspp(x)

        # Decoder
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)
        low_level_feat = self.low_level_conv(low_level_feat)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.decoder(x)

        # Final upsampling
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        if self.training:
            return x, aux_out
        return x

def get_preprocessing_transform(image_size=540):
    """Create preprocessing transform for images"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def predict_single_image(model, image_path, device, image_size=540, threshold=0.5):
    """Make prediction for a single image"""
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get transform
    transform = get_preprocessing_transform(image_size)
    
    # Apply transforms
    transformed = transform(image=image)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    # Process prediction
    prediction = (torch.sigmoid(output) > threshold).float()
    prediction = prediction.cpu().numpy().squeeze()
    
    
    return image, prediction

def predict_batch(model, image_paths, device, image_size=540, threshold=0.3):
    """Make predictions for multiple images"""
    model.eval()
    transform = get_preprocessing_transform(image_size)
    results = []
    
    for img_path in image_paths:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transformed = transform(image=image)
        input_tensor = transformed['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        prediction = (torch.sigmoid(output) > threshold).float()
        prediction = prediction.cpu().numpy().squeeze()
    
    return results


def load_Deeplab_model():
    model_path = 'Model/final_shoreline_deeplabv3.pt'  # Update with your model path

    # Initialize model
    model = DeepLabV3Plus(num_classes=1)

    try:
        # Load saved weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()  # Set to evaluation mode
        print("Model loaded and ready for inference.")
    except Exception as e:
        print(f"Error loading the model: {e}")

    return model

def run(image_path, model, output_path=None):
    # Make prediction
    image, prediction = predict_single_image(model, image_path, device)

    # Save the prediction if an output path is provided
    if output_path:
        save_deeplab_segmented_image(prediction, output_path)

    print("Prediction completed and visualized")
    return prediction


def save_deeplab_segmented_image(prediction, output_path):
    """
    Save the predicted segmented image.
    
    Parameters:
        prediction (np.array): The predicted image as a NumPy array (binary mask).
        output_path (str): The path where the segmented image should be saved.
    """
    # Ensure the prediction is in the correct format for saving (uint8)
    prediction_img = (prediction * 255).astype(np.uint8)

    # Save the prediction image
    cv2.imwrite(output_path, prediction_img)
    print(f"Deeplab segmented image saved to: {output_path}")

