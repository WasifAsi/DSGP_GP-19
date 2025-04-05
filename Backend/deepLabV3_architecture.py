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

# Helper function to ensure group normalization works correctly
def get_groups(channels, target_groups=32):
    # Find the largest divisor of channels that is <= target_groups
    for i in range(target_groups, 0, -1):
        if channels % i == 0:
            return i
    return 1  # Fallback to 1 group (which is always valid)

# Model architecture classes
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_gn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        if use_gn:
            self.norm1 = nn.GroupNorm(get_groups(out_channels), out_channels)
        else:
            self.norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        if use_gn:
            self.norm2 = nn.GroupNorm(get_groups(out_channels), out_channels)
        else:
            self.norm2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if use_gn:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.GroupNorm(get_groups(out_channels), out_channels)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, use_gn=False):
        if use_gn:
            super().__init__(
                nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
                nn.GroupNorm(get_groups(out_channels), out_channels),
                nn.ReLU()
            )
        else:
            super().__init__(
                nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels, use_gn=False):
        super().__init__()
        if use_gn:
            self.pooling = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(get_groups(out_channels), out_channels),
                nn.ReLU()
            )
        else:
            self.pooling = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.pooling(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256, use_gn=False):
        super().__init__()
        modules = []
        if use_gn:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(get_groups(out_channels), out_channels),
                nn.ReLU()))
        else:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate, use_gn))

        modules.append(ASPPPooling(in_channels, out_channels, use_gn))

        self.convs = nn.ModuleList(modules)

        if use_gn:
            self.project = nn.Sequential(
                nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
                nn.GroupNorm(get_groups(out_channels), out_channels),
                nn.ReLU(),
                nn.Dropout(0.5))
        else:
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
    def __init__(self, num_classes=1, use_gn=True):
        super().__init__()

        # Initial conv layers with reduced stride
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Using stride=2 to reduce memory
        if use_gn:
            self.norm1 = nn.GroupNorm(get_groups(64), 64)
        else:
            self.norm1 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Reduce memory with stride=2

        # ResNet-like blocks with modified strides for memory efficiency
        self.layer1 = self._make_layer(64, 64, 3, stride=1, use_gn=use_gn)
        self.layer2 = self._make_layer(64, 128, 4, stride=2, use_gn=use_gn)
        self.layer3 = self._make_layer(128, 256, 6, stride=2, use_gn=use_gn)
        self.layer4 = self._make_layer(256, 512, 3, stride=1, use_gn=use_gn)

        # Memory-efficient ASPP
        self.aspp = ASPP(512, [3, 6, 9], use_gn=use_gn)  # Reduced dilation rates

        # Low-level features conversion
        if use_gn:
            self.low_level_conv = nn.Sequential(
                nn.Conv2d(64, 48, 1, bias=False),
                nn.GroupNorm(get_groups(48), 48),
                nn.ReLU()
            )
        else:
            self.low_level_conv = nn.Sequential(
                nn.Conv2d(64, 48, 1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU()
            )

        # Auxiliary decoder
        if use_gn:
            self.aux_decoder = nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1),
                nn.GroupNorm(get_groups(256), 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv2d(256, 1, 1)
            )
        else:
            self.aux_decoder = nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv2d(256, 1, 1)
            )

        # Main decoder
        if use_gn:
            self.decoder = nn.Sequential(
                nn.Conv2d(304, 256, 3, padding=1, bias=False),
                nn.GroupNorm(get_groups(256), 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv2d(256, 256, 3, padding=1, bias=False),
                nn.GroupNorm(get_groups(256), 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv2d(256, num_classes, 1)
            )
        else:
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

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, use_gn=False):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride, use_gn=use_gn))
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels, use_gn=use_gn))
        return nn.Sequential(*layers)

    def forward(self, x):
        input_size = x.size()[-2:]

        # Initial convolutions
        x = F.relu(self.norm1(self.conv1(x)))
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


def get_preprocessing_transform( image_size=1024):
    return A.Compose([
        A.CenterCrop(height=image_size, width=image_size),  # Center crop to square
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def preprocesss_deeplabv3(image_path):
    """Make prediction for a single image"""
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

# def predict_batch(model, image_paths, device, image_size=540, threshold=0.3):
#     """Make predictions for multiple images"""
#     model.eval()
#     transform = get_preprocessing_transform(image_size)
#     results = []
    
#     for img_path in image_paths:
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         transformed = transform(image=image)
#         input_tensor = transformed['image'].unsqueeze(0).to(device)
        
#         with torch.no_grad():
#             output = model(input_tensor)
        
#         prediction = (torch.sigmoid(output) > threshold).float()
#         prediction = prediction.cpu().numpy().squeeze()
    
#     return results


def load_Deeplab_model():
    model_path = 'Model/best_shoreline_deeplabv3_new.pt'  # Update with your model path

    # Initialize model
    model = DeepLabV3Plus(num_classes=1)
    
    # Get device information first
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Load saved weights with appropriate device mapping
        if torch.cuda.is_available():
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location='cpu')
            
        model.load_state_dict(state_dict)
        model = model.to(device)  # Make sure to move model to device
        model.eval()  # Set to evaluation mode
        print(f"DeepLabV3+ model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading the model: {e}")

    return model, device



def run_deeplabv3(image, model, output_path=None, threshold=0.5):
    # Get the device from the model
    device = next(model.parameters()).device
    
    # Get transform
    transform = get_preprocessing_transform()
    
    # Apply transforms
    transformed = transform(image=image)
    input_tensor = transformed['image'].unsqueeze(0).to(device)  # Ensure tensor is on same device as model
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    # Process prediction
    prediction = (torch.sigmoid(output) > threshold).float()

    # Save the prediction if an output path is provided
    if output_path:
        save_deeplab_segmented_image(prediction, output_path)

    print("Prediction completed and visualized")
    


def save_deeplab_segmented_image(prediction, output_path):
    """
    Save the predicted segmented image.
    
    Parameters:
        prediction (np.array): The predicted image as a NumPy array (binary mask).
        output_path (str): The path where the segmented image should be saved.
    """

    prediction = prediction.cpu().numpy().squeeze()

    # Ensure the prediction is in the correct format for saving (uint8)
    prediction_img = (prediction * 255).astype(np.uint8)

    # Save the prediction image
    # cv2.imwrite(output_path, prediction_img)

    Image.fromarray(prediction_img).convert("L").save(output_path, format="PNG")

    print(f"Deeplab segmented image saved to: {output_path}")



def crop_to_square(image):
    height, width = image.shape[:2]

    # Ensure the image is at least 1024x1024
    if height < 1024 or width < 1024:
        raise ValueError("Image dimensions must be at least 1024x1024")

    # Calculate center crop box
    left = (width - 1024) // 2
    top = (height - 1024) // 2
    right = left + 1024
    bottom = top + 1024

    return image[top:bottom, left:right]