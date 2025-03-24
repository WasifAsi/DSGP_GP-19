import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import os 

# Define U-Net Model
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.encoder1 = self.double_conv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = self.double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = self.double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = self.double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.middle = self.double_conv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder1 = self.double_conv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = self.double_conv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = self.double_conv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder4 = self.double_conv(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1_pool = self.pool1(enc1)

        enc2 = self.encoder2(enc1_pool)
        enc2_pool = self.pool2(enc2)

        enc3 = self.encoder3(enc2_pool)
        enc3_pool = self.pool3(enc3)

        enc4 = self.encoder4(enc3_pool)
        enc4_pool = self.pool4(enc4)

        middle = self.middle(enc4_pool)

        dec1 = self.up1(middle)
        dec1 = torch.cat([enc4, dec1], dim=1)
        dec1 = self.decoder1(dec1)

        dec2 = self.up2(dec1)
        dec2 = torch.cat([enc3, dec2], dim=1)
        dec2 = self.decoder2(dec2)

        dec3 = self.up3(dec2)
        dec3 = torch.cat([enc2, dec3], dim=1)
        dec3 = self.decoder3(dec3)

        dec4 = self.up4(dec3)
        dec4 = torch.cat([enc1, dec4], dim=1)
        dec4 = self.decoder4(dec4)

        output = self.out_conv(dec4)
        return output

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


# Define transformations                                                    #define ur one
transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)


# U-Net model
def load_U_net_model():
    model_name = "Model/best-200-epoch-base-unet.pt"  # Model Name Saved one

    model = UNet(3, 2)  # 3 input channels (RGB), 2 output classes
    model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
    model.eval()

    return model

# Crop to square function 
def crop_to_square(image):
    height, width = image.shape[:2]  # Get image dimensions
    min_dim = min(height, width)  # Find the shortest side

    # Calculate center crop box
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    return image[top:bottom, left:right]  # Crop and return


def U_net_preprocess_image(image_path):
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load {image_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_to_square(img)  # Assuming crop_to_square is defined elsewhere
    img = cv2.resize(img, (512, 512))
    return img 



def U_net_predict(img, model):

    image_tensor= transform(np.array(img)) 
    if image_tensor is None:
        return {}

    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0)).squeeze(0)  # Add batch dimension and remove it after
        output = (output > 0.5).float()  # Apply threshold to get binary output

    return output




def U_net_save_segmented_image(output, save_path):
    # Squeeze the output tensor and convert to numpy array
    output = output.squeeze().cpu().detach().numpy()
    # Select the positive class if there are two channels
    output = output[1] if output.shape[0] == 2 else output
    # Convert to binary mask (0 or 255)
    output = (output > 0.5).astype(np.uint8) * 255

    # Ensure the output is 2D (in case of any multi-dimensional output)
    if len(output.shape) != 2:
        output = output.reshape(output.shape[1], output.shape[2])

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the image as a PNG (PIL automatically handles the format based on the extension)
    Image.fromarray(output).convert("L").save(save_path, format="PNG")  # Save as PNG


if __name__ == "__main__":
    # Load the model
    U_net_model = load_U_net_model()
    # Load the image
    image_path = "sentinel2_void_2023-11-23_ArugamBay.jpg"
    output = U_net_predict(image_path, U_net_model)
    # Save the segmented image
    save_path = "Result/4_segmented.png"
    U_net_save_segmented_image(output, save_path)
    print(f"Segmented image saved at: {save_path}")