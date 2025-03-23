import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import joblib

class FeatureExtractor(nn.Module):
    """
    Feature extractor model based on ResNet18 that outputs a 512-dimensional feature vector
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Load a pretrained ResNet18 model
        resnet = models.resnet18(pretrained=True)
        # Remove the final fully connected layer to obtain feature vector
        self.features = nn.Sequential(*list(resnet.children())[:-1])
    
    def forward(self, x):
        x = self.features(x)          # shape: [batch_size, 512, 1, 1]
        x = x.view(x.size(0), -1)     # flatten to: [batch_size, 512]
        return x

# Initialize global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
shoreline_feature_extractor = None
oc_svm = None
shoreline_transform = None

def load_shoreline_models():
    """
    Loads the feature extractor and SVM models for shoreline validation
    """
    global shoreline_feature_extractor, oc_svm, shoreline_transform
    
    # Define the directory for model files
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shorline validater models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize the feature extractor
    shoreline_feature_extractor = FeatureExtractor().to(device)
    
    # Model paths
    feature_extractor_path = os.path.join(model_dir, 'feature_extractor.pth')
    svm_path = os.path.join(model_dir, 'one_class_svm_model.pkl')
    
    # Check if model files exist
    if not os.path.exists(feature_extractor_path) or not os.path.exists(svm_path):
        raise FileNotFoundError(f"Shoreline validation model files not found. Make sure the files are in the {model_dir} directory.")
    
    # Load the models
    shoreline_feature_extractor.load_state_dict(torch.load(feature_extractor_path, map_location=device))
    shoreline_feature_extractor.eval()
    
    # Load the One-Class SVM model
    oc_svm = joblib.load(svm_path)
    
    # Define the transform pipeline
    shoreline_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    print("Shoreline validation models loaded successfully")

def is_shoreline(image_path):
    """
    Given an image path, checks if it contains a shoreline
    Returns True for shoreline, False otherwise
    """
    try:
        if shoreline_feature_extractor is None or oc_svm is None or shoreline_transform is None:
            load_shoreline_models()
            
        # Open and convert the image
        pil_image = Image.open(image_path).convert("RGB")
        
        # Apply transformation and add batch dimension
        img_tensor = shoreline_transform(pil_image).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            feat = shoreline_feature_extractor(img_tensor)
            
        # Convert to numpy for the SVM
        feat = feat.cpu().numpy()
        
        # Predict using SVM (+1 for shoreline, -1 for non-shoreline)
        pred = oc_svm.predict(feat)
        
        # Add feedback message
        is_shore = (pred[0] == 1)
        image_name = os.path.basename(image_path)
        
        if is_shore:
            # Simple one-line message that will appear in your Flask logs
            print(f"SHORELINE DETECTED: {image_name}")
        else:
            print(f"NO SHORELINE DETECTED: {image_name}")
            
        return is_shore
    except Exception as e:
        print(f"ERROR CHECKING SHORELINE: {os.path.basename(image_path)} - {str(e)}")
        return False