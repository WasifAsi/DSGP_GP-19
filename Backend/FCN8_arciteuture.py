import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

def load_FCN8_model():
    model = load_model('Model/fcn8_model.h5', compile=False)
    return model 

# Image Preprocessing Function
def preprocess_image(image_path, target_size=(512, 512)):
    """Load and preprocess image for model prediction."""
    img = load_img(image_path, target_size=target_size)  # Resize image
    img = img_to_array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def run_FCN8(image, model, output_path=None):
    """
    Run FCN8 segmentation model on an input image.
    
    Parameters:
        image_path (str): Path to the input image
        model: The loaded FCN8 model
        output_path (str, optional): Where to save the prediction. If None, no saving.
        
    Returns:
        numpy.ndarray: The predicted mask
    """

    pred_mask = model.predict(image)[0, :, :, 0]  # Get single channel prediction
    prediction = (pred_mask > 0.5).astype(np.uint8) * 255  # Thresholding & scale to 255


    # Remove the batch dimension and get the predicted mask
    predicted_mask = prediction.squeeze()  # This will remove extra dimensions if any

    # If it's a binary mask, apply thresholding
    threshold = 0.5
    predicted_mask = (predicted_mask > threshold).astype(np.uint8) * 255

    # Save the prediction if an output path is provided
    if output_path:
        save_predicted_image(predicted_mask, output_path)
    
    # Return the mask for further processing
    return predicted_mask

def save_predicted_image(prediction, output_path):
    """
    Save the predicted segmented image.
    
    Parameters:
        prediction (np.array): The predicted image as a NumPy array (binary mask).
        output_path (str): The path where the segmented image should be saved.
    """
    # Ensure the prediction is in the correct format for saving (uint8)
    cv2.imwrite(output_path, prediction)
    print(f"Predicted image saved to: {output_path}")