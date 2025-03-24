import cv2
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

def load_FCN8_model():
    model = load_model('Model/fcn8_model.h5', compile=False)
    return model 

# Add this utility function to ensure consistent image handling
def ensure_numpy_array(image):
    """
    Convert different image types to NumPy arrays.
    
    Parameters:
        image: Input image (PIL Image, OpenCV image, or NumPy array)
        
    Returns:
        numpy.ndarray: Image as a NumPy array
    """
    # Check if it's already a NumPy array
    if isinstance(image, np.ndarray):
        return image
    
    # Check if it's a PIL Image
    if isinstance(image, Image.Image):
        return img_to_array(image)
    
    # If it's something else we can't handle
    print(f"Warning: Unknown image type {type(image)}")
    try:
        # Try to convert to NumPy array
        return np.array(image)
    except:
        raise TypeError(f"Cannot convert {type(image)} to NumPy array")

# Image Preprocessing Function
def preprocess_image(image_path, target_size=(512, 512)):
    """
    Load and preprocess image for model prediction.
    
    Parameters:
        image_path (str): Path to the input image
        target_size (tuple): Target size for resizing (width, height)
        
    Returns:
        numpy.ndarray: Preprocessed image as NumPy array
    """
    # Load image with PIL
    img = load_img(image_path, target_size=target_size)
    # Convert to NumPy array immediately
    return ensure_numpy_array(img)

def run_FCN8(img, model, output_path=None):
    """
    Run FCN8 segmentation model on an input image.
    
    Parameters:
        img: The preprocessed input image (can be PIL Image, OpenCV image, or NumPy array)
        model: The loaded FCN8 model
        output_path (str, optional): Where to save the prediction. If None, no saving.
        
    Returns:
        numpy.ndarray: The predicted mask
    """
    # Make sure the image is a NumPy array
    img = ensure_numpy_array(img)
    
    # Normalize and add batch dimension
    img_normalized = img / 255.0  # Normalize
    img_batch = np.expand_dims(img_normalized, axis=0)  # Add batch dimension
    
    # Get prediction from model
    pred_mask = model.predict(img_batch)[0, :, :, 0]  # Get single channel prediction
    
    # Thresholding & scale to 255
    predicted_mask = (pred_mask > 0.5).astype(np.uint8) * 255

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
    # Ensure prediction is a numpy array
    prediction = ensure_numpy_array(prediction)
    
    # Print shape for debugging
    print(f"Prediction shape before saving: {prediction.shape}")
    
    # Ensure the prediction is in the correct format for saving
    if len(prediction.shape) == 2:
        # It's a single-channel grayscale image - this is what we want
        cv2.imwrite(output_path, prediction)
    elif len(prediction.shape) == 3 and prediction.shape[2] == 3:
        # It's a 3-channel color image
        cv2.imwrite(output_path, prediction)
    elif len(prediction.shape) == 3 and prediction.shape[2] == 1:
        # It's a single-channel image with extra dimension
        cv2.imwrite(output_path, prediction.squeeze())
    else:
        # Handle unusual cases
        print(f"Warning: Unusual prediction shape {prediction.shape}")
        reshaped = prediction.squeeze()  # Try to squeeze any extra dimensions
        if len(reshaped.shape) <= 2:  # If we got it down to 1 or 2 dimensions
            cv2.imwrite(output_path, reshaped)
        else:
            print(f"Error: Cannot save prediction with shape {reshaped.shape}")
            return
    
    print(f"Predicted image saved to: {output_path}")