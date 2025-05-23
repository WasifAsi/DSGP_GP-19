import cv2
import numpy as np

from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def load_Segnet_model():
    model = load_model('Model/segnet_model.h5', compile=False)
    return model 


# Function to load images and preprocess
def load_and_preprocess_image(image_path):
    # Read the image from file
    image = cv2.imread(image_path)

    # If image is read properly, preprocess it
    if image is not None:
        # Resize the image to 540x540 (OpenCV uses (width, height) format)
        image_resized = cv2.resize(image, (540, 540))
        return image_resized
    else:
        raise ValueError(f"Error: Could not load image from {image_path}")
    
    
def run_segnet(image_resized, model, output_path=None):
    """
    Run SegNet model prediction on a preprocessed image.
    
    Parameters:
        image_resized (np.array): The preprocessed input image
        model: The loaded SegNet model
        output_path (str, optional): Path to save the prediction output
        
    Returns:
        np.array: The predicted segmentation mask
    """
    # Normalize the image (scaling pixel values to 0-1 range)
    image_normalized = image_resized / 255.0

    # Add batch dimension (1, 540, 540, 3)
    image_batch = np.expand_dims(image_normalized, axis=0)

    # Prediction
    prediction = model.predict(image_batch)

    # Remove the batch dimension and get the predicted mask
    predicted_mask = prediction.squeeze()  # This will remove the (1, height, width, channels) dimension

    # If it's a binary mask, apply thresholding
    threshold = 0.5
    predicted_mask = (predicted_mask > threshold).astype(np.uint8) * 255

    # Save the prediction if an output path is provided
    if output_path:
        save_predicted_image(predicted_mask, output_path)
    
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