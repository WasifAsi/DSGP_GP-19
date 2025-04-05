import os
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Global variables to store models
autoencoder = None
encoder = None
ocsvm = None
models_loaded = False

def load_shoreline_models():
    global autoencoder, encoder, ocsvm, models_loaded
    
    if models_loaded:
        return True
    
    try:
        # Load the autoencoder model
        autoencoder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shorline_validater_models/autoencoder.h5')
        autoencoder = load_model(autoencoder_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
        print("\nAutoencoder loaded successfully.")
        
        # Load the encoder model
        encoder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shorline_validater_models/encoder.h5')
        encoder = load_model(encoder_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
        print("Encoder loaded successfully.")
        
        # Load the One-Class SVM model
        ocsvm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shorline_validater_models/ocsvm_model.pkl')
        with open(ocsvm_path, "rb") as f:
            ocsvm = pickle.load(f)
        print("One-Class SVM loaded successfully.")
        
        models_loaded = True
        return True
    
    except Exception as e:
        print(f"Error loading shoreline validation models: {str(e)}")
        print("Make sure autoencoder.h5, encoder.h5, and ocsvm_model.pkl are in the current directory")
        return False




def preprocess_image_for_validation(image_path, target_size=(128, 128)):
    try:
        # Load the image
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
        
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None





def is_shoreline(image_path):
    """
    Check if an image contains a shoreline
    
    Args:
        image_path: Path to the image file
        
    Returns:
        bool: True if the image contains a shoreline, False otherwise
    """
    global autoencoder, encoder, ocsvm, models_loaded
    
    if not models_loaded:
        success = load_shoreline_models()
        if not success:
            print("Models not loaded - assuming image is valid")
            return True
    
    try:
        # Preprocess the image
        test_img = preprocess_image_for_validation(image_path)
        if test_img is None:
            # If preprocessing fails, assume it's not valid
            return False
        
        # Extract latent features using the encoder
        test_encoded = encoder.predict(test_img)
        
        # Flatten the features if necessary
        test_flattened = test_encoded.reshape(test_encoded.shape[0], -1)
        
        # Predict using the One-Class SVM
        test_prediction = ocsvm.predict(test_flattened)
        prediction_result = test_prediction[0]  # 1 means similar to shoreline, -1 means anomaly
        
        # Add feedback message
        is_shore = (prediction_result == 1)
        image_name = os.path.basename(image_path)
        
        if is_shore:
            print(f"SHORELINE DETECTED: {image_name}")
        else:
            print(f"NO SHORELINE DETECTED: {image_name} (anomaly)")
            
        return is_shore
    
    except Exception as e:
        print(f"ERROR CHECKING SHORELINE: {os.path.basename(image_path)} - {str(e)}")
        return False