# Shoreline Movement Analysis Backend

## Description
The **Shoreline Movement Analysis Backend** is a Flask-based web application that integrates a PyTorch U-net, DeepLabV3+ models and Tensorflow Segnet model and FCN-8 model to perform shoreline segmentation. This backend handles image uploads, preprocesses the images, runs the segmentation model, and saves and returns the predicted results to the frontend.

The project is designed to assist in shoreline movement analysis, helping researchers and environmentalists detect changes in shorelines over time through automated segmentation.

## Features
- **Image Upload**: Users can upload images to the backend.
- **Shoreline Segmentation**: The backend processes uploaded images and performs shoreline segmentation using the models.
- **Preprocessing**: Images undergo necessary preprocessing before being passed through the model for prediction.
- **Prediction Saving**: The segmented output is saved and returned to the user.
- **Model Integration**: The model is integrated into Flask for easy access and usage via HTTP requests.
- **AI Chat Bot**: An integrated chat bot assists users with questions about shoreline analysis, model performance, and system usage.

## Directory Structure
```
/shoreline-movement-analysis-backend
│
├── Flask_connection.py         # Main Flask application file
├── Models/                     # Directory to store model files
│   ├── best-200-epoch-base-unet.pt    
│   ├── final_shoreline_deeplabv3.pt
│   ├── segnet_model.h5
│   ├── FCN8.pth
│  
├── U_net_arciteuture.py            # model arciteutures
├── Segnet_architecture.py
├── deepLabV3_architecture.py
├── FCN8_arciteuture.py
│  
│  
├── EPR_NSM_calculation.py     # EPR and NSM Calculation
├── shoreline_validator.py     # check the image is a shoreline or not
│   
└── README.md


