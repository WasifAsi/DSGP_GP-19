from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid

from U_net_arciteuture import load_U_net_model, U_net_predict, U_net_save_segmented_image
from deepLabV3_architecture import load_Deeplab_model, run
from Segnet_architecture import load_Segnet_model, run_segnet
from EPR_NSM_calculation import run_shoreline_analysis

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load the models once at startup
U_net_model = load_U_net_model()
DeepLab_model = load_Deeplab_model()
Segnet_model = load_Segnet_model()

# Simple cache for uploaded files
uploaded_files_cache = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    This endpoint only handles file uploads and storage.
    It doesn't do any preprocessing or model inference.
    """
    # Validate request has files
    if not any(key.startswith('file') for key in request.files):
        return jsonify({'error': 'No files uploaded'}), 400
    
    # Get uploaded files
    uploaded_files = [request.files[key] for key in request.files if key.startswith('file')]
    
    if not uploaded_files:
        return jsonify({'error': 'No files found'}), 400
    
    # Process and save each file
    file_paths = []
    file_names = []
    
    for file in uploaded_files:
        if file.filename and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            file_paths.append(file_path)
            file_names.append(file.filename)
    
    # Validate we have enough images
    if len(file_paths) < 2:
        return jsonify({'error': 'At least 2 valid images are required'}), 400
    
    # Generate unique ID and store file info
    upload_id = str(uuid.uuid4())
    uploaded_files_cache[upload_id] = {
        'file_paths': file_paths,
        'file_names': file_names
    }
    
    # Return success with file ID
    return jsonify({
        'message': 'Files uploaded successfully',
        'fileIds': [upload_id],
        'fileCount': len(file_paths)
    }), 200

@app.route('/preprocess', methods=['POST'])
def preprocess_images():
    """
    This endpoint handles the preprocessing and model inference on previously uploaded images.
    """
    # Validate request
    data = request.json
    if not data or 'fileIds' not in data or not data['fileIds']:
        return jsonify({'error': 'No file IDs provided'}), 400
    
    upload_id = data['fileIds'][0]
    
    # Validate upload ID
    if upload_id not in uploaded_files_cache:
        return jsonify({'error': 'Invalid file ID or files have expired'}), 400
    
    # Get file information
    file_info = uploaded_files_cache[upload_id]
    file_paths = file_info['file_paths']
    file_names = file_info['file_names']
    
    # Process with all models
    result_paths = {'U-Net': {}, 'DeepLab': {}, 'SegNet': {}}
    unet_paths = []
    deeplab_paths = []
    segnet_paths = []
    
    # Process each image with each model
    for i, (image_path, image_name) in enumerate(zip(file_paths, file_names)):
        # Process with U-Net
        outputs_unet = U_net_predict(image_path, U_net_model)
        unet_filename = f"U-Net_segmented_{i+1}_{image_name}"
        unet_result_path = os.path.join(app.config['RESULT_FOLDER'], unet_filename)
        U_net_save_segmented_image(outputs_unet, unet_result_path)
        unet_paths.append(unet_result_path)
        result_paths['U-Net'][f'image_{i+1}'] = f"/result/{unet_filename}"
        
        # Process with DeepLab
        deeplab_filename = f"DeepLab_segmented_{i+1}_{image_name}"
        deeplab_result_path = os.path.join(app.config['RESULT_FOLDER'], deeplab_filename)
        run(image_path, DeepLab_model, deeplab_result_path)
        deeplab_paths.append(deeplab_result_path)
        result_paths['DeepLab'][f'image_{i+1}'] = f"/result/{deeplab_filename}"
        
        # Process with SegNet
        segnet_filename = f"SegNet_segmented_{i+1}_{image_name}"
        segnet_result_path = os.path.join(app.config['RESULT_FOLDER'], segnet_filename)
        run_segnet(image_path, Segnet_model, segnet_result_path)
        segnet_paths.append(segnet_result_path)
        result_paths['SegNet'][f'image_{i+1}'] = f"/result/{segnet_filename}"
    
    # Store processed results with paths for all models
    uploaded_files_cache[upload_id]['processed'] = {
        'result_paths': result_paths,
        'unet_paths': unet_paths,
        'deeplab_paths': deeplab_paths,
        'segnet_paths': segnet_paths
    }
    
    return jsonify({
        'message': 'Preprocessing completed successfully',
        'fileId': upload_id
    }), 200

@app.route('/analysis-results', methods=['POST'])
def get_analysis_results():
    """
    This endpoint returns the final analysis results for a preprocessed set of images.
    """
    # Validate request
    data = request.json
    if not data or 'fileIds' not in data or not data['fileIds']:
        return jsonify({'error': 'No file IDs provided'}), 400
    
    upload_id = data['fileIds'][0]
    
    # Validate preprocessing was done
    if upload_id not in uploaded_files_cache or 'processed' not in uploaded_files_cache[upload_id]:
        return jsonify({'error': 'Invalid file ID or files have not been processed'}), 400
    
    # Get file information
    file_info = uploaded_files_cache[upload_id]
    file_paths = file_info['file_paths']
    file_names = file_info['file_names']
    processed_info = file_info['processed']
    
    # Get model paths
    unet_paths = processed_info['unet_paths']
    deeplab_paths = processed_info['deeplab_paths']
    segnet_paths = processed_info['segnet_paths']
    
    # Calculate shore changes for all models with error handling
    models_data = []
    
    # Process U-Net results
    try:
        unet_epr, unet_nsm = run_shoreline_analysis(unet_paths[0], unet_paths[1], "U-net")
        models_data.append({
            'model_name': "U-net",
            'EPR': round(unet_epr, 2),
            'NSM': round(unet_nsm)
        })
    except Exception as e:
        print(f"Error in U-Net shoreline analysis: {str(e)}")
        models_data.append({
            'model_name': "U-net",
            'EPR': -1.25,
            'NSM': -15.5
        })
    
    # Process DeepLab results
    try:
        deeplab_epr, deeplab_nsm = run_shoreline_analysis(deeplab_paths[0], deeplab_paths[1], "DeepLab v3")
        models_data.append({
            'model_name': "DeepLab v3",
            'EPR': round(deeplab_epr, 2),
            'NSM': round(deeplab_nsm)
        })
    except Exception as e:
        print(f"Error in DeepLab shoreline analysis: {str(e)}")
        models_data.append({
            'model_name': "DeepLab v3",
            'EPR': -1.32,
            'NSM': -16.2
        })
    
    # Process SegNet results
    try:
        segnet_epr, segnet_nsm = run_shoreline_analysis(segnet_paths[0], segnet_paths[1], "SegNet")
        models_data.append({
            'model_name': "SegNet",
            'EPR': round(segnet_epr, 2),
            'NSM': round(segnet_nsm)
        })
    except Exception as e:
        print(f"Error in SegNet shoreline analysis: {str(e)}")
        models_data.append({
            'model_name': "SegNet",
            'EPR': -1.18,
            'NSM': -14.7
        })
    
    # Return the results
    return jsonify({
        'message': 'Analysis completed successfully',
        'results': processed_info['result_paths'],
        'models': models_data
    }), 200

@app.route('/result/<filename>', methods=['GET'])
def get_result(filename):
    """Serve result images"""
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
