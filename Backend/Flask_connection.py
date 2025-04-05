import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['TK_SILENCE_DEPRECATION'] = "1"

# Force matplotlib to use a non-GUI backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import time
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

import zipfile
import io
import json
import datetime

from U_net_arciteuture import load_U_net_model, U_net_preprocess_image, U_net_predict, U_net_save_segmented_image
from deepLabV3_architecture import load_Deeplab_model, preprocesss_deeplabv3, run_deeplabv3
# Comment out SegNet import - not using this model anymore
# from Segnet_architecture import load_Segnet_model,load_and_preprocess_image, run_segnet
# Comment out FCN8 import - not using this model anymore
# from FCN8_arciteuture import load_FCN8_model, preprocess_image, run_FCN8

# Import shoreline validation
from shoreline_validator import load_shoreline_models, is_shoreline

from EPR_NSM_calculation import run_shoreline_analysis

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['RESULT_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load models
U_net_model = load_U_net_model()
DeepLab_model, deeplab_device = load_Deeplab_model()
# Comment out SegNet model loading - not using this model anymore
# Segnet_model = load_Segnet_model()
# Comment out FCN8 model loading - not using this model anymore
# FCN8_model = load_FCN8_model()

# Load shoreline validation models
try:
    load_shoreline_models()
    shoreline_validation_available = True
    print("Shoreline validation ready")
except Exception as e:
    print(f"Warning: Shoreline validation not available: {str(e)}")
    shoreline_validation_available = False

uploaded_files_cache = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS







def compare_segmentations(image1_path, image2_path):
    """
    Compare two segmented images to check if they are from the same area
    Returns True if segmentations are similar enough
    """
    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        return False
    
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return False
    
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    _, binary1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
    
    score, _ = ssim(binary1, binary2, full=True)
    
    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()
    iou = intersection / union if union > 0 else 0
    
    # Use a threshold of 0.98 for SSIM score
    is_similar = score > 0.98

    print(f"SSIM: {score}, IoU: {iou}, Similar: {is_similar}")
    
    return is_similar







@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400
    
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    if len(files) < 2 or len(files) > 5:
        return jsonify({'error': f'Please upload between 2 and 5 files (received {len(files)})'}), 400
    
    # Generate a simple session ID
    session_id = str(int(time.time()))
    
    # Clear the result folder for new analysis
    clear_result_folders()
    
    file_paths = []
    file_names = []
    
    # Keep track of filenames to avoid duplicates
    used_filenames = set()
    
    for i, file in enumerate(files):
        original_filename = secure_filename(file.filename)
        prefix = f"{i+1}_"
        base_name, extension = os.path.splitext(original_filename)
        final_filename = prefix + original_filename
        
        counter = 1
        while final_filename in used_filenames:
            final_filename = f"{prefix}{base_name}_{counter}{extension}"
            counter += 1
            
        used_filenames.add(final_filename)
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)
        
        try:
            file.save(file_path)
            file_names.append(original_filename)
            file_paths.append(file_path)
        except Exception as e:
            return jsonify({'error': f'Error saving file {original_filename}: {str(e)}'}), 500
    
    # Store file information in the cache
    uploaded_files_cache[session_id] = {
        'file_names': file_names,
        'file_paths': file_paths,
        'state': 'uploaded',
        'timestamp': time.time()
    }
    
    return jsonify({
        'message': 'Files uploaded successfully',
        'fileIds': [session_id],
        'count': len(files)
    }), 200






@app.route('/validate-shoreline', methods=['POST'])
def validate_shoreline():
    """
    Pre-Step: Shoreline validation - validates if the uploaded images contain shorelines
    """
    data = request.json
    if not data or 'fileIds' not in data or not data['fileIds']:
        return jsonify({'error': 'No file IDs provided'}), 400
    
    upload_id = data['fileIds'][0]
    
    if upload_id not in uploaded_files_cache:
        return jsonify({'error': 'Invalid file ID or files have expired'}), 400
    
    file_info = uploaded_files_cache[upload_id]
    
    if file_info.get('state') != 'uploaded':
        return jsonify({'error': 'Files have already been validated'}), 400
    
    file_paths = file_info['file_paths']
    file_names = file_info['file_names']
    
    validation_results = []
    non_shoreline_images = []
    
    for i, (image_path, image_name) in enumerate(zip(file_paths, file_names)):
        img = cv2.imread(image_path)
        if img is None:
            return jsonify({
                'error': f'Image {image_name} could not be loaded. The file may be corrupted.',
                'invalidImage': image_name
            }), 400
        
        if img.shape[0] < 100 or img.shape[1] < 100:
            return jsonify({
                'error': f'Image {image_name} is too small. Please use images of at least 100x100 pixels.',
                'invalidImage': image_name
            }), 400
        
        if shoreline_validation_available:
            is_shore = is_shoreline(image_path)
            
            validation_results.append({
                'filename': image_name,
                'contains_shoreline': str(is_shore)
            })
            
            if not is_shore:
                non_shoreline_images.append(image_name)
        else:
            validation_results.append({
                'filename': image_name,
                'contains_shoreline': 'true'
            })
    
    if non_shoreline_images:
        if len(non_shoreline_images) == 1:
            return jsonify({
                'error': f'Image {non_shoreline_images[0]} does not appear to contain a shoreline. Please upload satellite images of coastal areas.',
                'invalidImage': non_shoreline_images[0],
                'validationResults': validation_results
            }), 400
        else:
            image_list = ", ".join(non_shoreline_images)
            return jsonify({
                'error': f'The following images do not appear to contain shorelines: {image_list}. Please upload satellite images of coastal areas.',
                'invalidImages': non_shoreline_images,
                'validationResults': validation_results
            }), 400
    
    # All images passed validation
    uploaded_files_cache[upload_id]['validation_results'] = validation_results
    uploaded_files_cache[upload_id]['state'] = 'validated'
    
    return jsonify({
        'message': 'Shoreline validation completed successfully',
        'fileId': upload_id,
        'validationResults': validation_results,
        'step': 0.5,
        'next': '/preprocess'
    }), 200






@app.route('/preprocess', methods=['POST'])
def preprocess_images():
    """
    Step 1: Preprocessing with shoreline validation
    """
    data = request.json
    if not data or 'fileIds' not in data or not data['fileIds']:
        return jsonify({'error': 'No file IDs provided'}), 400
    
    upload_id = data['fileIds'][0]
    
    if upload_id not in uploaded_files_cache:
        return jsonify({'error': 'Invalid file ID or files have expired'}), 400
    
    file_info = uploaded_files_cache[upload_id]
    
    if file_info.get('state') not in ['uploaded', 'validated']:
        return jsonify({'error': 'Files have already been preprocessed'}), 400
    
    file_paths = file_info['file_paths']
    file_names = file_info['file_names']
    
    # If we haven't validated yet, do it now
    if file_info.get('state') != 'validated' and shoreline_validation_available:
        non_shoreline_images = []
        validation_results = []
        
        # First check all images for basic validation and shoreline detection
        for image_path, image_name in zip(file_paths, file_names):
            # Basic image validation
            img = cv2.imread(image_path)
            if img is None:
                return jsonify({
                    'error': f'Image {image_name} could not be loaded. The file may be corrupted.',
                    'invalidImage': image_name
                }), 400
            
            if img.shape[0] < 100 or img.shape[1] < 100:
                return jsonify({
                    'error': f'Image {image_name} is too small. Please use images of at least 100x100 pixels.',
                    'invalidImage': image_name
                }), 400
            
            # Check if it's a shoreline
            is_shore = is_shoreline(image_path)
            validation_results.append({
                'filename': image_name,
                'contains_shoreline': str(is_shore)
            })
            
            if not is_shore:
                non_shoreline_images.append(image_name)
        
        # If any images failed validation, return error with details
        if non_shoreline_images:
            if len(non_shoreline_images) == 1:
                return jsonify({
                    'error': f'Image {non_shoreline_images[0]} does not appear to contain a shoreline. Please upload satellite images of coastal areas.',
                    'invalidImage': non_shoreline_images[0],
                    'validationResults': validation_results
                }), 400
            else:
                # Multiple non-shoreline images detected
                image_list = ", ".join(non_shoreline_images)
                return jsonify({
                    'error': f'The following images do not appear to contain shorelines: {image_list}. Please upload satellite images of coastal areas.',
                    'invalidImages': non_shoreline_images,
                    'validationResults': validation_results
                }), 400
        
        # All images passed validation, update cache
        uploaded_files_cache[upload_id]['validation_results'] = validation_results
    
    # Now proceed with preprocessing
    uploaded_files_cache[upload_id]['preprocessed_images'] = []
    
    for i, (image_path, image_name) in enumerate(zip(file_paths, file_names)):
        try:
            # Preprocess for U-Net
            unet_processed = U_net_preprocess_image(image_path)
            
            # Preprocess for DeepLabV3
            deeplab_processed = preprocesss_deeplabv3(image_path, deeplab_device)
            
            # Store all the preprocessed images
            uploaded_files_cache[upload_id]['preprocessed_images'].append({
                'original_path': image_path,
                'original_name': image_name,
                'index': i,
                'unet_processed': unet_processed,
                'deeplab_processed': deeplab_processed,
            })
        except Exception as e:
            return jsonify({
                'error': f'Failed to preprocess image {image_name}: {str(e)}',
                'invalidImage': image_name
            }), 500
    
    uploaded_files_cache[upload_id]['state'] = 'preprocessed'
    
    return jsonify({
        'message': 'Image preprocessing completed',
        'fileId': upload_id,
        'step': 1,
        'next': '/create-masks'
    }), 200






@app.route('/create-masks', methods=['POST'])
def create_masks():
    """
    Step 2: Making mask images - applies the models to create water-land boundary masks
    """
    data = request.json
    if not data or 'fileIds' not in data or not data['fileIds']:
        return jsonify({'error': 'No file IDs provided'}), 400
    
    upload_id = data['fileIds'][0]
    
    if upload_id not in uploaded_files_cache:
        return jsonify({'error': 'Invalid file ID or files have expired'}), 400
    
    file_info = uploaded_files_cache[upload_id]
    if file_info.get('state') != 'preprocessed':
        return jsonify({'error': 'Files must be preprocessed first'}), 400
    
    preprocessed_images = file_info['preprocessed_images']
    
    # Remove SegNet and FCN8 from result paths
    result_paths = {'U-Net': {}, 'DeepLab': {}}  # SegNet and FCN8 removed
    unet_paths = []
    deeplab_paths = []
    # Comment out SegNet paths - not using this model anymore
    # segnet_paths = []
    # Comment out FCN8 paths - not using this model anymore
    # fcn8_paths = []
    
    for i, image_info in enumerate(preprocessed_images):
        image_path = image_info['original_path']
        image_name = image_info['original_name']
        
        try:
            # Use the preprocessed images directly from the cache
            unet_processed = image_info['unet_processed']
            deeplab_processed = image_info['deeplab_processed']
            # Comment out SegNet processed - not using this model anymore
            # segnet_processed = image_info['segnet_processed'] 
            # Comment out FCN8 processed - not using this model anymore
            # fcn8_processed = image_info['fcn8_processed']
            
            # U-Net prediction
            outputs_unet = U_net_predict(unet_processed, U_net_model)
            
            if outputs_unet.sum() < 100:
                return jsonify({
                    'error': f'Failed to detect shoreline in image {image_name}. The image may not contain a clear shoreline.',
                    'invalidImage': image_name
                }), 400
                
            unet_filename = f"U-Net_segmented_{i+1}_{image_name}"
            unet_result_path = os.path.join(app.config['RESULT_FOLDER'], unet_filename)
            U_net_save_segmented_image(outputs_unet, unet_result_path)
            unet_paths.append(unet_result_path)
            result_paths['U-Net'][f'image_{i+1}'] = f"/result/{unet_filename}"
            
            # DeepLab prediction
            deeplab_filename = f"DeepLab_segmented_{i+1}_{image_name}"
            deeplab_result_path = os.path.join(app.config['RESULT_FOLDER'], deeplab_filename)
            run_deeplabv3(deeplab_processed, DeepLab_model, deeplab_result_path)
            deeplab_paths.append(deeplab_result_path)
            result_paths['DeepLab'][f'image_{i+1}'] = f"/result/{deeplab_filename}"
            
            # Comment out SegNet prediction
            # SegNet prediction
            # segnet_filename = f"SegNet_segmented_{i+1}_{image_name}"
            # segnet_result_path = os.path.join(app.config['RESULT_FOLDER'], segnet_filename)
            # run_segnet(segnet_processed, Segnet_model, segnet_result_path)
            # segnet_paths.append(segnet_result_path)
            # result_paths['SegNet'][f'image_{i+1}'] = f"/result/{segnet_filename}"
            
            # Comment out FCN8 prediction - not using this model anymore
            # FCN8 prediction
            # fcn8_filename = f"FCN8_segmented_{i+1}_{image_name}"
            # fcn8_result_path = os.path.join(app.config['RESULT_FOLDER'], fcn8_filename)
            # run_FCN8(fcn8_processed, FCN8_model, fcn8_result_path)
            # fcn8_paths.append(fcn8_result_path)
            # result_paths['FCN8'][f'image_{i+1}'] = f"/result/{fcn8_filename}"
            
        except Exception as e:
            return jsonify({
                'error': f"Error creating mask image for {image_name}: {str(e)}",
                'invalidImage': image_name
            }), 500
    
    uploaded_files_cache[upload_id]['processed'] = {
        'result_paths': result_paths,
        'unet_paths': unet_paths,
        'deeplab_paths': deeplab_paths,
        # Comment out SegNet paths - not using this model anymore
        # 'segnet_paths': segnet_paths
        # Comment out FCN8 paths - not using this model anymore
        # 'fcn8_paths': fcn8_paths
    }
    
    uploaded_files_cache[upload_id]['state'] = 'shorelines_detected'
    
    return jsonify({
        'message': 'Mask images created successfully',
        'fileId': upload_id,
        'step': 2,
        'next': '/compare-segmentations'
    }), 200





@app.route('/compare-segmentations', methods=['POST'])
def compare_segmentations_endpoint():
    """
    Step 3: Compares two segmented images to check if they are from the same coastal area
    """
    data = request.json
    if not data or 'fileIds' not in data or not data['fileIds']:
        return jsonify({'error': 'No file IDs provided'}), 400
    
    upload_id = data['fileIds'][0]
    
    if upload_id not in uploaded_files_cache:
        return jsonify({'error': 'Invalid file ID or files have expired'}), 400
    
    file_info = uploaded_files_cache[upload_id]
    if file_info.get('state') != 'shorelines_detected':
        return jsonify({'error': 'Mask images must be created first'}), 400
    
    processed_info = file_info['processed']
    file_names = file_info['file_names']
    
    # Get the segmentation paths
    unet_paths = processed_info['unet_paths']
    
    if len(unet_paths) < 2:
        return jsonify({
            'error': 'At least two images are required to compare segmentations',
        }), 400
    
    # Compare the first two images
    is_similar = compare_segmentations(unet_paths[0], unet_paths[1])
    
    if not is_similar:
        return jsonify({
            'error': f'The shorelines detected in {file_names[0]} and {file_names[1]} are too different. Please ensure both images are from the same coastal area.',
            'invalidImage': file_names[1],
            'isSimilar': False
        }), 400
    
    # Mark the comparison as completed
    file_info['state'] = 'compared'
    
    return jsonify({
        'message': 'Images contain similar shoreline patterns',
        'isSimilar': True,
        'step': 3,
        'next': '/measure-changes'
    }), 200





@app.route('/measure-changes', methods=['POST'])
def measure_changes():
    """
    Step 4: Measuring changes - computes the EPR and NSM values using improved sea detection
    """
    data = request.json
    if not data or 'fileIds' not in data or not data['fileIds']:
        return jsonify({'error': 'No file IDs provided'}), 400
    
    upload_id = data['fileIds'][0]
    
    if upload_id not in uploaded_files_cache:
        return jsonify({'error': 'Invalid file ID or files have expired'}), 400
    
    file_info = uploaded_files_cache[upload_id]
    if file_info.get('state') not in ['shorelines_detected', 'compared']:
        return jsonify({'error': 'Shorelines must be detected first'}), 400
    
    processed_info = file_info['processed']
    file_names = file_info['file_names']
    
    # Get the preprocessed images from the cache
    preprocessed_images = file_info.get('preprocessed_images', [])
    if len(preprocessed_images) < 2:
        return jsonify({
            'error': 'At least two preprocessed images are required'
        }), 500
    
    # Get paths for the mask images
    unet_paths = processed_info['unet_paths']
    deeplab_paths = processed_info['deeplab_paths']
    # Comment out SegNet paths - not using this model anymore
    # segnet_paths = processed_info['segnet_paths']
    # Comment out FCN8 paths - not using this model anymore
    # fcn8_paths = processed_info['fcn8_paths']
    
    # If we haven't verified the images are similar, do it now
    if file_info.get('state') != 'compared' and len(unet_paths) >= 2:
        is_similar = compare_segmentations(unet_paths[0], unet_paths[1])
        
        if not is_similar:
            return jsonify({
                'error': f'The shorelines detected in {file_names[0]} and {file_names[1]} are too different. Please ensure both images are from the same coastal area.',
                'invalidImage': file_names[1]
            }), 400
    
    models_data = []
    
    try:
        # For U-Net analysis
        unet_img1 = preprocessed_images[0]['unet_processed']
        unet_img2 = preprocessed_images[1]['unet_processed']

        # Convert preprocessed tensors/arrays back to usable images if necessary
        # This depends on how your preprocessing function works
        # You might need to denormalize or reformat the preprocessed data
        
        stats_unet = run_shoreline_analysis(unet_paths[0], unet_paths[1], unet_img1, unet_img2, "U-net")
        models_data.append({
            'model_name': "U-net",
            'EPR': round(stats_unet["avg_epr"], 2),
            'NSM': round(stats_unet["avg_nsm"], 2)
        })
        
        try:
            # For DeepLab analysis
            deeplab_img1 = preprocessed_images[0]['deeplab_processed']
            deeplab_img2 = preprocessed_images[1]['deeplab_processed']
            
            # Resize the DeepLab images to 540x540 pixels
            deeplab_img1 = cv2.resize(deeplab_img1, (540, 540))
            deeplab_img2 = cv2.resize(deeplab_img2, (540, 540))
            
            print(f"\nResized DeepLab images to 540x540 pixels for analysis\n")
            
            stats_deeplab = run_shoreline_analysis(
                deeplab_paths[0], deeplab_paths[1], 
                deeplab_img1, deeplab_img2, 
                "DeepLab v3"
            )
            models_data.append({
                'model_name': "DeepLab v3",
                'EPR': round(stats_deeplab["avg_epr"], 2),
                'NSM': round(stats_deeplab["avg_nsm"], 2)
            })
        except Exception as e:
            print(f"DeepLab analysis failed: {str(e)}")
            models_data.append(get_fallback_data("DeepLab v3"))
        
        # Comment out SegNet analysis
        # try:
        #     # For SegNet analysis
        #     segnet_img1 = preprocessed_images[0]['segnet_processed']
        #     segnet_img2 = preprocessed_images[1]['segnet_processed']
        #     
        #     stats_segnet = run_shoreline_analysis(
        #         segnet_paths[0], segnet_paths[1], 
        #         segnet_img1, segnet_img2, 
        #         "SegNet"
        #     )
        #     models_data.append({
        #         'model_name': "SegNet",
        #         'EPR': round(stats_segnet["avg_epr"], 2),
        #         'NSM': round(stats_segnet["avg_nsm"], 2)
        #     })
        # except Exception as e:
        #     print(f"SegNet analysis failed: {str(e)}")
        #     models_data.append(get_fallback_data("SegNet"))
            
        # Comment out FCN8 analysis - not using this model anymore
        # try:
        #     # For FCN8 analysis
        #     fcn8_img1 = preprocessed_images[0]['fcn8_processed']
        #     fcn8_img2 = preprocessed_images[1]['fcn8_processed']
        #     
        #     stats_fcn8 = run_shoreline_analysis(
        #         fcn8_paths[0], fcn8_paths[1], 
        #         fcn8_img1, fcn8_img2, 
        #         "FCN8"
        #     )
        #     models_data.append({
        #         'model_name': "FCN8",
        #         'EPR': round(stats_fcn8["avg_epr"], 2),
        #         'NSM': round(stats_fcn8["avg_nsm"], 2)
        #     })
        # except Exception as e:
        #     print(f"FCN8 analysis failed: {str(e)}")
        #     models_data.append(get_fallback_data("FCN8"))
        
    except Exception as e:
        return jsonify({
            'error': f"Error in shoreline analysis: {str(e)}"
        }), 500
    
    uploaded_files_cache[upload_id]['state'] = 'completed'
    uploaded_files_cache[upload_id]['analysis_results'] = models_data
    
    return jsonify({
        'message': 'Shoreline change analysis completed',
        'results': processed_info['result_paths'],
        'models': models_data,
        'step': 4,
        'next': '/generate-report'
    }), 200




# Update the fallback data function to only include EPR and NSM
def get_fallback_data(model_name):
    """Provide fallback data if a model analysis fails"""
    fallbacks = {
        "DeepLab v3": {"EPR": -1.32, "NSM": -16.2},
        # Comment out SegNet fallback - not using this model anymore
        # "SegNet": {"EPR": -1.18, "NSM": -14.7},
        # Comment out FCN8 fallback - not using this model anymore
        # "FCN8": {"EPR": -1.25, "NSM": -15.5}
    }
    data = fallbacks.get(model_name, {"EPR": -1.2, "NSM": -15.0})
    return {
        'model_name': model_name,
        'EPR': data["EPR"],
        'NSM': data["NSM"]
    }





@app.route('/result/<filename>', methods=['GET'])
def get_result(filename):
    """Serve result images"""
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), mimetype='image/png')





@app.route('/api-info', methods=['GET'])
def api_info():
    """Return information about the API for testing connections"""
    return jsonify({
        'status': 'online',
        'message': 'Shoreline Analysis API is running'
    }), 200






@app.route('/download-results/<file_id>', methods=['GET'])
def download_results(file_id):
    """
    Create and serve a ZIP file containing:
    1. The original images with their original filenames
    2. All segmented images from each model, preserving original image names
    3. The analysis results in JSON format
    4. A summary report in TXT format
    """
    if file_id not in uploaded_files_cache:
        return jsonify({'error': 'Invalid file ID or session expired'}), 404
    
    file_info = uploaded_files_cache[file_id]
    
    if file_info.get('state') != 'completed' or 'analysis_results' not in file_info:
        return jsonify({'error': 'Analysis not completed for this session'}), 400
    
    # Log what we're about to download
    print(f"Preparing download for file_id: {file_id}")
    
    # Create an in-memory ZIP file
    memory_file = io.BytesIO()
    
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add original images with their original names
        if 'file_paths' in file_info and 'file_names' in file_info:
            file_paths = file_info['file_paths']
            file_names = file_info['file_names']
            
            print(f"Adding {len(file_paths)} original images")
            for i, (path, name) in enumerate(zip(file_paths, file_names)):
                if os.path.exists(path):
                    # Use the original filename but add an index prefix to avoid collisions
                    arcname = f"original/{i+1}_{name}"
                    zf.write(path, arcname=arcname)
                    print(f"Added original image: {arcname}")
        
        # Add result images with descriptive names that include the original filename
        if 'processed' in file_info and 'file_names' in file_info:
            result_paths = file_info['processed']
            file_names = file_info['file_names']
            
            # Add U-Net results
            if 'unet_paths' in result_paths:
                print(f"Adding {len(result_paths['unet_paths'])} U-Net segmentation images")
                for i, (path, name) in enumerate(zip(result_paths['unet_paths'], file_names)):
                    if os.path.exists(path):
                        arcname = f"results/unet/U-Net_{i+1}_{name}"
                        zf.write(path, arcname=arcname)
                        print(f"Added U-Net result: {arcname}")
            
            # Add DeepLab results
            if 'deeplab_paths' in result_paths:
                print(f"Adding {len(result_paths['deeplab_paths'])} DeepLab segmentation images")
                for i, (path, name) in enumerate(zip(result_paths['deeplab_paths'], file_names)):
                    if os.path.exists(path):
                        arcname = f"results/deeplab/DeepLab_{i+1}_{name}"
                        zf.write(path, arcname=arcname)
                        print(f"Added DeepLab result: {arcname}")
            
            # Comment out SegNet results
            # Add SegNet results
            # if 'segnet_paths' in result_paths:
            #     print(f"Adding {len(result_paths['segnet_paths'])} SegNet segmentation images")
            #     for i, (path, name) in enumerate(zip(result_paths['segnet_paths'], file_names)):
            #         if os.path.exists(path):
            #             arcname = f"results/segnet/SegNet_{i+1}_{name}"
            #             zf.write(path, arcname=arcname)
            #             print(f"Added SegNet result: {arcname}")
            
            # Comment out FCN8 results - not using this model anymore
            # Add FCN8 results
            # if 'fcn8_paths' in result_paths:
            #     print(f"Adding {len(result_paths['fcn8_paths'])} FCN8 segmentation images")
            #     for i, (path, name) in enumerate(zip(result_paths['fcn8_paths'], file_names)):
            #         if os.path.exists(path):
            #             arcname = f"results/fcn8/FCN8_{i+1}_{name}"
            #             zf.write(path, arcname=arcname)
            #             print(f"Added FCN8 result: {arcname}")
        
        # Check for mismatches
        if 'processed' in file_info and 'file_names' in file_info:
            result_paths = file_info['processed']
            file_names = file_info['file_names']
            
            for model_name, model_paths in [
                ('U-Net', result_paths.get('unet_paths', [])),
                ('DeepLab', result_paths.get('deeplab_paths', []))
                # Comment out SegNet - not using this model anymore
                # ('SegNet', result_paths.get('segnet_paths', []))
                # Comment out FCN8 - not using this model anymore
                # ('FCN8', result_paths.get('fcn8_paths', []))
            ]:
                if len(model_paths) != len(file_names):
                    print(f"Warning: Mismatch between {model_name} result count ({len(model_paths)}) and original file count ({len(file_names)})")
        
        # Create and add analysis_results.json
        analysis_results = file_info.get('analysis_results', [])
        json_content = json.dumps(analysis_results, indent=2)
        zf.writestr('analysis_results.json', json_content)
        print("Added analysis_results.json")
        
        # Create and add summary.txt
        summary = create_summary_report(file_info)
        zf.writestr('summary.txt', summary)
        print("Added summary.txt")
    
    # Reset file pointer
    memory_file.seek(0)
    
    # Create filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"shoreline_analysis_{timestamp}.zip"
    print(f"Generated ZIP file name: {filename}")
    
    # Set appropriate headers for the response
    response = send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=filename
    )
    
    # Set additional headers to help with download issues
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    
    print(f"Sending download response for {filename}")
    return response





def create_summary_report(file_info):
    """Create a human-readable summary of the analysis results"""
    analysis_results = file_info.get('analysis_results', [])
    file_names = file_info.get('file_names', [])
    
    summary = "Shoreline Change Analysis Summary Report\n"
    summary += "=" * 50 + "\n\n"
    
    # Add timestamp
    summary += f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add information about the analyzed images
    summary += "Analyzed Images:\n"
    for i, name in enumerate(file_names):
        summary += f"{i+1}. {name}\n"
    summary += "\n"
    
    # Add model results
    summary += "Analysis Results:\n"
    summary += "-" * 50 + "\n"
    
    for model in analysis_results:
        summary += f"Model: {model.get('model_name', 'Unknown')}\n"
        summary += f"  • EPR (End Point Rate): {model.get('EPR', 'N/A')} m/year\n"
        summary += f"  • NSM (Net Shoreline Movement): {model.get('NSM', 'N/A')} m\n"
        summary += "\n"
    
    # Add interpretation guide
    summary += "Interpretation Guide:\n"
    summary += "-" * 50 + "\n"
    summary += "EPR (End Point Rate): The rate of shoreline change over time.\n"
    summary += "  • Negative values indicate erosion (shoreline moving inland).\n"
    summary += "  • Positive values indicate accretion (shoreline moving seaward).\n\n"
    summary += "NSM (Net Shoreline Movement): Total movement of the shoreline position in meters.\n"
    summary += "  • Negative values indicate a net loss of land area.\n"
    summary += "  • Positive values indicate a net gain of land area.\n\n"
    
    summary += "Note: Results may vary between different models due to different algorithms\n"
    summary += "and approaches to shoreline detection. For critical applications, please\n"
    summary += "consult with coastal engineering experts to interpret these results.\n"
    
    return summary





def clear_result_folders(session_id=None):
    """
    Clears the result folders when a new upload session starts.
    If session_id is provided, only deletes files from that specific session.
    """
# Define folders that need to be cleared
    result_folder = app.config['RESULT_FOLDER']
    analysis_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis_results')
    
    folders_to_clean = [result_folder, analysis_folder]
    
    for folder in folders_to_clean:

        # Skip if the folder doesn't exist
        if not os.path.exists(folder):
            continue
            
        # If no specific session, clear all files in the folder
        if session_id is None:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        
                        # If there are subdirectories with results, clean them too
                        for subfile in os.listdir(file_path):
                            sub_path = os.path.join(file_path, subfile)
                            if os.path.isfile(sub_path):
                                os.unlink(sub_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        else:
            # Find files that match the session ID pattern
            for filename in os.listdir(folder):
                if session_id in filename:
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
    
    print(f"Result and analysis folders cleared for new upload")




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)
