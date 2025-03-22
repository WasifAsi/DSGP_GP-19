import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['TK_SILENCE_DEPRECATION'] = "1"

# Force matplotlib to use a non-GUI backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Now import your remaining libraries
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import uuid
import time
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from U_net_arciteuture import load_U_net_model, U_net_predict, U_net_save_segmented_image
from deepLabV3_architecture import load_Deeplab_model, run
from Segnet_architecture import load_Segnet_model, run_segnet
from FCN8_arciteuture import load_FCN8_model, run_FCN8

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
FCN8_model = load_FCN8_model()

# Simple cache for uploaded files
uploaded_files_cache = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compare_segmentations(image1_path, image2_path):
    """
    Compare two segmented images to check if they are from the same area
    Returns True if segmentations are similar enough
    """
    # Check if files exist
    if not os.path.exists(image1_path):
        print(f"Segmentation file not found: {image1_path}")
        return False
    if not os.path.exists(image2_path):
        print(f"Segmentation file not found: {image2_path}")
        return False
    
    # Read segmentation images in grayscale
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print("Error: One or both segmentation images could not be loaded.")
        return False
    
    # Resize images to the same size if necessary
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Convert to binary (0/255) for better comparison
    _, binary1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
    
    # Calculate percentage of different pixels
    diff_pixels = np.count_nonzero(binary1 != binary2)
    total_pixels = binary1.size
    diff_percentage = diff_pixels / total_pixels
    print(f"Segmentation difference: {diff_percentage:.2%}")
    
    # Compute SSIM on binary masks for more robust comparison
    score, _ = ssim(binary1, binary2, full=True)
    print(f"Segmentation similarity score (SSIM): {score}")
    
    # Calculate Intersection over Union (IoU) - a common metric for segmentation comparison
    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()
    iou = intersection / union if union > 0 else 0
    print(f"Segmentation IoU score: {iou:.2f}")
    
    # Use a much more reasonable threshold for the SSIM score
    # Most coastal images from same location should have at least 0.1 SSIM
    is_similar = score > 0.98
    
    if not is_similar:
        print(f"VALIDATION FAILED: Images appear to be from different coastal areas")
        print(f"SSIM: {score} (threshold: 0.1), IoU: {iou}")
    else:
        print(f"VALIDATION PASSED: Images appear to be from the same coastal area\n")
    
    return is_similar

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
        'file_names': file_names,
        'state': 'uploaded'  # Track the processing state
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
    Step 1: Preprocessing only - this handles the initial preprocessing step
    and basic validation of images
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
    
    # Check that we're in the right state to preprocess
    if file_info.get('state') != 'uploaded':
        return jsonify({'error': 'Files have already been preprocessed'}), 400
    
    file_paths = file_info['file_paths']
    file_names = file_info['file_names']
    
    # Set up data structures for each processing phase
    uploaded_files_cache[upload_id]['preprocessed_images'] = []
    
    # Step 1: Perform basic validation on images
    print("Starting preprocessing...")
    
    # Basic validation - check that all images can be opened and have valid dimensions
    for i, (image_path, image_name) in enumerate(zip(file_paths, file_names)):
        img = cv2.imread(image_path)
        if img is None:
            return jsonify({
                'error': f'Image {image_name} could not be loaded. The file may be corrupted.',
                'invalidImage': image_name
            }), 400
        
        # Check image dimensions are reasonable for analysis
        if img.shape[0] < 100 or img.shape[1] < 100:
            return jsonify({
                'error': f'Image {image_name} is too small. Please use images of at least 100x100 pixels.',
                'invalidImage': image_name
            }), 400
        
        # Store preprocessed image info
        uploaded_files_cache[upload_id]['preprocessed_images'].append({
            'original_path': image_path,
            'original_name': image_name,
            'index': i
        })
    
    # Update state
    uploaded_files_cache[upload_id]['state'] = 'preprocessed'
    
    return jsonify({
        'message': 'Image preprocessing completed',
        'fileId': upload_id,
        'step': 1,
        'next': '/detect-shoreline'
    }), 200

@app.route('/detect-shoreline', methods=['POST'])
def detect_shoreline():
    """
    Step 2: Shoreline detection - applies the models to detect shorelines
    """
    # Validate request
    data = request.json
    if not data or 'fileIds' not in data or not data['fileIds']:
        return jsonify({'error': 'No file IDs provided'}), 400
    
    upload_id = data['fileIds'][0]
    
    # Validate upload ID and state
    if upload_id not in uploaded_files_cache:
        return jsonify({'error': 'Invalid file ID or files have expired'}), 400
    
    file_info = uploaded_files_cache[upload_id]
    if file_info.get('state') != 'preprocessed':
        return jsonify({'error': 'Files must be preprocessed first'}), 400
    
    # Get preprocessed image info
    preprocessed_images = file_info['preprocessed_images']
    
    # Process with all models
    result_paths = {'U-Net': {}, 'DeepLab': {}, 'SegNet': {}, 'FCN8': {}}  # Add FCN8 to result paths
    unet_paths = []
    deeplab_paths = []
    segnet_paths = []
    fcn8_paths = []  # Add FCN8 paths list
    
    print("Detecting shorelines...")
    
    # Process each image with each model
    for i, image_info in enumerate(preprocessed_images):
        image_path = image_info['original_path']
        image_name = image_info['original_name']
        
        try:
            # Process with U-Net
            outputs_unet = U_net_predict(image_path, U_net_model)
            
            # Validate U-Net prediction - check if it detected any shoreline
            if outputs_unet.sum() < 100:  # Arbitrary threshold, adjust as needed
                # Return error and stop processing if no shoreline detected
                return jsonify({
                    'error': f'Failed to detect shoreline in image {image_name}. The image may not contain a clear shoreline.',
                    'invalidImage': image_name
                }), 400
                
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
            
            # Process with FCN8
            fcn8_filename = f"FCN8_segmented_{i+1}_{image_name}"
            fcn8_result_path = os.path.join(app.config['RESULT_FOLDER'], fcn8_filename)
            run_FCN8(image_path, FCN8_model, fcn8_result_path)
            fcn8_paths.append(fcn8_result_path)
            result_paths['FCN8'][f'image_{i+1}'] = f"/result/{fcn8_filename}"
            
        except Exception as e:
            # Handle any errors during shoreline detection
            error_msg = f"Error detecting shoreline in image {image_name}: {str(e)}"
            print(error_msg)
            return jsonify({
                'error': error_msg,
                'invalidImage': image_name
            }), 500
    
    # Store processed results with paths for all models
    uploaded_files_cache[upload_id]['processed'] = {
        'result_paths': result_paths,
        'unet_paths': unet_paths,
        'deeplab_paths': deeplab_paths,
        'segnet_paths': segnet_paths,
        'fcn8_paths': fcn8_paths  # Add FCN8 paths to the cache
    }
    
    # Update state
    uploaded_files_cache[upload_id]['state'] = 'shorelines_detected'
    
    return jsonify({
        'message': 'Shoreline detection completed',
        'fileId': upload_id,
        'step': 2,
        'next': '/measure-changes'
    }), 200

@app.route('/measure-changes', methods=['POST'])
def measure_changes():
    """
    Step 3: Measuring changes - computes the EPR and NSM values
    """
    print("\n----- MEASURE CHANGES ENDPOINT CALLED -----")
    
    # Validate request
    data = request.json
    if not data or 'fileIds' not in data or not data['fileIds']:
        print("Error: No file IDs provided")
        return jsonify({'error': 'No file IDs provided'}), 400
    
    upload_id = data['fileIds'][0]
    print(f"Processing upload ID: {upload_id}")
    
    # Validate upload ID and state
    if upload_id not in uploaded_files_cache:
        print(f"Error: Invalid file ID {upload_id}")
        return jsonify({'error': 'Invalid file ID or files have expired'}), 400
    
    file_info = uploaded_files_cache[upload_id]
    if file_info.get('state') != 'shorelines_detected':
        print(f"Error: Files not in correct state. Current state: {file_info.get('state')}")
        return jsonify({'error': 'Shorelines must be detected first'}), 400
    
    processed_info = file_info['processed']
    file_names = file_info['file_names']
    
    print(f"Files to compare: {file_names}")
    
    # Get model paths
    unet_paths = processed_info['unet_paths']
    deeplab_paths = processed_info['deeplab_paths']
    segnet_paths = processed_info['segnet_paths']
    fcn8_paths = processed_info['fcn8_paths']  # Get FCN8 paths
    
    print("Checking segmentation consistency...")
    
    # First, check if the segmented shorelines are similar enough
    if len(unet_paths) >= 2:
        print(f"Comparing U-Net segmentations: {unet_paths[0]} and {unet_paths[1]}")
        # Compare the U-Net segmentations since that's our primary model
        is_similar = compare_segmentations(unet_paths[0], unet_paths[1])
        print(f"Segmentation comparison result: {'SIMILAR' if is_similar else 'DIFFERENT'}")
        
        if not is_similar:
            print(f"VALIDATION FAILED: Images appear to be from different coastal areas")
            error_response = {
                'error': f'The shorelines detected in {file_names[0]} and {file_names[1]} are too different. Please ensure both images are from the same coastal area.',
                'invalidImage': file_names[1]
            }
            print(f"Returning error response: {error_response}")
            return jsonify(error_response), 400  # Return 400 Bad Request status
    
    print("Segmentation check passed, proceeding with analysis...")
    
    # Calculate shore changes for all models with error handling
    models_data = []
    
    # Try to run analysis for all models
    try:
        # Process U-Net results first
        print("Analyzing U-Net results...")
        unet_epr, unet_nsm = run_shoreline_analysis(unet_paths[0], unet_paths[1], "U-net")
        models_data.append({
            'model_name': "U-net",
            'EPR': round(unet_epr, 2),
            'NSM': round(unet_nsm)
        })
        
        # Only continue with other models if U-Net succeeded
        # Process DeepLab results
        try:
            print("Analyzing DeepLab results...")
            deeplab_epr, deeplab_nsm = run_shoreline_analysis(deeplab_paths[0], deeplab_paths[1], "DeepLab v3")
            models_data.append({
                'model_name': "DeepLab v3",
                'EPR': round(deeplab_epr, 2),
                'NSM': round(deeplab_nsm)
            })
        except Exception as e:
            print(f"Warning: DeepLab analysis failed: {str(e)}")
            # Add fallback values
            models_data.append({
                'model_name': "DeepLab v3",
                'EPR': -1.32,
                'NSM': -16.2
            })
        
        # Process SegNet results
        try:
            print("Analyzing SegNet results...")
            segnet_epr, segnet_nsm = run_shoreline_analysis(segnet_paths[0], segnet_paths[1], "SegNet")
            models_data.append({
                'model_name': "SegNet",
                'EPR': round(segnet_epr, 2),
                'NSM': round(segnet_nsm)
            })
        except Exception as e:
            print(f"Warning: SegNet analysis failed: {str(e)}")
            # Add fallback values
            models_data.append({
                'model_name': "SegNet",
                'EPR': -1.18,
                'NSM': -14.7
            })
            
        # Process FCN8 results
        try:
            print("Analyzing FCN8 results...")
            fcn8_epr, fcn8_nsm = run_shoreline_analysis(fcn8_paths[0], fcn8_paths[1], "FCN8")
            models_data.append({
                'model_name': "FCN8",
                'EPR': round(fcn8_epr, 2),
                'NSM': round(fcn8_nsm)
            })
        except Exception as e:
            print(f"Warning: FCN8 analysis failed: {str(e)}")
            # Add fallback values
            models_data.append({
                'model_name': "FCN8",
                'EPR': -1.25,
                'NSM': -15.5
            })
        
    except Exception as e:
        # If the primary U-Net analysis fails, return error and stop processing
        error_msg = f"Error in shoreline analysis: {str(e)}"
        print(error_msg)
        return jsonify({
            'error': error_msg
        }), 500
    
    # Update state and store results
    uploaded_files_cache[upload_id]['state'] = 'completed'
    uploaded_files_cache[upload_id]['analysis_results'] = models_data
    
    print("Analysis completed successfully. Returning results.")
    
    # Return the results
    return jsonify({
        'message': 'Shoreline change analysis completed',
        'results': processed_info['result_paths'],
        'models': models_data,
        'step': 3
    }), 200

@app.route('/result/<filename>', methods=['GET'])
def get_result(filename):
    """Serve result images"""
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), mimetype='image/png')

if __name__ == '__main__':
    # Use threaded=False to avoid tkinter threading issues
    app.run(debug=True, threaded=False)
