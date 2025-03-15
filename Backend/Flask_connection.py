from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os

from U_net_arciteuture import load_U_net_model, U_net_predict, U_net_save_segmented_image
from deepLabV3_architecture import load_Deeplab_model , run
from Segnet_architecture import  load_Segnet_model, run_segnet

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load the models 
U_net_model = load_U_net_model()
DeepLab_model = load_Deeplab_model()
Segnet_model = load_Segnet_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if any files were uploaded
    if not any(key.startswith('file') for key in request.files):
        return jsonify({'error': 'No files uploaded'}), 400
    
    # Get the number of files (optional)
    file_count = int(request.form.get('fileCount', '0'))
    
    # Find all files in the request
    uploaded_files = [request.files[key] for key in request.files if key.startswith('file')]
    
    if not uploaded_files:
        return jsonify({'error': 'No files found'}), 400
    
    # Process each file
    file_paths = []
    file_names = []  # Store original filenames
    
    for i, file in enumerate(uploaded_files):
        if file.filename == '':
            continue
            
        if file and allowed_file(file.filename):
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            file_paths.append(file_path)
            file_names.append(file.filename)  # Store the filename for later use
    
    # Make sure we have at least 2 images
    if len(file_paths) < 2:
        return jsonify({'error': 'At least 2 valid images are required'}), 400
    
    # Process the images
    result_paths = {}
    unet_predictions = []
    deeplab_predictions = []
    segnet_predictions = []
    
    # Process each image with each model
    for i, (image_path, image_name) in enumerate(zip(file_paths, file_names)):
        # Process with U-Net
        outputs_unet = U_net_predict(image_path, U_net_model)
        unet_predictions.append(outputs_unet)
        
        # Save the U-Net result
        unet_filename = f"U-Net_segmented_{i+1}_{image_name}"
        unet_result_path = os.path.join(app.config['RESULT_FOLDER'], unet_filename)
        U_net_save_segmented_image(outputs_unet, unet_result_path)
        
        # Store the result path using a unique key for each image
        if 'U-Net' not in result_paths:
            result_paths['U-Net'] = {}
        result_paths['U-Net'][f'image_{i+1}'] = f"/result/{unet_filename}"
        
        

        
        # Process with DeepLab
        
        
        # Save the DeepLab result
        deeplab_filename = f"DeepLab_segmented_{i+1}_{image_name}"
        deeplab_result_path = os.path.join(app.config['RESULT_FOLDER'], deeplab_filename)

        outputs_deeplab = run(image_path, DeepLab_model,deeplab_result_path)
        deeplab_predictions.append(outputs_deeplab)

        # Use appropriate save function for DeepLab
        # DeepLab_save_segmented_image(outputs_deeplab, deeplab_result_path)
        
        # Store the path
        if 'DeepLab' not in result_paths:
            result_paths['DeepLab'] = {}
        result_paths['DeepLab'][f'image_{i+1}'] = f"/result/{deeplab_filename}"
        
        # Process with SegNet
        
        # Save the SegNet result
        segnet_filename = f"SegNet_segmented_{i+1}_{image_name}"
        segnet_result_path = os.path.join(app.config['RESULT_FOLDER'], segnet_filename)

        
        outputs_segnet = run_segnet(image_path, Segnet_model,segnet_result_path)
        segnet_predictions.append(outputs_segnet)
        # # Use appropriate save function for SegNet
        # # SegNet_save_segmented_image(outputs_segnet, segnet_result_path)
        
        # Store the path
        if 'SegNet' not in result_paths:
            result_paths['SegNet'] = {}
        result_paths['SegNet'][f'image_{i+1}'] = f"/result/{segnet_filename}"
    
    # Calculate shoreline changes between images
    # For now, we'll use mock data, but in a real application 
    # you would analyze the predictions to generate these values
    pair_results = []
    
    for i in range(len(file_paths) - 1):
        earlier_name = file_names[i]
        later_name = file_names[i + 1]
        
        # Create a unique pair ID
        pair_id = f"pair_{i+1}"
        
        # Add analysis results for this pair
        pair_results.append({
            'pair_id': pair_id,
            'earlier_image': earlier_name,
            'later_image': later_name,
            'models': [
                {
                    'model_name': "U-net",
                    'EPR': -1.25 - (i * 0.1),  # Mock data - replace with real calculations
                    'NSM': -15.5 - (i * 0.5)
                },
                {
                    'model_name': "DeepLab v3",
                    'EPR': -1.32 - (i * 0.1),
                    'NSM': -16.2 - (i * 0.5)
                },
                {
                    'model_name': "SegNet",
                    'EPR': -1.18 - (i * 0.1),
                    'NSM': -14.7 - (i * 0.5)
                }
            ]
        })
    
    # Return comprehensive results to the frontend
    return jsonify({
        'message': 'Files processed successfully',
        'image_count': len(file_paths),
        'pair_count': len(file_paths) - 1,
        'results': result_paths,
        'pairs': pair_results,
        # Also include the legacy 'models' array for backward compatibility
        'models': [
            {
                'model_name': "U-net",
                'EPR': -1.25,
                'NSM': -15.5
            },
            {
                'model_name': "DeepLab v3",
                'EPR': -1.32,
                'NSM': -16.2
            },
            {
                'model_name': "SegNet",
                'EPR': -1.18,
                'NSM': -14.7
            }
        ]
    }), 200


@app.route('/result/<filename>', methods=['GET'])
def get_result(filename):
    # Return the segmented image
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
