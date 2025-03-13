from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os

from U_net_arciteuture import load_U_net_model, U_net_predict, U_net_save_segmented_image

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


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        result_paths = {}

        outputs_unet = U_net_predict(file_path, U_net_model)
        result_filename = f"U-Net_segmented_{file.filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        U_net_save_segmented_image(outputs_unet, result_path)
        result_paths['U-Net'] = f"/result/{result_filename}"

        
        return jsonify({'message': 'File processed successfully', 'results': result_paths}), 200

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/result/<filename>', methods=['GET'])
def get_result(filename):
    # Return the segmented image
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
