import os
from flask import Flask, request, render_template, jsonify, url_for, send_from_directory
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import time
import io
from PIL import Image
import base64

app = Flask(__name__)

# Configuration
app.config['CV_FOLDER'] = 'cv'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create CV directory only (no uploads folder needed)
os.makedirs(app.config['CV_FOLDER'], exist_ok=True)

# Model path
MODEL_PATH = 'models/deepfake.keras'

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Load model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image_from_memory(img_file):
    """Prepare image for prediction directly from memory"""
    # Open image using PIL from file stream
    img = Image.open(img_file)
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Resize to model input size
    img = img.resize((224, 224))
    # Convert to array and normalize
    x = np.array(img) / 255.0
    return np.expand_dims(x, axis=0)

def image_to_base64(img_file):
    """Convert uploaded image to base64 for display without saving"""
    # Reset file pointer to beginning
    img_file.seek(0)
    # Read image data
    img_data = img_file.read()
    # Convert to base64
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    # Determine image format
    img_file.seek(0)
    img = Image.open(img_file)
    format_lower = img.format.lower() if img.format else 'jpeg'
    # Create data URL
    return f"data:image/{format_lower};base64,{img_base64}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download_cv')
def download_cv():
    try:
        # Look for CV file in cv folder
        cv_files = [f for f in os.listdir(app.config['CV_FOLDER']) if f.lower().endswith(('.pdf', '.doc', '.docx'))]
        if cv_files:
            return send_from_directory(app.config['CV_FOLDER'], cv_files[0], as_attachment=True)
        else:
            return jsonify({'error': 'CV not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        print("Analyze endpoint called")
        
        if 'file' not in request.files:
            print("No file in request")
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        print(f"File received: {file.filename}")
        
        if file.filename == '':
            print("Empty filename")
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            print("Invalid file type")
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        if model is None:
            print("Model not loaded")
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500
        
        # Process image directly from memory without saving
        print("Processing image from memory...")
        
        # Create a copy of the file stream for base64 conversion
        file_copy = io.BytesIO(file.read())
        file.seek(0)  # Reset original file pointer
        
        # Prepare image for prediction from memory
        x = prepare_image_from_memory(file)
        prediction = model.predict(x)[0][0]
        
        # Determine result
        is_real = prediction > 0.5
        confidence = float(prediction if is_real else 1 - prediction)
        
        # Convert image to base64 for display (no file saving)
        image_data_url = image_to_base64(file_copy)
        
        result = {
            'success': True,
            'label': 'Real' if is_real else 'Deepfake',
            'confidence': f"{confidence*100:.1f}%",
            'confidence_score': confidence,
            'image_url': image_data_url,  # Base64 data URL instead of file path
            'filename': secure_filename(file.filename)  # Just for display purposes
        }
        
        print(f"Analysis complete: {result['label']} with {result['confidence']} confidence")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in analyze: {str(e)}")
        return jsonify({'success': False, 'error': f'Analysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
