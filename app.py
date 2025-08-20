#!/usr/bin/env python3
"""
SplitSonic Web Application
A Flask-based web interface for the SplitSonic audio separation tool.
Provides REST API endpoints and serves the frontend interface.
"""

import os
import uuid
import json
import threading
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import tempfile
import shutil
from datetime import datetime, timedelta

# Import existing SplitSonic functions (no modifications needed)
from modelsplitter import separate_stems, remove_guitar, remove_guitar_keep_keyboards, remove_vocals

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['SECRET_KEY'] = 'splitsonic-secret-key-change-in-production'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# In-memory storage for job status (use Redis in production)
jobs = {}

# Supported audio file extensions
ALLOWED_EXTENSIONS = {
    'mp3', 'wav', 'flac', 'm4a', 'aac', 'ogg', 'wma'
}

def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files():
    """Clean up files older than 1 hour."""
    cutoff_time = datetime.now() - timedelta(hours=1)
    
    for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.getctime(filepath) < cutoff_time.timestamp():
                try:
                    if os.path.isfile(filepath):
                        os.remove(filepath)
                    elif os.path.isdir(filepath):
                        shutil.rmtree(filepath)
                except Exception as e:
                    print(f"Error cleaning up {filepath}: {e}")

@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and return job ID for processing.
    
    Returns:
        JSON response with job_id and file info
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not supported. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        # Add job_id prefix to avoid filename conflicts
        safe_filename = f"{job_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)
        
        # Initialize job status
        jobs[job_id] = {
            'status': 'uploaded',
            'filename': filename,
            'filepath': filepath,
            'created_at': datetime.now().isoformat(),
            'progress': 0,
            'message': 'File uploaded successfully',
            'results': {}
        }
        
        # Clean up old files
        cleanup_old_files()
        
        return jsonify({
            'job_id': job_id,
            'filename': filename,
            'message': 'File uploaded successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process', methods=['POST'])
def process_audio():
    """
    Start audio processing with selected operations.
    
    Expected JSON payload:
    {
        "job_id": "uuid",
        "operations": ["separate_stems", "remove_guitar", "remove_vocals", "selective_guitar"],
        "guitar_filter_strength": 0.9  // optional, for selective guitar removal
    }
    """
    try:
        data = request.get_json()
        job_id = data.get('job_id')
        operations = data.get('operations', [])
        guitar_filter_strength = data.get('guitar_filter_strength', 0.9)
        
        if not job_id or job_id not in jobs:
            return jsonify({'error': 'Invalid job ID'}), 400
        
        if not operations:
            return jsonify({'error': 'No operations selected'}), 400
        
        job = jobs[job_id]
        if job['status'] != 'uploaded':
            return jsonify({'error': 'Job already processing or completed'}), 400
        
        # Start processing in background thread
        thread = threading.Thread(
            target=process_audio_background,
            args=(job_id, operations, guitar_filter_strength)
        )
        thread.start()
        
        # Update job status
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['message'] = 'Processing started...'
        jobs[job_id]['progress'] = 5
        
        return jsonify({'message': 'Processing started', 'job_id': job_id})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_audio_background(job_id, operations, guitar_filter_strength):
    """
    Background function to process audio using existing modelsplitter functions.
    
    Args:
        job_id: Unique job identifier
        operations: List of operations to perform
        guitar_filter_strength: Strength for selective guitar removal (0.0-1.0)
    """
    try:
        job = jobs[job_id]
        input_filepath = job['filepath']
        
        # Create output directory for this job
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename without extension
        base_filename = os.path.splitext(job['filename'])[0]
        
        total_operations = len(operations)
        completed_operations = 0
        
        # Process each requested operation
        for operation in operations:
            try:
                jobs[job_id]['message'] = f'Processing: {operation.replace("_", " ").title()}'
                
                if operation == 'separate_stems':
                    # Create stems directory
                    stems_dir = os.path.join(output_dir, 'stems')
                    stems = separate_stems(input_filepath, stems_dir)
                    
                    # Record stem files
                    stem_files = {}
                    for stem in stems:
                        stem_file = os.path.join(stems_dir, f"{base_filename}_{stem}.wav")
                        if os.path.exists(stem_file):
                            stem_files[stem] = f"/api/download/{job_id}/stems/{base_filename}_{stem}.wav"
                    
                    jobs[job_id]['results']['stems'] = {
                        'files': stem_files,
                        'description': 'Individual instrument stems'
                    }
                
                elif operation == 'remove_guitar':
                    output_path = os.path.join(output_dir, f"{base_filename}_no_guitar.wav")
                    remove_guitar(input_filepath, output_path)
                    jobs[job_id]['results']['no_guitar'] = {
                        'file': f"/api/download/{job_id}/{base_filename}_no_guitar.wav",
                        'description': 'Audio with guitar removed (traditional method)'
                    }
                
                elif operation == 'selective_guitar':
                    output_path = os.path.join(output_dir, f"{base_filename}_selective_no_guitar.wav")
                    remove_guitar_keep_keyboards(input_filepath, output_path, guitar_filter_strength=guitar_filter_strength)
                    jobs[job_id]['results']['selective_no_guitar'] = {
                        'file': f"/api/download/{job_id}/{base_filename}_selective_no_guitar.wav",
                        'description': f'Guitar removed selectively (strength: {guitar_filter_strength})'
                    }
                
                elif operation == 'remove_vocals':
                    output_path = os.path.join(output_dir, f"{base_filename}_instrumental.wav")
                    remove_vocals(input_filepath, output_path)
                    jobs[job_id]['results']['instrumental'] = {
                        'file': f"/api/download/{job_id}/{base_filename}_instrumental.wav",
                        'description': 'Instrumental version (vocals removed)'
                    }
                
                completed_operations += 1
                progress = int((completed_operations / total_operations) * 90) + 5  # 5-95%
                jobs[job_id]['progress'] = progress
                
            except Exception as e:
                jobs[job_id]['results'][f'{operation}_error'] = {
                    'error': str(e),
                    'description': f'Error processing {operation}'
                }
        
        # Mark job as completed
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['message'] = 'Processing completed successfully!'
        jobs[job_id]['completed_at'] = datetime.now().isoformat()
        
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['message'] = f'Processing failed: {str(e)}'
        jobs[job_id]['error'] = str(e)

@app.route('/api/status/<job_id>')
def get_job_status(job_id):
    """Get the current status of a processing job."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(jobs[job_id])

@app.route('/api/download/<job_id>/<path:filename>')
def download_file(job_id, filename):
    """Download a processed audio file."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    # Construct file path
    if filename.startswith('stems/'):
        # Handle stem files
        stem_filename = filename[6:]  # Remove 'stems/' prefix
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], job_id, 'stems', stem_filename)
    else:
        # Handle regular processed files
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], job_id, filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file_path, as_attachment=True)

@app.route('/api/jobs/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """Delete a job and its associated files."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    try:
        # Remove uploaded file
        job = jobs[job_id]
        if os.path.exists(job['filepath']):
            os.remove(job['filepath'])
        
        # Remove output directory
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        # Remove job from memory
        del jobs[job_id]
        
        return jsonify({'message': 'Job deleted successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("SplitSonic Web Interface starting...")
    print("Upload folder:", app.config['UPLOAD_FOLDER'])
    print("Output folder:", app.config['OUTPUT_FOLDER'])
    print("Access the app at: http://localhost:5000")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True) 