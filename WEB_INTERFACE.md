# SplitSonic Web Interface

A modern, single-page web application that provides an intuitive interface for the SplitSonic audio source separation tool.

## 🌐 Features

### User Interface
- **Modern Design**: Clean, responsive interface with beautiful gradients and animations
- **Drag & Drop Upload**: Simply drag audio files onto the upload area
- **Real-time Progress**: Live progress tracking with percentage and status updates
- **Multiple Processing Options**: Choose from 4 different audio processing modes
- **Instant Downloads**: Direct download links for all processed files
- **Mobile Responsive**: Works seamlessly on desktop, tablet, and mobile devices

### Processing Options
1. **Separate All Stems** - Extract drums, bass, vocals, and other instruments
2. **Remove Guitar (Traditional)** - Quick guitar removal using stem separation
3. **Selective Guitar Removal** - Advanced filtering with adjustable strength
4. **Remove Vocals (Instrumental)** - Create instrumental versions

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install main SplitSonic dependencies
pip install -r requirements.txt

# Install web interface dependencies  
pip install -r app_requirements.txt
```

### 2. Launch the Web Interface
```bash
# Option 1: Using the launcher script (recommended)
python run_web_interface.py

# Option 2: Direct Flask app
python app.py
```

### 3. Access the Interface
Open your browser and navigate to: **http://localhost:5000**

## 🏗️ Architecture

### Backend (Flask API)
```
app.py                 # Main Flask application
├── /api/upload       # File upload endpoint
├── /api/process      # Start processing job
├── /api/status/<id>  # Check job status
├── /api/download/... # Download processed files
└── /api/jobs/<id>    # Delete job and files
```

### Frontend (Single Page App)
```
templates/index.html   # Complete web interface
├── HTML Structure    # Semantic markup
├── CSS Styling      # Modern responsive design
└── JavaScript App   # SplitSonicApp class
```

### File Structure
```
splitsonic/
├── app.py                    # Flask web server
├── run_web_interface.py     # Launcher script
├── app_requirements.txt     # Web dependencies
├── templates/
│   └── index.html          # Web interface
├── uploads/                # Temporary uploaded files
├── outputs/               # Processed audio files
├── modelsplitter.py      # Core audio processing
└── ... (other files)
```

## 🔧 API Documentation

### Upload File
```http
POST /api/upload
Content-Type: multipart/form-data

Response:
{
  "job_id": "uuid-string",
  "filename": "song.mp3", 
  "message": "File uploaded successfully"
}
```

### Start Processing
```http
POST /api/process
Content-Type: application/json

{
  "job_id": "uuid-string",
  "operations": ["separate_stems", "remove_guitar"],
  "guitar_filter_strength": 0.9
}

Response:
{
  "message": "Processing started",
  "job_id": "uuid-string"
}
```

### Check Status
```http
GET /api/status/{job_id}

Response:
{
  "status": "processing",
  "progress": 45,
  "message": "Processing: Remove Guitar",
  "results": {}
}
```

### Download Files
```http
GET /api/download/{job_id}/{filename}
# Returns audio file for download
```

## 🎨 User Experience

### Upload Process
1. **File Selection**: Drag & drop or click to browse
2. **Validation**: Automatic file type and size checking  
3. **Confirmation**: Visual feedback with file information

### Processing Flow
1. **Option Selection**: Click cards to select processing types
2. **Advanced Settings**: Adjust guitar filter strength if needed
3. **Progress Tracking**: Real-time status updates with progress bar
4. **Results Display**: Organized download cards for each output

### Download Experience
- **Individual Downloads**: Separate download buttons for each result
- **Stem Downloads**: Individual buttons for drums, bass, vocals, other
- **Error Handling**: Clear error messages for failed operations

## ⚙️ Configuration

### File Limits
- **Maximum Size**: 500MB per file
- **Supported Formats**: MP3, WAV, FLAC, M4A, AAC, OGG, WMA
- **Processing Time**: Varies by file size and selected operations

### Server Settings
```python
# In app.py - modify these for your needs
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # File size limit
app.config['UPLOAD_FOLDER'] = 'uploads'               # Upload directory
app.config['OUTPUT_FOLDER'] = 'outputs'               # Output directory
```

### Advanced Options
- **Guitar Filter Strength**: 0.0 (gentle) to 1.0 (aggressive)
- **Model Selection**: Currently uses `htdemucs` (high quality)
- **Auto-cleanup**: Files older than 1 hour are automatically removed

## 🛠️ Development

### Frontend Development
The web interface is built with vanilla JavaScript using modern ES6+ features:

```javascript
class SplitSonicApp {
  constructor() {
    this.currentJobId = null;
    this.uploadedFile = null;
    this.progressInterval = null;
  }
  
  // Key methods:
  // - File upload handling
  // - Progress tracking  
  // - Results display
  // - Error handling
}
```

### Adding New Features
1. **Backend**: Add new API endpoints in `app.py`
2. **Frontend**: Extend the `SplitSonicApp` class
3. **Processing**: Add new functions to `modelsplitter.py`
4. **UI**: Update the HTML template with new options

## 🐛 Troubleshooting

### Common Issues

**Error: "Module not found"**
```bash
# Install missing dependencies
pip install flask werkzeug
```

**Error: "Port already in use"**
```bash
# Change port in app.py or kill existing process
lsof -ti:5000 | xargs kill -9  # macOS/Linux
```

**Error: "File too large"**
- Check file size (500MB limit)
- Compress audio file or adjust `MAX_CONTENT_LENGTH`

**Slow Processing**
- Enable GPU acceleration
- Use smaller files for testing
- Check system resources

### Debug Mode
Set `debug=True` in `app.py` for detailed error messages and auto-reload.

## 🔒 Security Notes

### Production Deployment
- Change `SECRET_KEY` in `app.py`
- Use a production WSGI server (gunicorn/waitress)
- Implement user authentication if needed
- Add file upload validation
- Configure reverse proxy (nginx)

### File Security
- Files are automatically cleaned up after 1 hour
- Each upload gets a unique UUID-based filename
- No file execution, only audio processing

## 📱 Mobile Support

The interface is fully responsive and works on:
- **Desktop**: Full feature set with drag & drop
- **Tablet**: Touch-friendly interface
- **Mobile**: Optimized layout with stacked options

## 🤝 Contributing

To contribute to the web interface:

1. Fork the repository
2. Create a feature branch
3. Test the interface thoroughly
4. Submit a pull request

### Testing Checklist
- [ ] File upload with various audio formats
- [ ] All processing options work correctly
- [ ] Progress tracking functions properly
- [ ] Downloads work for all result types
- [ ] Mobile responsive design
- [ ] Error handling displays appropriate messages

## 📄 License

Same license as the main SplitSonic project. 