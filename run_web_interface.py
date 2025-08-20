#!/usr/bin/env python3
"""
SplitSonic Web Interface Launcher
Simple script to start the web interface for SplitSonic
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import flask
        import modelsplitter
        print("Flask and SplitSonic modules found")
        return True
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("\nPlease install dependencies:")
        print("pip install -r requirements.txt")
        print("pip install -r app_requirements.txt")
        return False

def main():
    """Launch the SplitSonic web interface."""
    print("SplitSonic Web Interface Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('modelsplitter.py'):
        print("Error: modelsplitter.py not found")
        print("Please run this script from the SplitSonic project directory")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("Created necessary directories")
    print("Starting Flask application...")
    print("\n" + "=" * 50)
    print("Access SplitSonic at: http://localhost:5000")
    print("Upload limit: 500MB per file")
    print("Supported formats: MP3, WAV, FLAC, M4A, AAC, OGG, WMA")
    print("=" * 50)
    
    # Import and run the Flask app
    try:
        from app import app
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"Error starting web interface: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 