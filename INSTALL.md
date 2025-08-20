# Installation Instructions

## Prerequisites

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install FFmpeg** (required for audio processing):
   ```bash
   brew install ffmpeg
   ```

3. (Optional) **Install Conda/Miniconda**:
   ```bash
   brew install miniconda
   ```

## Environment Setup

1. **Create a new conda environment (optional)**:
   ```bash
   conda create -n guitartracker python=3.10
   conda activate guitartracker
   ```

2. **Install PyTorch**:
   ```bash
   conda install pytorch torchaudio -c pytorch
   ```

3. **Install remaining dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Alternative Installation (pip virtualenv)

If you prefer using pip only:

1. **Create a virtual environment**:
   ```bash
   python3 -m venv guitartracker_env
   source guitartracker_env/bin/activate
   ```

2. **Install all dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Verification

Test your installation:

```bash
python -c "import torch, torchaudio, demucs, librosa, soundfile; print('All dependencies installed successfully!')"
```

## Usage

Once installed, you can run the guitar tracker:

```bash
python modelsplitter.py your_audio_file.mp3
```

## Troubleshooting

### Common Issues on macOS (Apple Silicon):

1. **If you get "No module named 'demucs'" error**:
   ```bash
   pip install --upgrade demucs
   ```

2. **If FFmpeg is not found**:
   ```bash
   brew reinstall ffmpeg
   ```

3. **If PyTorch doesn't use Apple Silicon acceleration**:
   ```bash
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```
   Should return `True` for proper Apple Silicon support.

4. **Memory issues with large audio files**:
   - Very large files might cause high memory usage
   - Consider splitting large files or reducing the audio quality before processing

## Performance Notes

- On Apple Silicon, PyTorch uses the Metal Performance Shaders (MPS) backend when available.
- Processing times vary based on audio length and hardware.
- As a rough guide, a 3‑minute song may take a few minutes on CPU; NVIDIA GPUs are faster.