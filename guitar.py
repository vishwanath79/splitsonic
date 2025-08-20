
# List available models
def list_models():
    import subprocess
    result = subprocess.run(['audio-separator', '--list_models'], capture_output=True, text=True)
    print(result.stdout)

# Run separation
def separate_audio(audio_path):
    import subprocess
    subprocess.run(['audio-separator', '--model_filename', 'htdemucs_ft.yaml', '--output_dir', './separated', audio_path])


if __name__ == "__main__":
    list_models()
    separate_audio('oyhmtv.mp3')