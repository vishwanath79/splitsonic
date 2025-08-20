import os
import torch
import sys
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from scipy import signal

"""
Guitar Stem Separator and Remover

This module provides functionality to:
1. Separate audio tracks into stems (drums, bass, vocals, other) using Demucs
2. Remove guitar from audio tracks while preserving other instruments
3. Selectively remove guitar frequencies while preserving keyboards
4. Remove vocals to create instrumental versions

The module offers four main functions:
- separate_stems: Separates an audio file into individual instrument stems
- remove_guitar: Removes the "other" stem which contains guitars (traditional method)
- remove_guitar_keep_keyboards: Uses spectral filtering to selectively remove guitar frequencies
- remove_vocals: Removes only vocals while keeping all other instruments (drums, bass, guitar, keyboards)
"""

def separate_stems(audio_path, output_dir, model_name="htdemucs"):
    """
    Separate audio file into stems using the Demucs model
    
    This function takes an audio file, processes it through the Demucs model,
    and saves each instrument stem as a separate audio file.
    
    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save the separated stems
        model_name: Name of the demucs model to use (default: "htdemucs")
    
    Returns:
        List of source names (stems) that were separated
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"Processing file: {audio_path}")
        
        # Get the filename without extension to use as prefix
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Load audio file - explicitly forcing a 2D tensor [channels, time]
        # This bypasses torchaudio's potential batch dimension
        waveform_tmp, sample_rate = torchaudio.load(audio_path)
        
        print(f"Original waveform shape after loading: {waveform_tmp.shape}, dtype: {waveform_tmp.dtype}")
        
        # Force reshape to 2D by copying data if needed
        if len(waveform_tmp.shape) == 3:  # Has unwanted batch dimension
            waveform = waveform_tmp.squeeze(0)  # Remove batch dimension
        else:
            waveform = waveform_tmp
        
        print(f"Reshaped to 2D: {waveform.shape}")
        
        # Convert to mono if needed and ensure proper format
        if waveform.shape[0] > 2:
            waveform = waveform[:2]  # Keep only first two channels if more than stereo
        elif waveform.shape[0] == 1:
            # If mono, duplicate to stereo as Demucs expects stereo input
            waveform = torch.cat([waveform, waveform], dim=0)
        
        print(f"After channel processing: {waveform.shape}")
        
        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model first to avoid any tensor modifications that might be unnecessary
        model = get_model(model_name)
        model.to(device)
        model.eval()
        
        print(f"Model loaded: {model_name}")
        
        # Now prepare the tensor for the model
        waveform = waveform.to(device)
        
        # Check if this really is 2D before adding batch dimension
        if len(waveform.shape) != 2:
            raise ValueError(f"Expected 2D tensor but got {waveform.shape}")
        
        # Add batch dimension - the tensor is now 3D [1, channels, time]
        waveform = waveform.unsqueeze(0)
        
        # Ensure floating point
        waveform = waveform.float()
        
        print(f"Final tensor shape before model: {waveform.shape}")
        
        # Apply model - Demucs expects input shape [B, C, T]
        with torch.no_grad():
            sources = apply_model(model, waveform, device=device)
            print(f"Output sources shape: {sources.shape}")
        
        # Get source names from model
        source_names = model.sources
        print(f"Source names: {source_names}")
        
        # Save each stem - Fixed indexing to properly handle the returned tensor
        print(f"Saving stems to {output_dir}...")
        
        # The sources output has a different format than expected
        # For htdemucs, the output shape is [batch, sources, channels, time]
        if len(sources.shape) == 4:
            batch_size, num_sources, num_channels, time_len = sources.shape
            
            if num_sources == len(source_names):
                # This is the correct format - iterate over sources
                for i, name in enumerate(source_names):
                    # Get this source - shape [batch, channels, time]
                    this_source = sources[:, i]  # Note: indexing dim 1 for sources
                    
                    # Move to CPU for saving
                    this_source = this_source.cpu()
                    
                    # Remove batch dimension if it's 1
                    if this_source.shape[0] == 1:
                        this_source = this_source.squeeze(0)  # Now [channels, time]
                    
                    # Save the stem with the original filename as a prefix
                    output_path = os.path.join(output_dir, f"{basename}_{name}.wav")
                    print(f"Saving {name} stem with shape {this_source.shape} to {output_path}")
                    torchaudio.save(output_path, this_source, sample_rate)
            else:
                raise ValueError(f"Number of sources ({num_sources}) doesn't match source_names length ({len(source_names)})")
        else:
            raise ValueError(f"Unexpected sources shape: {sources.shape}")
        
        print(f"Successfully separated stems to {output_dir}")
        return source_names
    except Exception as e:
        print(f"Error during stem separation: {str(e)}")
        if 'sources' in locals():
            print(f"Sources shape: {sources.shape}")
        if 'waveform' in locals():
            print(f"Waveform shape: {waveform.shape}, device: {waveform.device}, dtype: {waveform.dtype}")
        elif 'waveform_tmp' in locals():
            print(f"Initial waveform shape: {waveform_tmp.shape}")
        raise

def remove_guitar(audio_path, output_path=None, model_name="htdemucs"):
    """
    Remove guitar from an audio file using the Demucs model.
    
    This function uses the traditional approach of completely removing the "other" stem
    which typically contains guitar and other instruments like keyboards.
    It isolates all the stems except "other" and mixes them back together.
    
    Args:
        audio_path: Path to the input audio file
        output_path: Path to save the processed audio (without guitar). 
                     If None, uses input filename as prefix.
        model_name: Name of the demucs model to use (default: "htdemucs")
    
    Returns:
        Path to the processed audio file
    """
    # Get the filename without extension to use as prefix
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    
    # If output_path is not specified, create one using the input filename as a prefix
    if output_path is None:
        output_dir = os.path.dirname(audio_path) if os.path.dirname(audio_path) else "."
        output_path = os.path.join(output_dir, f"{basename}_no_guitar.wav")
    
    # Create temporary directory for stems
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_stems")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        print(f"Processing file for guitar removal: {audio_path}")
        print(f"Output will be saved to: {output_path}")
        
        # Load audio file - explicitly forcing a 2D tensor [channels, time]
        waveform_tmp, sample_rate = torchaudio.load(audio_path)
        
        print(f"Original waveform shape after loading: {waveform_tmp.shape}, dtype: {waveform_tmp.dtype}")
        
        # Force reshape to 2D by copying data if needed
        if len(waveform_tmp.shape) == 3:  # Has unwanted batch dimension
            waveform = waveform_tmp.squeeze(0)  # Remove batch dimension
        else:
            waveform = waveform_tmp
        
        print(f"Reshaped to 2D: {waveform.shape}")
        
        # Convert to mono if needed and ensure proper format
        if waveform.shape[0] > 2:
            waveform = waveform[:2]  # Keep only first two channels if more than stereo
        elif waveform.shape[0] == 1:
            # If mono, duplicate to stereo as Demucs expects stereo input
            waveform = torch.cat([waveform, waveform], dim=0)
        
        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model first
        model = get_model(model_name)
        model.to(device)
        model.eval()
        
        # Now prepare the tensor for the model
        waveform = waveform.to(device)
        
        # Check if this really is 2D before adding batch dimension
        if len(waveform.shape) != 2:
            raise ValueError(f"Expected 2D tensor but got {waveform.shape}")
        
        # Add batch dimension - the tensor is now 3D [1, channels, time]
        waveform = waveform.unsqueeze(0)
        
        # Ensure floating point
        waveform = waveform.float()
        
        print(f"Final tensor shape before model: {waveform.shape}")
        
        # Process audio through Demucs model
        with torch.no_grad():
            sources = apply_model(model, waveform, device=device)
            print(f"Output sources shape: {sources.shape}")
        
        # Get source names
        source_names = model.sources
        print(f"Source names: {source_names}")
        
        # Find the index of "other" stem (which typically contains guitar)
        other_idx = source_names.index("other") if "other" in source_names else None
        
        # For htdemucs, the output shape is [batch, sources, channels, time]
        if len(sources.shape) != 4:
            raise ValueError(f"Unexpected sources shape: {sources.shape}")
            
        batch_size, num_sources, num_channels, time_len = sources.shape
        
        # If we can't find "other", try a different approach
        if other_idx is None:
            print("Warning: 'other' stem not found in model sources. Guitar isolation may not work as expected.")
            # Just create a mix of drums, bass, and vocals as a fallback
            mix = None
            for i, name in enumerate(source_names):
                if name in ["drums", "bass", "vocals"]:
                    source = sources[:, i]  # Get source i
                    if mix is None:
                        mix = source.clone()
                    else:
                        mix += source
        else:
            # Create a mix of everything except "other"
            mix = None
            for i, name in enumerate(source_names):
                if i != other_idx:
                    source = sources[:, i]  # Get source i
                    if mix is None:
                        mix = source.clone()
                    else:
                        mix += source
        
        # Save the result
        if mix is not None:
            # Move to CPU
            mix = mix.cpu()
            
            # Remove batch dimension if needed
            if mix.shape[0] == 1:
                mix = mix.squeeze(0)  # Now [channels, time]
                
            print(f"Final mix shape: {mix.shape}")
            torchaudio.save(output_path, mix, sample_rate)
            print(f"Successfully saved no-guitar version to {output_path}")
            
        # Clean up temporary directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            
        return output_path
    except Exception as e:
        print(f"Error during guitar removal: {str(e)}")
        # Print debug information
        if 'sources' in locals():
            print(f"Sources shape: {sources.shape}")
        if 'waveform' in locals():
            print(f"Waveform shape: {waveform.shape}, device: {waveform.device}, dtype: {waveform.dtype}")
        elif 'waveform_tmp' in locals():
            print(f"Initial waveform shape: {waveform_tmp.shape}")
        raise

def remove_guitar_keep_keyboards(audio_path, output_path=None, model_name="htdemucs", guitar_filter_strength=0.9):
    """
    Remove guitar while preserving keyboards from an audio file.
    
    This function uses advanced spectral filtering to selectively attenuate guitar frequencies
    while preserving keyboard sounds in the "other" stem. It combines this processed
    "other" stem with the drums, bass, and vocals to create a version with reduced
    guitar but intact keyboards.
    
    Args:
        audio_path: Path to the input audio file
        output_path: Path to save the processed audio. If None, uses input filename as prefix.
        model_name: Name of the demucs model to use (default: "htdemucs")
        guitar_filter_strength: Strength of the guitar filtering (0.0-1.0) 
                               Higher values remove more guitar (default: 0.9)
    
    Returns:
        Path to the processed audio file
    """
    # Get the filename without extension to use as prefix
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    
    # If output_path is not specified, create one using the input filename as a prefix
    if output_path is None:
        output_dir = os.path.dirname(audio_path) if os.path.dirname(audio_path) else "."
        output_path = os.path.join(output_dir, f"{basename}_no_guitar_keep_keys.wav")
    
    # Create temporary directory for stems
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_stems")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        print(f"Processing file for selective guitar removal: {audio_path}")
        print(f"Output will be saved to: {output_path}")
        print(f"Guitar filter strength: {guitar_filter_strength}")
        
        # First, separate the stems using Demucs
        # Load audio file and create the model as before
        waveform_tmp, sample_rate = torchaudio.load(audio_path)
        
        # Force reshape to 2D by copying data if needed
        if len(waveform_tmp.shape) == 3:
            waveform = waveform_tmp.squeeze(0)
        else:
            waveform = waveform_tmp
        
        # Convert to mono if needed and ensure proper format
        if waveform.shape[0] > 2:
            waveform = waveform[:2]
        elif waveform.shape[0] == 1:
            waveform = torch.cat([waveform, waveform], dim=0)
        
        # Get device and prepare model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = get_model(model_name)
        model.to(device)
        model.eval()
        
        # Prepare input tensor
        waveform = waveform.to(device).unsqueeze(0).float()
        
        # Apply model
        with torch.no_grad():
            sources = apply_model(model, waveform, device=device)
        
        # Get source names
        source_names = model.sources
        print(f"Source names: {source_names}")
        
        # Find the index of "other" stem (which typically contains both guitar and keyboards)
        other_idx = source_names.index("other") if "other" in source_names else None
        
        if other_idx is None:
            print("Warning: 'other' stem not found. Using default guitar removal method.")
            return remove_guitar(audio_path, output_path, model_name)
        
        # For htdemucs, the output shape is [batch, sources, channels, time]
        if len(sources.shape) != 4:
            raise ValueError(f"Unexpected sources shape: {sources.shape}")
        
        # Extract the "other" stem containing guitars and keyboards
        other_stem = sources[:, other_idx].cpu().squeeze(0).numpy()  # Now [channels, time]
        
        # Save this stem temporarily for analysis
        other_stem_path = os.path.join(temp_dir, f"{basename}_other_stem.wav")
        torchaudio.save(other_stem_path, torch.tensor(other_stem), sample_rate)
        
        # Extract the "drums", "bass", and "vocals" stems
        drums_bass_vocals = None
        for i, name in enumerate(source_names):
            if name in ["drums", "bass", "vocals"]:
                source = sources[:, i].cpu()  # Get source i
                if drums_bass_vocals is None:
                    drums_bass_vocals = source.clone()
                else:
                    drums_bass_vocals += source
        
        # Convert to numpy array and remove batch dimension
        if drums_bass_vocals is not None:
            drums_bass_vocals = drums_bass_vocals.squeeze(0).numpy()  # Now [channels, time]
        else:
            drums_bass_vocals = np.zeros_like(other_stem)
        
        # Process the "other" stem to extract keyboard and remove guitar
        # Since guitars and keyboards can occupy similar frequency ranges,
        # we'll use advanced spectral processing techniques
        
        # Now process the "other" stem using spectral processing 
        # to attenuate guitar frequencies while preserving keyboards
        processed_other = []
        
        # STFT parameters - larger window size for better frequency resolution
        n_fft = 4096  # Larger FFT size for better frequency resolution
        hop_length = 1024  # Overlap factor for better reconstruction
        win_length = 4096  # Window size
        
        for channel in range(other_stem.shape[0]):
            # Load the audio channel
            y = other_stem[channel]
            
            # Compute the Short-Time Fourier Transform (STFT) with larger window
            D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            
            # Get magnitude and phase
            magnitude, phase = librosa.magphase(D)
            
            # Guitar typically has significant energy in specific frequency ranges
            # Create a multi-band filter to target different guitar characteristics
            freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
            guitar_mask = np.ones_like(magnitude)
            
            # Define multiple frequency bands to target different aspects of guitar sound
            # Low-mids (rhythm guitar)
            guitar_low_mids = (300, 800, 600, 0.95)  # (start_freq, end_freq, center_freq, attenuation)
            # Mids (guitar body)
            guitar_mids = (800, 2500, 1400, 0.95)    # Where most guitar energy is concentrated
            # Upper-mids (guitar presence/attack)
            guitar_hi_mids = (2500, 5000, 3500, 0.85)
            # High frequencies (pick/string noise)
            guitar_highs = (5000, 8000, 6500, 0.7)
            
            # List of frequency bands to process
            guitar_bands = [guitar_low_mids, guitar_mids, guitar_hi_mids, guitar_highs]
            
            # Process each frequency band
            for start_freq, end_freq, center_freq, band_strength in guitar_bands:
                # Calculate bin indices for this band
                idx_start = np.argmin(np.abs(freqs - start_freq))
                idx_end = np.argmin(np.abs(freqs - end_freq))
                idx_center = np.argmin(np.abs(freqs - center_freq))
                
                # Create a mask for this frequency band
                # Use triangle shape with maximum attenuation at the center frequency
                left_slope = np.arange(idx_start, idx_center) - idx_start
                if (idx_center - idx_start) > 0:  # Prevent division by zero
                    left_slope = left_slope / (idx_center - idx_start)
                
                right_slope = np.arange(idx_center, idx_end) - idx_center
                if (idx_end - idx_center) > 0:  # Prevent division by zero
                    right_slope = 1.0 - (right_slope / (idx_end - idx_center))
                
                # Combine slopes into a triangular mask
                band_mask = np.ones(idx_end - idx_start)
                band_mask[:len(left_slope)] = 1.0 - (band_strength * left_slope * guitar_filter_strength)
                band_mask[len(left_slope):] = 1.0 - (band_strength * right_slope * guitar_filter_strength)
                
                # Apply this band's mask to the full mask
                guitar_mask[idx_start:idx_end, :] = band_mask[:, np.newaxis]
            
            # Apply the mask to the magnitude
            filtered_magnitude = magnitude * guitar_mask
            
            # Additional transient suppression to reduce guitar pick attacks
            # This helps remove more of the guitar's percussive elements
            if guitar_filter_strength > 0.7:
                # Calculate the temporal envelope
                temporal_envelope = np.mean(np.abs(filtered_magnitude), axis=0)
                # Smooth the envelope
                window_size = 5
                smoothed_envelope = np.convolve(temporal_envelope, 
                                              np.ones(window_size)/window_size, 
                                              mode='same')
                # Detect transients (sharp increases in energy)
                transients = temporal_envelope > (1.5 * smoothed_envelope)
                # Attenuate frames with transients
                transient_mask = np.ones_like(filtered_magnitude)
                transient_mask[:, transients] = 0.7  # Reduce energy at transient locations
                filtered_magnitude = filtered_magnitude * transient_mask
            
            # Reconstruct the audio
            filtered_D = filtered_magnitude * phase
            processed_audio = librosa.istft(filtered_D, hop_length=hop_length, 
                                          win_length=win_length, length=len(y))
            
            processed_other.append(processed_audio)
        
        # Convert back to appropriate shape
        processed_other = np.array(processed_other)
        
        # Apply a small gain reduction to the processed "other" stem
        # This helps to further reduce any remaining guitar elements
        processed_other = processed_other * 0.85
        
        # Mix with the drums, bass, and vocals
        final_mix = drums_bass_vocals + processed_other
        
        # Save to output
        sf.write(output_path, final_mix.T, sample_rate)
        print(f"Successfully saved guitar-reduced version to {output_path}")
        
        # Clean up temporary directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            
        return output_path
    except Exception as e:
        print(f"Error during selective guitar removal: {str(e)}")
        # Print debug information
        import traceback
        traceback.print_exc()
        raise

def remove_vocals(audio_path, output_path=None, model_name="htdemucs"):
    """
    Remove vocals from an audio file using the Demucs model.
    
    This function removes only the vocal stem while preserving all other instruments
    including drums, bass, guitar, keyboards, and other instruments.
    It creates an instrumental version of the song.
    
    Args:
        audio_path: Path to the input audio file
        output_path: Path to save the processed audio (without vocals). 
                     If None, uses input filename as prefix.
        model_name: Name of the demucs model to use (default: "htdemucs")
    
    Returns:
        Path to the processed audio file
    """
    # Get the filename without extension to use as prefix
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    
    # If output_path is not specified, create one using the input filename as a prefix
    if output_path is None:
        output_dir = os.path.dirname(audio_path) if os.path.dirname(audio_path) else "."
        output_path = os.path.join(output_dir, f"{basename}_no_vocals.wav")
    
    # Create temporary directory for stems
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_stems")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        print(f"Processing file for vocal removal: {audio_path}")
        print(f"Output will be saved to: {output_path}")
        
        # Load audio file - explicitly forcing a 2D tensor [channels, time]
        waveform_tmp, sample_rate = torchaudio.load(audio_path)
        
        print(f"Original waveform shape after loading: {waveform_tmp.shape}, dtype: {waveform_tmp.dtype}")
        
        # Force reshape to 2D by copying data if needed
        if len(waveform_tmp.shape) == 3:  # Has unwanted batch dimension
            waveform = waveform_tmp.squeeze(0)  # Remove batch dimension
        else:
            waveform = waveform_tmp
        
        print(f"Reshaped to 2D: {waveform.shape}")
        
        # Convert to mono if needed and ensure proper format
        if waveform.shape[0] > 2:
            waveform = waveform[:2]  # Keep only first two channels if more than stereo
        elif waveform.shape[0] == 1:
            # If mono, duplicate to stereo as Demucs expects stereo input
            waveform = torch.cat([waveform, waveform], dim=0)
        
        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model first
        model = get_model(model_name)
        model.to(device)
        model.eval()
        
        # Now prepare the tensor for the model
        waveform = waveform.to(device)
        
        # Check if this really is 2D before adding batch dimension
        if len(waveform.shape) != 2:
            raise ValueError(f"Expected 2D tensor but got {waveform.shape}")
        
        # Add batch dimension - the tensor is now 3D [1, channels, time]
        waveform = waveform.unsqueeze(0)
        
        # Ensure floating point
        waveform = waveform.float()
        
        print(f"Final tensor shape before model: {waveform.shape}")
        
        # Process audio through Demucs model
        with torch.no_grad():
            sources = apply_model(model, waveform, device=device)
            print(f"Output sources shape: {sources.shape}")
        
        # Get source names
        source_names = model.sources
        print(f"Source names: {source_names}")
        
        # Find the index of "vocals" stem
        vocals_idx = source_names.index("vocals") if "vocals" in source_names else None
        
        # For htdemucs, the output shape is [batch, sources, channels, time]
        if len(sources.shape) != 4:
            raise ValueError(f"Unexpected sources shape: {sources.shape}")
            
        batch_size, num_sources, num_channels, time_len = sources.shape
        
        # If we can't find "vocals", try a different approach
        if vocals_idx is None:
            print("Warning: 'vocals' stem not found in model sources. Vocal removal may not work as expected.")
            # Just create a mix of drums, bass, and other as a fallback
            mix = None
            for i, name in enumerate(source_names):
                if name in ["drums", "bass", "other"]:
                    source = sources[:, i]  # Get source i
                    if mix is None:
                        mix = source.clone()
                    else:
                        mix += source
        else:
            # Create a mix of everything except "vocals"
            mix = None
            for i, name in enumerate(source_names):
                if i != vocals_idx:
                    source = sources[:, i]  # Get source i
                    if mix is None:
                        mix = source.clone()
                    else:
                        mix += source
        
        # Save the result
        if mix is not None:
            # Move to CPU
            mix = mix.cpu()
            
            # Remove batch dimension if needed
            if mix.shape[0] == 1:
                mix = mix.squeeze(0)  # Now [channels, time]
                
            print(f"Final mix shape: {mix.shape}")
            torchaudio.save(output_path, mix, sample_rate)
            print(f"Successfully saved no-vocals version to {output_path}")
            
        # Clean up temporary directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            
        return output_path
    except Exception as e:
        print(f"Error during vocal removal: {str(e)}")
        # Print debug information
        if 'sources' in locals():
            print(f"Sources shape: {sources.shape}")
        if 'waveform' in locals():
            print(f"Waveform shape: {waveform.shape}, device: {waveform.device}, dtype: {waveform.dtype}")
        elif 'waveform_tmp' in locals():
            print(f"Initial waveform shape: {waveform_tmp.shape}")
        raise

# Example usage
if __name__ == "__main__":
    try:
        # Check if a command line argument was provided
        if len(sys.argv) < 2:
            print("Usage: python modelsplitter.py <audio_file>")
            sys.exit(1)
            
        # Example audio file from command line argument
        audio_file = sys.argv[1]
        
        # Get the filename without extension to use as prefix
        basename = os.path.splitext(os.path.basename(audio_file))[0]
        
        print(f"Processing audio file: {audio_file}")
        
        # Example 1: Separate all stems into individual files
        print("\n=== Separating all stems ===")
        separate_stems(audio_file, f"./{basename}_stems")
        
        # Example 2: Remove guitar (traditional method - removes entire "other" stem)
        print("\n=== Removing guitar (traditional method) ===")
        remove_guitar(audio_file)
        
        # Example 3: Remove guitar while preserving keyboards (advanced spectral filtering)
        print("\n=== Removing guitar while keeping keyboards ===")
        remove_guitar_keep_keyboards(audio_file, guitar_filter_strength=0.9)
        
        # Example 4: Remove vocals to create instrumental version
        print("\n=== Removing vocals (creating instrumental) ===")
        remove_vocals(audio_file)
        
        print("\nProcessing complete!")
        print(f"Output files created:")
        print(f"  - {basename}_stems/ (directory with individual stems)")
        print(f"  - {basename}_no_guitar.wav (no guitar, traditional method)")
        print(f"  - {basename}_no_guitar_keep_keys.wav (guitar removed, keyboards preserved)")
        print(f"  - {basename}_no_vocals.wav (instrumental version)")
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
