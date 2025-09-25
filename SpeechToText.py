#!/usr/bin/env python3
"""
Enhanced Speech-to-Text Converter with Audience Detection
Combines all functionality into a single, comprehensive script
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Import audio processing libraries
try:
    import librosa
    import noisereduce as nr
    from scipy import signal
    from scipy.signal import butter, filtfilt
    import soundfile as sf
    import whisper
except ImportError as e:
    print(f"Missing audio processing library: {e}")
    print("Please install required packages: pip install -r requirements.txt")
    sys.exit(1)

def format_timestamp(seconds):
    """Format timestamp as [MM:SS]"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"[{minutes:02d}:{seconds:02d}]"

def get_audio_file():
    """Get audio file path from user input"""
    print("\nAvailable audio files in current directory:")
    audio_files = [f for f in os.listdir('.') if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac'))]
    
    if not audio_files:
        print("No audio files found in current directory.")
        return None
    
    for i, file in enumerate(audio_files, 1):
        print(f"{i}. {file}")
    
    while True:
        try:
            choice = input(f"\nSelect audio file (1-{len(audio_files)}) or enter custom path: ").strip()
            
            if choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(audio_files):
                    return audio_files[choice_num - 1]
                else:
                    print(f"Please enter a number between 1 and {len(audio_files)}")
                    continue
            
            if os.path.exists(choice):
                return choice
            else:
                print(f"File not found: {choice}")
                print("Please enter a valid file path or number selection.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            return None
        except Exception as e:
            print(f"Error: {e}")
            continue

def preprocess_audio(audio_file, target_sr=16000):
    """Enhanced audio preprocessing with denoising and enhancement"""
    print("Loading and preprocessing audio...")
    
    # Load audio with librosa for better processing
    audio, sr = librosa.load(audio_file, sr=target_sr, mono=True)
    
    print(f"Original audio: {len(audio)/sr:.2f} seconds, {sr} Hz")
    
    # Apply noise reduction
    print("Applying noise reduction...")
    audio_denoised = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.8)
    
    # Apply high-pass filter to remove low-frequency noise
    print("Applying high-pass filter...")
    nyquist = sr * 0.5
    high_cutoff = 80  # Hz
    high = high_cutoff / nyquist
    b, a = butter(4, high, btype='high')
    audio_filtered = filtfilt(b, a, audio_denoised)
    
    # Normalize audio
    audio_normalized = librosa.util.normalize(audio_filtered)
    
    # Apply dynamic range compression
    print("Applying dynamic range compression...")
    audio_compressed = librosa.effects.preemphasis(audio_normalized)
    
    print("Audio preprocessing completed!")
    return audio_compressed, sr

def detect_speaker_changes(audio, sr, segment_length=3.0):
    """Detect potential speaker changes and audience participation"""
    print("Analyzing audio for speaker changes...")
    
    # Calculate energy and spectral features
    hop_length = int(sr * 0.1)  # 100ms windows
    frame_length = int(sr * 0.5)  # 500ms frames
    
    # Energy-based voice activity detection
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Spectral centroid for voice characteristics
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
    
    # Zero crossing rate for speech characteristics
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Combine features to detect speaker changes
    features = np.vstack([energy, spectral_centroid, zcr])
    features_norm = (features - features.mean(axis=1, keepdims=True)) / (features.std(axis=1, keepdims=True) + 1e-8)
    
    # Simple change point detection
    change_points = []
    window_size = int(segment_length * sr / hop_length)
    
    for i in range(window_size, len(features_norm[0]) - window_size):
        before = features_norm[:, i-window_size:i].mean(axis=1)
        after = features_norm[:, i:i+window_size].mean(axis=1)
        
        # Calculate distance between feature vectors
        distance = np.linalg.norm(after - before)
        if distance > 0.5:  # Threshold for speaker change
            change_points.append(i * hop_length / sr)
    
    print(f"Detected {len(change_points)} potential speaker changes")
    return change_points

def segment_audio_by_speakers(audio, sr, change_points):
    """Segment audio based on detected speaker changes"""
    print("Segmenting audio by speakers...")
    
    segments = []
    start_time = 0
    
    for change_point in change_points:
        if change_point - start_time > 1.0:  # Minimum segment length
            segments.append((start_time, change_point))
            start_time = change_point
    
    # Add final segment
    if len(audio) / sr - start_time > 1.0:
        segments.append((start_time, len(audio) / sr))
    
    print(f"Created {len(segments)} audio segments")
    return segments

def detect_audience_participation(audio, sr):
    """Specifically detect audience participation and questions"""
    print("Detecting audience participation...")
    
    # Parameters for audience detection
    hop_length = int(sr * 0.05)  # 50ms windows for fine detection
    frame_length = int(sr * 0.2)  # 200ms frames
    
    # Energy analysis
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Voice activity detection with lower threshold for audience
    energy_threshold = np.percentile(energy, 20)  # Lower threshold for audience
    voice_activity = energy > energy_threshold
    
    # Detect sudden changes in energy (audience reactions)
    energy_diff = np.diff(energy)
    sudden_changes = np.abs(energy_diff) > np.std(energy_diff) * 2
    
    # Combine features for audience detection
    audience_segments = []
    in_audience_segment = False
    segment_start = 0
    
    for i, (is_voice, is_sudden) in enumerate(zip(voice_activity, np.concatenate([[False], sudden_changes]))):
        time_point = i * hop_length / sr
        
        if is_voice and not in_audience_segment:
            # Start of potential audience segment
            in_audience_segment = True
            segment_start = time_point
        elif not is_voice and in_audience_segment:
            # End of audience segment
            if time_point - segment_start > 0.5:  # Minimum duration
                audience_segments.append((segment_start, time_point))
            in_audience_segment = False
        elif is_sudden and in_audience_segment:
            # Mark as audience reaction
            audience_segments.append((segment_start, time_point))
            segment_start = time_point
    
    # Handle case where segment continues to end
    if in_audience_segment and len(audio) / sr - segment_start > 0.5:
        audience_segments.append((segment_start, len(audio) / sr))
    
    print(f"Detected {len(audience_segments)} audience participation segments")
    return audience_segments

def enhance_audience_audio(audio, sr, audience_segments):
    """Apply specific enhancements to audience segments"""
    print("Enhancing audience audio segments...")
    
    enhanced_audio = audio.copy()
    
    for start_time, end_time in audience_segments:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Extract audience segment
        audience_segment = audio[start_sample:end_sample]
        
        # Apply specific enhancements for audience
        # 1. Boost high frequencies for clarity
        audience_enhanced = librosa.effects.preemphasis(audience_segment, coef=0.97)
        
        # 2. Apply gentle compression
        audience_compressed = np.tanh(audience_enhanced * 2) * 0.8
        
        # 3. Normalize
        audience_normalized = librosa.util.normalize(audience_compressed)
        
        # Replace in original audio
        enhanced_audio[start_sample:end_sample] = audience_normalized
    
    return enhanced_audio

def transcribe_with_enhanced_model(audio_file, model_size="large-v3"):
    """Use enhanced Whisper model with optimized parameters"""
    print(f"Loading enhanced Whisper model ({model_size})...")
    
    model = whisper.load_model(model_size)
    
    print("Starting enhanced transcription...")
    
    # Enhanced transcription with better parameters
    result = model.transcribe(
        audio_file,
        word_timestamps=True,
        verbose=False,
        language="en",  # Specify language for better accuracy
        temperature=0.0,  # Deterministic output
        compression_ratio_threshold=2.4,  # Better compression detection
        logprob_threshold=-1.0,  # Lower threshold for better detection
        no_speech_threshold=0.6,  # Better silence detection
        condition_on_previous_text=True,  # Use context
        fp16=False  # Use full precision for better accuracy
    )
    
    return result

def identify_speakers(segments, result):
    """Identify different speakers based on audio characteristics"""
    print("Identifying speakers...")
    
    speaker_segments = []
    speaker_count = 1
    
    for i, segment in enumerate(segments):
        start_time, end_time = segment
        
        # Find corresponding text segments
        segment_texts = []
        for text_segment in result["segments"]:
            if start_time <= text_segment["start"] < end_time:
                segment_texts.append(text_segment)
        
        if segment_texts:
            # Simple heuristic: longer segments are likely main speaker
            segment_duration = end_time - start_time
            if segment_duration > 10:  # Longer segments
                speaker = "Main Speaker"
            elif segment_duration > 3:  # Medium segments
                speaker = f"Speaker {speaker_count}"
                speaker_count += 1
            else:  # Short segments (likely audience)
                speaker = "Audience"
            
            speaker_segments.append({
                'speaker': speaker,
                'start': start_time,
                'end': end_time,
                'text_segments': segment_texts
            })
    
    return speaker_segments

def classify_speaker_type(segment, duration, energy_level, spectral_features):
    """Classify whether segment is main speaker or audience"""
    if duration > 15:  # Long segments are likely main speaker
        return "Main Speaker"
    elif duration > 5:  # Medium segments
        if energy_level > 0.1:  # High energy
            return "Main Speaker"
        else:
            return "Audience Member"
    else:  # Short segments
        if energy_level > 0.05:  # Moderate energy
            return "Audience Question"
        else:
            return "Audience Reaction"

def main():
    """Main function with enhanced audio processing and audience detection"""
    print("=" * 60)
    print("ENHANCED SPEECH-TO-TEXT CONVERTER")
    print("With Audience Detection & Audio Enhancement")
    print("=" * 60)
    
    # Get audio file from user
    audio_file = get_audio_file()
    if not audio_file:
        return
    
    try:
        print(f"\nProcessing audio file: {audio_file}")
        
        # Step 1: Preprocess audio
        audio_processed, sr = preprocess_audio(audio_file)
        
        # Save processed audio temporarily
        temp_audio_file = "temp_processed_audio.wav"
        try:
            sf.write(temp_audio_file, audio_processed, sr)
        except Exception as e:
            print(f"Error saving processed audio: {e}")
            return
        
        # Step 2: Detect speaker changes
        change_points = detect_speaker_changes(audio_processed, sr)
        
        # Step 3: Segment audio by speakers
        segments = segment_audio_by_speakers(audio_processed, sr, change_points)
        
        # Step 4: Detect audience participation
        audience_segments = detect_audience_participation(audio_processed, sr)
        
        # Step 5: Enhance audience audio
        audio_enhanced = enhance_audience_audio(audio_processed, sr, audience_segments)
        
        # Save enhanced audio
        temp_enhanced_file = "temp_enhanced_audio.wav"
        try:
            sf.write(temp_enhanced_file, audio_enhanced, sr)
        except Exception as e:
            print(f"Error saving enhanced audio: {e}")
            return
        
        # Step 6: Transcribe with enhanced model
        try:
            result = transcribe_with_enhanced_model(temp_enhanced_file, "large-v3")
        except Exception as e:
            print(f"Error during transcription with large-v3: {e}")
            print("Falling back to base model...")
            try:
                result = transcribe_with_enhanced_model(temp_enhanced_file, "base")
            except Exception as e2:
                print(f"Error with base model: {e2}")
                print("Please check your system memory and try again.")
                return
        
        # Step 7: Identify speakers
        speaker_segments = identify_speakers(segments, result)
        
        print("Transcription completed!")
        
        # Step 8: Generate enhanced output
        output_file = "enhanced_transcript.txt"
        print(f"Saving enhanced transcript to: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("ENHANCED SPEECH-TO-TEXT TRANSCRIPT\n")
            f.write("With Audience Detection & Audio Enhancement\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Audio File: {audio_file}\n")
            f.write(f"Model: OpenAI Whisper Large-v3 (Enhanced)\n")
            f.write(f"Repository: https://github.com/openai/whisper\n")
            f.write(f"Cost: Free (Open Source)\n")
            f.write(f"Audio Processing: Denoising, Filtering, Speaker Detection\n\n")
            f.write("TRANSCRIPT:\n")
            f.write("-" * 20 + "\n\n")
            
            # Write all segments in chronological order without speaker labels
            for segment in result["segments"]:
                start_time = format_timestamp(segment["start"])
                text = segment["text"].strip()
                
                if text:
                    f.write(f"{start_time} {text}\n")
        
        # Clean up temporary files
        for temp_file in [temp_audio_file, temp_enhanced_file]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        print("Enhanced transcript saved successfully!")
        
        print("\n" + "=" * 60)
        print("ENHANCED CONVERSION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Enhanced transcript saved to: {output_file}")
        print(f"Total segments: {len(result['segments'])}")
        print(f"Speaker segments: {len(speaker_segments)}")
        print(f"Audience segments detected: {len(audience_segments)}")
        print(f"Audio processing: Denoising, filtering, speaker detection applied")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the audio file exists")
        print("2. Check that the audio file is not corrupted")
        print("3. Try: pip install -r requirements.txt")
        print("4. Ensure you have sufficient memory for the large model")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
