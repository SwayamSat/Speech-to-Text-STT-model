import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import librosa
    import noisereduce as nr
    from scipy import signal
    from scipy.signal import butter, filtfilt
    import soundfile as sf
    import whisper
except ImportError as e:
    print(f"Missing library: {e}")
    print("Install: pip install -r requirements.txt")
    sys.exit(1)

def format_timestamp(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"[{minutes:02d}:{seconds:02d}]"

def calculate_speech_flow_similarity(segment1, segment2):
    """Calculate similarity between two speech segments for flow detection"""
    text1 = segment1["text"].strip().lower()
    text2 = segment2["text"].strip().lower()
    
    if not text1 or not text2:
        return 0.0
    
    continuation_patterns = [
        ("so", "you", "know"),
        ("that", "you", "know"),
        ("i", "mean"),
        ("like", "i", "said"),
        ("as", "i", "was", "saying"),
        ("and", "then"),
        ("but", "also"),
        ("however",),
        ("therefore",),
        ("moreover",),
    ]
    
    for pattern in continuation_patterns:
        if all(word in text1 for word in pattern) or all(word in text2 for word in pattern):
            return 0.8
    
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def is_speech_continuation(segment1, segment2, silence_gap=2.0):
    """Determine if two segments are part of the same speech flow"""
    if silence_gap <= 0.5:
        return True
    
    if silence_gap > 5.0:
        return False
    
    text1 = segment1["text"].strip()
    incomplete_endings = ["...", "so", "and", "but", "or", "that", "which", "who", "where", "when", "why", "how"]
    
    if any(text1.lower().endswith(ending) for ending in incomplete_endings):
        return True
    
    text2 = segment2["text"].strip()
    sentence_starters = ["so", "and", "but", "or", "that", "which", "who", "where", "when", "why", "how", "it", "this", "that", "there"]
    
    if any(text2.lower().startswith(starter) for starter in sentence_starters):
        return True
    
    similarity = calculate_speech_flow_similarity(segment1, segment2)
    return similarity > 0.3 and silence_gap <= 3.0

def is_repetitive_text(text):
    """Check if text is repetitive (like 'I don't know' repeated)"""
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    repetitive_patterns = [
        "i don't know",
        "hello hello",
        "yes yes",
        "no no",
        "ok ok",
        "uh uh",
        "um um"
    ]
    
    for pattern in repetitive_patterns:
        if pattern in text_lower and text_lower.count(pattern) > 2:
            return True
    
    words = text_lower.split()
    if len(words) > 3:
        unique_words = len(set(words))
        if unique_words < len(words) * 0.3:
            return True
    
    return False

def clean_repetitive_text(text):
    """Clean up repetitive text patterns"""
    if not text:
        return text
    
    import re
    
    repetitive_patterns = [
        (r'\b(i don\'t know\s*){3,}', 'I don\'t know'),
        (r'\b(hello\s*){3,}', 'Hello'),
        (r'\b(yes\s*){3,}', 'Yes'),
        (r'\b(no\s*){3,}', 'No'),
        (r'\b(ok\s*){3,}', 'OK'),
        (r'\b(uh\s*){3,}', 'Uh'),
        (r'\b(um\s*){3,}', 'Um'),
    ]
    
    cleaned_text = text
    for pattern, replacement in repetitive_patterns:
        cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
    
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def intelligent_chunking(segments, max_chunk_duration=30):
    """Group segments that are part of the same speech flow"""
    if not segments:
        return segments
    
    chunks = []
    current_chunk = [segments[0]]
    
    for i in range(1, len(segments)):
        current_segment = segments[i]
        previous_segment = segments[i-1]
        
        silence_gap = current_segment["start"] - previous_segment["end"]
        chunk_duration = current_segment["end"] - current_chunk[0]["start"]
        is_repetitive = is_repetitive_text(current_segment["text"])
        is_continuation = is_speech_continuation(previous_segment, current_segment, silence_gap)
        
        if is_repetitive and not is_repetitive_text(previous_segment["text"]):
            chunks.append(merge_segments(current_chunk))
            current_chunk = [current_segment]
            continue
        
        if (silence_gap <= 2.0 and 
            chunk_duration <= max_chunk_duration and
            is_continuation and
            not (is_repetitive and not is_repetitive_text(previous_segment["text"]))):
            current_chunk.append(current_segment)
        else:
            chunks.append(merge_segments(current_chunk))
            current_chunk = [current_segment]
    
    if current_chunk:
        chunks.append(merge_segments(current_chunk))
    
    return chunks

def merge_segments(segments):
    """Merge multiple segments into one chunk"""
    if not segments:
        return None
    
    if len(segments) == 1:
        return segments[0]
    
    combined_text = " ".join(segment["text"].strip() for segment in segments if segment["text"].strip())
    cleaned_text = clean_repetitive_text(combined_text)
    
    return {
        "start": segments[0]["start"],
        "end": segments[-1]["end"],
        "text": cleaned_text
    }

def get_audio_file():
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
    print("Loading and preprocessing audio...")
    audio, sr = librosa.load(audio_file, sr=target_sr, mono=True)
    print(f"Original audio: {len(audio)/sr:.2f} seconds, {sr} Hz")
    
    print("Applying noise reduction...")
    audio_denoised = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.8)
    
    print("Applying high-pass filter...")
    nyquist = sr * 0.5
    high_cutoff = 80
    high = high_cutoff / nyquist
    b, a = butter(4, high, btype='high')
    audio_filtered = filtfilt(b, a, audio_denoised)
    
    audio_normalized = librosa.util.normalize(audio_filtered)
    
    print("Applying dynamic range compression...")
    audio_compressed = librosa.effects.preemphasis(audio_normalized)
    
    print("Audio preprocessing completed!")
    return audio_compressed, sr

def detect_speaker_changes(audio, sr, segment_length=3.0):
    print("Analyzing audio for speaker changes...")
    
    hop_length = int(sr * 0.1)
    frame_length = int(sr * 0.5)
    
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    features = np.vstack([energy, spectral_centroid, zcr])
    features_norm = (features - features.mean(axis=1, keepdims=True)) / (features.std(axis=1, keepdims=True) + 1e-8)
    
    change_points = []
    window_size = int(segment_length * sr / hop_length)
    
    for i in range(window_size, len(features_norm[0]) - window_size):
        before = features_norm[:, i-window_size:i].mean(axis=1)
        after = features_norm[:, i:i+window_size].mean(axis=1)
        
        distance = np.linalg.norm(after - before)
        if distance > 0.5:
            change_points.append(i * hop_length / sr)
    
    print(f"Detected {len(change_points)} potential speaker changes")
    return change_points

def segment_audio_by_speakers(audio, sr, change_points):
    print("Segmenting audio by speakers...")
    
    segments = []
    start_time = 0
    
    for change_point in change_points:
        if change_point - start_time > 1.0:
            segments.append((start_time, change_point))
            start_time = change_point
    
    if len(audio) / sr - start_time > 1.0:
        segments.append((start_time, len(audio) / sr))
    
    print(f"Created {len(segments)} audio segments")
    return segments

def detect_audience_participation(audio, sr):
    print("Detecting audience participation...")
    
    hop_length = int(sr * 0.05)
    frame_length = int(sr * 0.2)
    
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    energy_threshold = np.percentile(energy, 20)
    voice_activity = energy > energy_threshold
    
    energy_diff = np.diff(energy)
    sudden_changes = np.abs(energy_diff) > np.std(energy_diff) * 2
    
    audience_segments = []
    in_audience_segment = False
    segment_start = 0
    
    for i, (is_voice, is_sudden) in enumerate(zip(voice_activity, np.concatenate([[False], sudden_changes]))):
        time_point = i * hop_length / sr
        
        if is_voice and not in_audience_segment:
            in_audience_segment = True
            segment_start = time_point
        elif not is_voice and in_audience_segment:
            if time_point - segment_start > 0.5:
                audience_segments.append((segment_start, time_point))
            in_audience_segment = False
        elif is_sudden and in_audience_segment:
            audience_segments.append((segment_start, time_point))
            segment_start = time_point
    
    if in_audience_segment and len(audio) / sr - segment_start > 0.5:
        audience_segments.append((segment_start, len(audio) / sr))
    
    print(f"Detected {len(audience_segments)} audience participation segments")
    return audience_segments

def enhance_audience_audio(audio, sr, audience_segments):
    print("Enhancing audience audio segments...")
    
    enhanced_audio = audio.copy()
    
    for start_time, end_time in audience_segments:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        audience_segment = audio[start_sample:end_sample]
        audience_enhanced = librosa.effects.preemphasis(audience_segment, coef=0.97)
        audience_compressed = np.tanh(audience_enhanced * 2) * 0.8
        audience_normalized = librosa.util.normalize(audience_compressed)
        
        enhanced_audio[start_sample:end_sample] = audience_normalized
    
    return enhanced_audio

def transcribe_with_enhanced_model(audio_file, model_size="large-v3"):
    print(f"Loading enhanced Whisper model ({model_size})...")
    
    model = whisper.load_model(model_size)
    
    print("Starting enhanced transcription...")
    
    result = model.transcribe(
        audio_file,
        word_timestamps=True,
        verbose=False,
        language="en",
        temperature=0.0,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
        fp16=False
    )
    
    return result

def identify_speakers(segments, result):
    print("Identifying speakers...")
    
    speaker_segments = []
    speaker_count = 1
    
    for i, segment in enumerate(segments):
        start_time, end_time = segment
        
        segment_texts = []
        for text_segment in result["segments"]:
            if start_time <= text_segment["start"] < end_time:
                segment_texts.append(text_segment)
        
        if segment_texts:
            segment_duration = end_time - start_time
            if segment_duration > 10:
                speaker = "Main Speaker"
            elif segment_duration > 3:
                speaker = f"Speaker {speaker_count}"
                speaker_count += 1
            else:
                speaker = "Audience"
            
            speaker_segments.append({
                'speaker': speaker,
                'start': start_time,
                'end': end_time,
                'text_segments': segment_texts
            })
    
    return speaker_segments

def classify_speaker_type(segment, duration, energy_level, spectral_features):
    if duration > 15:
        return "Main Speaker"
    elif duration > 5:
        if energy_level > 0.1:
            return "Main Speaker"
        else:
            return "Audience Member"
    else:
        if energy_level > 0.05:
            return "Audience Question"
        else:
            return "Audience Reaction"

def detect_unclear_audio(segments):
    """Detect if audio is unclear based on repetitive patterns"""
    if not segments:
        return False
    
    repetitive_count = 0
    total_segments = len(segments)
    
    for segment in segments:
        if is_repetitive_text(segment["text"]):
            repetitive_count += 1
    
    return repetitive_count / total_segments > 0.5

def process_entire_file(audio_file):
    """Process the entire audio file with intelligent chunking"""
    print(f"\nProcessing entire audio file: {audio_file}")
    
    audio_processed, sr = preprocess_audio(audio_file)
    
    temp_audio_file = "temp_processed_audio.wav"
    try:
        sf.write(temp_audio_file, audio_processed, sr)
    except Exception as e:
        print(f"Error saving processed audio: {e}")
        return
    
    change_points = detect_speaker_changes(audio_processed, sr)
    segments = segment_audio_by_speakers(audio_processed, sr, change_points)
    audience_segments = detect_audience_participation(audio_processed, sr)
    audio_enhanced = enhance_audience_audio(audio_processed, sr, audience_segments)
    
    temp_enhanced_file = "temp_enhanced_audio.wav"
    try:
        sf.write(temp_enhanced_file, audio_enhanced, sr)
    except Exception as e:
        print(f"Error saving enhanced audio: {e}")
        return
    
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
    
    # Check if audio is unclear
    is_unclear = detect_unclear_audio(result["segments"])
    
    if is_unclear:
        print("⚠️  Warning: Audio appears to be unclear or noisy")
        print("   The transcript may contain repetitive phrases")
        print("   Consider using a clearer audio file for better results")
    
    # Apply intelligent chunking for better flow
    print("Applying intelligent chunking for natural speech flow...")
    intelligent_chunks = intelligent_chunking(result["segments"])
    
    speaker_segments = identify_speakers(segments, result)
    
    print("Transcription completed!")
    
    output_file = "enhanced_transcript.txt"
    print(f"Saving enhanced transcript to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("ENHANCED SPEECH-TO-TEXT TRANSCRIPT\n")
        f.write("With Intelligent Chunking & Audio Enhancement\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Audio File: {os.path.basename(audio_file)}\n")
        f.write(f"Model: OpenAI Whisper Large-v3 (Enhanced)\n")
        f.write(f"Repository: https://github.com/openai/whisper\n")
        f.write(f"Cost: Free (Open Source)\n")
        f.write(f"Audio Processing: Denoising, Filtering, Intelligent Chunking\n")
        f.write(f"Original Segments: {len(result['segments'])}\n")
        f.write(f"Intelligent Chunks: {len(intelligent_chunks)}\n")
        f.write(f"Compression Ratio: {len(result['segments'])/len(intelligent_chunks):.2f}x\n")
        if is_unclear:
            f.write(f"⚠️  Audio Quality: Unclear/Noisy (repetitive patterns detected)\n")
        f.write("\n")
        f.write("TRANSCRIPT:\n")
        f.write("-" * 20 + "\n\n")
        
        for chunk in intelligent_chunks:
            start_time = format_timestamp(chunk["start"])
            text = chunk["text"].strip()
            
            if text:
                f.write(f"{start_time} {text}\n")
    
    for temp_file in [temp_audio_file, temp_enhanced_file]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print("Enhanced transcript saved successfully!")
    print(f"Original segments: {len(result['segments'])}")
    print(f"Intelligent chunks: {len(intelligent_chunks)}")
    print(f"Compression ratio: {len(result['segments'])/len(intelligent_chunks):.2f}x")
    
    if is_unclear:
        print("\n⚠️  Note: Audio quality appears unclear. Consider using a clearer audio file.")


def main():
    print("=" * 60)
    print("ENHANCED SPEECH-TO-TEXT CONVERTER")
    print("With Intelligent Chunking & Audio Enhancement")
    print("=" * 60)
    
    audio_file = get_audio_file()
    if not audio_file:
        return
    
    try:
        print(f"\nProcessing entire audio file: {audio_file}")
        process_entire_file(audio_file)
        
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
