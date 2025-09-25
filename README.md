# Enhanced Speech-to-Text Converter

## Project Overview

This project provides a comprehensive speech-to-text transcription solution using OpenAI's Whisper model with intelligent chunking and advanced audio processing. The system automatically groups sentences spoken in one flow for natural conversation flow and handles unclear audio with smart repetitive text detection and cleaning.

## Model and Technology Used

### OpenAI Whisper (Open Source)
- **Model**: Whisper Large-v3 (with Base fallback)
- **Repository**: https://github.com/openai/whisper
- **Cost**: **FREE** (Open Source)
- **License**: MIT License
- **Paper**: "Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al., 2022)

### Technical Specifications
- **Framework**: PyTorch
- **Audio Format**: WAV, MP3, M4A, FLAC
- **Timestamp Format**: [MM:SS] format
- **Model Size**: Large-v3 (high accuracy) with Base fallback

## Features

- **Single Script Solution**: All functionality in one easy-to-use script
- **Intelligent Chunking**: Groups sentences spoken in one flow for natural conversation flow
- **Repetitive Text Cleaning**: Automatically detects and cleans unclear audio patterns
- **Audio Quality Detection**: Warns users about unclear audio and suggests improvements
- **High-Quality Transcription**: Converts audio files to text with high accuracy using OpenAI Whisper
- **Precise Timestamps**: Includes timestamps in [MM:SS] format for each speech segment
- **Audio Denoising**: Advanced noise reduction using multiple algorithms
- **Speaker Diarization**: Automatic detection and separation of different speakers
- **Audience Detection**: Specifically identifies audience participation and questions
- **Audio Enhancement**: Preprocessing with filtering, normalization, and compression
- **Enhanced Model**: Uses Whisper Large-v3 with automatic fallback to base model
- **User-Friendly Interface**: Interactive file selection and progress indicators
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Multiple Audio Formats**: Supports WAV, MP3, M4A, and FLAC files

## Setup Instructions

### Quick Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the script:
   ```bash
   python SpeechToText.py
   ```

### Optional: Virtual Environment

1. Create virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate virtual environment:
   - **Windows:** `venv\Scripts\activate`
   - **macOS/Linux:** `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your audio file in the project directory (supports WAV, MP3, M4A, FLAC)
2. Run the script:
   ```bash
   python SpeechToText.py
   ```
3. Select your audio file from the interactive menu
4. The script will automatically process the entire audio file with intelligent chunking and generate `enhanced_transcript.txt`

### Intelligent Chunking Example
The enhanced transcript uses intelligent chunking to group related speech segments:

**Before Intelligent Chunking:**
```
[01:17] So you know I…
[01:19] It was a very ambitious kind of an ad.
[01:21] So in that way I connected with it.
[01:24] That you know it was a very good way to you know linking your brand with the other brand and
[01:27] showing your superiority.
```

**After Intelligent Chunking:**
```
[01:17] So you know I… It was a very ambitious kind of an ad. So in that way I connected with it. That you know it was a very good way to you know linking your brand with the other brand and showing your superiority.
```

### Example Output
The script will display progress information and save the transcript:
```
============================================================
ENHANCED SPEECH-TO-TEXT CONVERTER
With Intelligent Chunking & Audio Enhancement
============================================================

Available audio files in current directory:
1. GMT20250919-031940_Recording.wav
2. presentation_audio.mp3

Select audio file (1-2) or enter custom path: 1

Processing audio file: GMT20250919-031940_Recording.wav
Loading and preprocessing audio...
Applying noise reduction...
Applying high-pass filter...
Applying dynamic range compression...
Audio preprocessing completed!
Analyzing audio for speaker changes...
Detected 15 potential speaker changes
Segmenting audio by speakers...
Created 12 audio segments
Detecting audience participation...
Detected 8 audience participation segments
Enhancing audience audio segments...
Loading enhanced Whisper model (large-v3)...
Starting enhanced transcription...
Transcription completed!
Saving enhanced transcript to: enhanced_transcript.txt
Enhanced transcript saved successfully!
```


## Output Format

### Smart Chunking Example
The enhanced transcript uses smart chunking to group related speech segments:

**Before Smart Chunking:**
```
[01:17] So you know I…
[01:19] It was a very ambitious kind of an ad.
[01:21] So in that way I connected with it.
[01:24] That you know it was a very good way to you know linking your brand with the other brand and
[01:27] showing your superiority.
```

**After Smart Chunking:**
```
[01:17] So you know I… It was a very ambitious kind of an ad. So in that way I connected with it. That you know it was a very good way to you know linking your brand with the other brand and showing your superiority.
```

### Standard Format
```
[00:00] Morning. Good morning.
[00:03] Good morning sir.
[00:06] I would greatly appreciate if you call me Saurabh and not sir.
[00:10] Thank you for that clarification, Saurabh.
[00:15] Let's begin today's presentation on business storytelling.
```

## How It Works

The Speech-to-Text converter uses a sophisticated multi-stage pipeline to process audio and generate accurate transcripts:

### 1. **Audio Input & Detection**
- **File Scanning**: Automatically scans the current directory for audio files (WAV, MP3, M4A, FLAC)
- **Interactive Selection**: Presents a numbered menu for easy file selection
- **Format Support**: Handles multiple audio formats with automatic conversion

### 2. **Audio Preprocessing Pipeline**
- **Loading**: Uses librosa to load audio at 16kHz sample rate for optimal Whisper processing
- **Noise Reduction**: Applies advanced noise reduction using the `noisereduce` library
- **High-Pass Filtering**: Removes low-frequency noise (below 80Hz) using Butterworth filter
- **Normalization**: Ensures consistent audio levels across the entire recording
- **Dynamic Range Compression**: Applies pre-emphasis to enhance speech clarity

### 3. **Speaker Analysis & Detection**
- **Feature Extraction**: Analyzes energy, spectral centroid, and zero-crossing rate
- **Change Point Detection**: Identifies potential speaker changes using statistical analysis
- **Audience Detection**: Specifically looks for audience participation with lower energy thresholds
- **Segmentation**: Creates intelligent audio segments based on speaker characteristics

### 4. **Audio Enhancement**
- **Audience-Specific Processing**: Applies special enhancements to audience segments
- **Frequency Boosting**: Enhances high frequencies for better clarity
- **Compression**: Applies gentle compression to audience audio
- **Quality Optimization**: Ensures all audio segments are optimally processed

### 5. **Transcription Process**
- **Model Selection**: Uses Whisper Large-v3 for maximum accuracy with automatic fallback to base model
- **Optimized Parameters**: Configured for better audience detection and speech recognition
- **Context Awareness**: Uses previous text context for improved accuracy
- **Language Specification**: Explicitly set to English for better performance

### 6. **Output Generation**
- **Chronological Ordering**: Arranges all speech segments in time sequence
- **Timestamp Formatting**: Converts timestamps to [MM:SS] format
- **Text Cleaning**: Removes empty segments and normalizes text
- **File Output**: Saves to `enhanced_transcript.txt` with metadata

### 7. **Error Handling & Fallbacks**
- **Memory Management**: Automatically falls back to smaller model if memory insufficient
- **Dependency Checking**: Validates all required libraries before processing
- **File Validation**: Ensures audio files exist and are readable
- **Cleanup**: Removes temporary files after processing

### 8. **Performance Optimization**
- **Parallel Processing**: Uses efficient audio processing libraries
- **Memory Efficiency**: Processes audio in chunks to manage memory usage
- **Caching**: Leverages Whisper's model caching for faster subsequent runs
- **Temporary Files**: Uses temporary files to avoid memory overflow

## Technical Details

### Dependencies
- `openai-whisper>=20231117` - Core transcription engine
- `torch>=2.0.0` - PyTorch framework
- `torchaudio>=2.0.0` - Audio processing
- `librosa>=0.10.0` - Advanced audio analysis
- `noisereduce>=3.0.0` - Noise reduction algorithms
- `scipy>=1.10.0` - Scientific computing
- `numpy>=1.24.0` - Numerical computing
- `soundfile>=0.12.0` - Audio file I/O

### Model Performance
- **Whisper Base**: Balanced speed and accuracy (original)
- **Whisper Large-v3**: Superior accuracy for complex audio (enhanced)
- **Processing Time**: ~2-3 minutes for 4-minute audio (base), ~5-7 minutes (large-v3)
- **Accuracy**: High accuracy for clear speech, enhanced for audience participation
- **Language Support**: Multi-language (English in this case)
- **Audio Processing**: Advanced denoising, filtering, and speaker detection

## Cost Analysis

- **Model**: OpenAI Whisper (Open Source) - **$0.00**
- **Infrastructure**: Local processing - **$0.00**
- **Total Cost**: **FREE**

## File Structure

```
├── SpeechToText.py                  # Single comprehensive script
├── requirements.txt                  # Python dependencies
├── README.md                        # This documentation
├── GMT20250919-031940_Recording.wav # Input audio file (WAV format)
├── enhanced_transcript.txt          # Enhanced transcript output
└── venv/                           # Virtual environment (optional)
    ├── Scripts/                    # Windows activation scripts
    ├── Lib/                        # Python packages
    └── pyvenv.cfg                  # Virtual environment config
```

## Results

The transcription successfully processed the audio file, producing a well-structured transcript with accurate timestamps. The output demonstrates the model's capability to handle conversational speech, including:

- **Professional Presentation**: Clear transcription of business storytelling lecture content
- **Interactive Elements**: Captured audience participation and Q&A sessions
- **Accurate Timestamps**: Precise timing for each speech segment
- **Natural Language Processing**: Proper punctuation and sentence structure
- **Multi-speaker Recognition**: Handles different speakers and conversation flow

The transcript shows a business storytelling class session with instructor Saurabh Arora from Waterfield Advisors, demonstrating the model's effectiveness for educational and professional content.

## Key Benefits

- **Open Source Model**: OpenAI Whisper with proper GitHub repository citation
- **Cost Transparency**: Completely FREE (open source, no API costs)
- **High Accuracy**: Professional-grade transcription quality
- **User-Friendly**: Automated setup and intuitive interface
- **Flexible Input**: Supports various audio formats (WAV recommended)
- **Structured Output**: Clean, timestamped transcript format
- **Cross-Platform**: Works on Windows, macOS, and Linux  

## References

1. Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). Robust speech recognition via large-scale weak supervision. arXiv preprint arXiv:2212.04356.
2. OpenAI Whisper GitHub Repository: https://github.com/openai/whisper
3. Whisper Model Card: https://github.com/openai/whisper#available-models-and-languages
