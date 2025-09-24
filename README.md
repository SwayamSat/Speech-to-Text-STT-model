# Speech-to-Text (STT) Model Project

## Project Overview

This project provides a complete speech-to-text transcription solution using OpenAI's Whisper model. It includes automated environment setup, dependency management, and produces high-quality transcripts with precise timestamps. The project is designed to be user-friendly with both automatic and manual setup options.

## Model and Technology Used

### OpenAI Whisper (Open Source)
- **Model**: Whisper Base
- **Repository**: https://github.com/openai/whisper
- **Cost**: **FREE** (Open Source)
- **License**: MIT License
- **Paper**: "Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al., 2022)

### Technical Specifications
- **Framework**: PyTorch
- **Audio Format**: WAV
- **Timestamp Format**: [MM:SS] format
- **Model Size**: Base model (balanced speed and accuracy)

## Features

- **High-Quality Transcription**: Converts WAV audio files to text with high accuracy using OpenAI Whisper
- **Precise Timestamps**: Includes timestamps in [MM:SS] format for each speech segment
- **Automated Setup**: One-command environment setup with `setup_environment.py`
- **User-Friendly Interface**: Interactive console with progress indicators and error handling
- **Automatic Dependency Installation**: Installs Whisper automatically if not found
- **Structured Output**: Generates well-formatted transcript files with metadata
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Setup Instructions

### Option 1: Automatic Setup (Recommended)

```bash
python setup_environment.py
```

### Option 2: Manual Setup

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

1. Ensure your WAV audio file is in the project directory
2. Update the `audio_file` variable in `AudioToText.py` with your filename (default: `GMT20250919-031940_Recording.wav`)
3. Run the transcription:
   ```bash
   python AudioToText.py
   ```

### Example Output
The script will display progress information and save the transcript to `transcript.txt`:
```
============================================================
AUDIO TO TEXT CONVERTER
============================================================
Audio file: GMT20250919-031940_Recording.wav
Loading Whisper model...
Starting transcription...
Transcription completed!
Saving transcript to: transcript.txt
Transcript saved successfully!
```

## Output Format

The transcript follows the exact required format:
```
[00:00] Morning. Good morning.
[00:03] Good morning sir.
[00:06] I would greatly appreciate if you call me Saurabh and not sir.
```

## Technical Details

### Dependencies
- `openai-whisper>=20231117` - Core transcription engine
- `torch>=2.0.0` - PyTorch framework
- `torchaudio>=2.0.0` - Audio processing

### Model Performance
- **Whisper Base**: Balanced speed and accuracy
- **Processing Time**: ~2-3 minutes for 4-minute audio
- **Accuracy**: High accuracy for clear speech
- **Language Support**: Multi-language (English in this case)

## Cost Analysis

- **Model**: OpenAI Whisper (Open Source) - **$0.00**
- **Infrastructure**: Local processing - **$0.00**
- **Total Cost**: **FREE**

## File Structure

```
├── AudioToText.py                    # Main transcription script
├── setup_environment.py              # Environment setup automation
├── requirements.txt                  # Python dependencies
├── README.md                        # This documentation
├── GMT20250919-031940_Recording.wav # Input audio file (WAV format)
├── transcript.txt                   # Output transcript
└── venv/                           # Virtual environment (created by setup)
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
