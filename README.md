# Audio to Text Transcription Project

## Project Overview

This project demonstrates speech-to-text transcription capabilities using open-source models. The audio file has been trimmed from 00:13:50 to 00:18:00 (4 minutes 10 seconds) and converted from M4A to WAV format for optimal processing.

## Model and Technology Used

### OpenAI Whisper (Open Source)
- **Model**: Whisper Base
- **Repository**: https://github.com/openai/whisper
- **Cost**: **FREE** (Open Source)
- **License**: MIT License
- **Paper**: "Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al., 2022)

### Technical Specifications
- **Framework**: PyTorch
- **Audio Format**: WAV (converted from M4A)
- **Duration**: 4 minutes 10 seconds (trimmed segment)
- **Timestamp Format**: [MM:SS] as per requirements

## Features

-  Converts WAV audio files to text with high accuracy
-  Includes precise timestamps in [MM:SS] format
-  Uses OpenAI Whisper Base model (free, open-source)
-  Outputs structured transcript following required format
-  Automated environment setup and dependency management

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
2. Update the `audio_file` variable in `audio_to_text.py` with your filename
3. Run the transcription:
   ```bash
   python audio_to_text.py
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
├── audio_to_text.py          # Main transcription script
├── setup_environment.py      # Environment setup automation
├── requirements.txt          # Python dependencies
├── README.md                # This documentation
├── GMT20250919-031940_Recording.wav  # Input audio file
└── transcript.txt           # Output transcript
```

## Results

The transcription successfully processed the 4-minute 10-second audio segment, producing a well-structured transcript with accurate timestamps. The output demonstrates the model's capability to handle conversational speech with proper punctuation and sentence structure.

## Compliance with Guidelines

 **Open Source Model**: OpenAI Whisper with proper GitHub repository citation  
 **Cost Transparency**: Clearly stated as FREE (open source)  
 **Audio Length**: Limited to 4 minutes 10 seconds (within reasonable processing limits)  
 **Output Format**: Exact [Timestamp] Sentence format as required  
 **Accuracy Focus**: High-quality transcription with proper structure  

## References

1. Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). Robust speech recognition via large-scale weak supervision. arXiv preprint arXiv:2212.04356.
2. OpenAI Whisper GitHub Repository: https://github.com/openai/whisper
3. Whisper Model Card: https://github.com/openai/whisper#available-models-and-languages
