import os
import sys

def format_timestamp(seconds):
    """Format timestamp as [MM:SS]"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"[{minutes:02d}:{seconds:02d}]"

def main():
    """Main function"""
    print("=" * 60)
    print("AUDIO TO TEXT CONVERTER")
    print("=" * 60)
    
    audio_file = "GMT20250919-031940_Recording.wav"
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        print("Please make sure your WAV file is in this folder")
        input("Press Enter to exit...")
        return
    
    try:
        print(f"Audio file: {audio_file}")
        print("Loading Whisper model...")
        
        import whisper
        model = whisper.load_model("base")
        
        print("Starting transcription...")
        
        result = model.transcribe(
            audio_file,
            word_timestamps=True,
            verbose=False
        )
        
        print("Transcription completed!")
        
        output_file = "transcript.txt"
        print(f"Saving transcript to: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("SPEECH-TO-TEXT TRANSCRIPT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: OpenAI Whisper Base\n")
            f.write(f"Repository: https://github.com/openai/whisper\n")
            f.write(f"Cost: Free (Open Source)\n\n")
            f.write("TRANSCRIPT:\n")
            f.write("-" * 20 + "\n\n")
            
            for segment in result["segments"]:
                start_time = format_timestamp(segment["start"])
                text = segment["text"].strip()
                
                if text:
                    f.write(f"{start_time} {text}\n")
        
        print("Transcript saved successfully!")
        
        print("\n" + "=" * 60)
        print("CONVERSION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Transcript saved to: {output_file}")
        print(f"Total segments: {len(result['segments'])}")
        
    except ImportError:
        print("Whisper not installed!")
        print("Installing OpenAI Whisper...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper"])
        print("Whisper installed! Please run the script again.")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the audio file exists")
        print("2. Check that the audio file is not corrupted")
        print("3. Try: pip install --upgrade openai-whisper")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
