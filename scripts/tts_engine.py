import pyttsx3
import sys
import os
import argparse

def generate_audio(text, output_file, speed=150):
    engine = pyttsx3.init()
    
    # 1. Set Speed (Rate)
    # Standard is usually around 200. "Slightly fast" might be 220, but let's stick to safe defaults and let args control it.
    engine.setProperty('rate', speed)

    # 2. Set Voice (Male)
    voices = engine.getProperty('voices')
    # Try to find a male voice
    male_voice_id = None
    for voice in voices:
        # Heuristic: "David" (Windows), "male" in ID
        if "david" in voice.name.lower() or "male" in voice.name.lower():
            male_voice_id = voice.id
            break
    
    # If no specific male voice found, default usually is male on many systems, or just use 0
    if male_voice_id:
        engine.setProperty('voice', male_voice_id)
    
    # 3. Save to file
    # pyttsx3 save_to_file is async-ish in nature, handled by runAndWait
    engine.save_to_file(text, output_file)
    engine.runAndWait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Text to speak")
    parser.add_argument("output_path", help="Path to save mp3/wav")
    parser.add_argument("--speed", type=int, default=175, help="Speech rate (words per minute)")
    
    args = parser.parse_args()
    
    try:
        generate_audio(args.text, args.output_path, args.speed)
        print(f"Success: {args.output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

