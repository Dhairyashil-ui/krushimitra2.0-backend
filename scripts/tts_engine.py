import asyncio
import sys
import os
import argparse
import edge_tts

# Voice Mapping
VOICE_MAP = {
    "hi": "hi-IN-MadhurNeural",
    "hi-IN": "hi-IN-MadhurNeural",
    "en": "en-US-ChristopherNeural", 
    "en-US": "en-US-ChristopherNeural",
    "mr": "mr-IN-AarohiNeural", # Marathi
    "gu": "gu-IN-DhwaniNeural", # Gujarati
    "ta": "ta-IN-ValluvarNeural", # Tamil
    "te": "te-IN-MohanNeural",    # Telugu
    "kn": "kn-IN-GaganNeural",    # Kannada
    "ml": "ml-IN-MidhunNeural"    # Malayalam
}

async def generate_audio(text, output_file, speed_wpm=175, lang="en"):
    # 1. Determine Voice
    voice = VOICE_MAP.get(lang.split("-")[0], "en-US-ChristopherNeural")
    if lang in VOICE_MAP:
        voice = VOICE_MAP[lang]
        
    # 2. Convert WPM to Rate Percentage
    # Base speed ~150-160 WPM. 
    # Formula approximation: (Target - 150) / 150 * 100 ? No that's too much drop.
    # edge-tts default is quite fast already compared to pyttsx3 default.
    # Let's try a simple mapping:
    # 175 wpm -> +10%
    # 150 wpm -> +0%
    # 200 wpm -> +25%
    
    # Let's just use string passing for now, or keep it 0% if unsure.
    # User said "base speed ~175". If that's what they were passing to pyttsx3...
    # Let's map 175 -> +0% because users might have tuned 175 to mean "normal" for them.
    # If they pass > 175, we speed up.
    
    # Actually, edge-tts is generally good at default rate.
    # Let's map it:
    # deviation = (speed_wpm - 160) // 2 
    # 175 -> +7%
    # 150 -> -5%
    # 200 -> +20%
    
    rate_val = int((speed_wpm - 160) * 0.8) # Tuning factor
    sign = "+" if rate_val >= 0 else ""
    rate_str = f"{sign}{rate_val}%"

    print(f"DEBUG: Using voice {voice}, rate {rate_str} for lang {lang}", file=sys.stderr)

    communicate = edge_tts.Communicate(text, voice, rate=rate_str)
    await communicate.save(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Text to speak")
    parser.add_argument("output_path", help="Path to save mp3")
    parser.add_argument("--speed", type=int, default=175, help="Target WPM (approx)")
    parser.add_argument("--lang", type=str, default="en", help="Language code (e.g. en, hi)")
    
    args = parser.parse_args()
    
    try:
        dirname = os.path.dirname(args.output_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
            
        asyncio.run(generate_audio(args.text, args.output_path, args.speed, args.lang))
        # No output to stdout to keep it clean, unless debugging
        # But tts.js doesn't read stdout for success, it just checks file existence/exit code.
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
