#!/usr/bin/env python3
"""Generate sample German medical dictation audio using gTTS."""

from gtts import gTTS
from pathlib import Path
import sys

def main():
    """Create sample audio file."""
    # German medical dictation text
    text = """
    Patientin kommt mit starken Kopfschmerzen und Übelkeit seit drei Tagen.
    Temperatur ist 38 Komma 2 Grad Celsius.
    Bei der Untersuchung zeigt die Patientin Lichtempfindlichkeit und Nackenstarre.
    Diagnose: Verdacht auf Migräne mit Aura.
    Therapie: Verschreibe Ibuprofen 600 Milligramm bei Bedarf, maximal dreimal täglich.
    Nachkontrolle in einer Woche oder früher bei Verschlechterung.
    """

    # Output path
    output_path = Path(__file__).parent.parent / "examples" / "sample_medical_de.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Generating German medical dictation audio...")
    print(f"Text length: {len(text.strip())} characters")

    try:
        # Generate speech
        tts = gTTS(text=text.strip(), lang='de', slow=False)

        # Save as MP3 first (gTTS default)
        mp3_path = output_path.with_suffix('.mp3')
        tts.save(str(mp3_path))

        print(f"✓ Audio generated: {mp3_path}")
        print(f"File size: {mp3_path.stat().st_size / 1024:.1f} KB")

        # Convert to WAV using pydub
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(str(mp3_path))
            audio.export(str(output_path), format="wav")
            mp3_path.unlink()  # Remove MP3 file
            print(f"✓ Converted to WAV: {output_path}")
            print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
        except ImportError:
            print("⚠ pydub not available, keeping MP3 format")
            mp3_path.rename(output_path.with_suffix('.mp3'))
        except FileNotFoundError as e:
            print(f"⚠ ffmpeg not found, keeping MP3 format")
            print(f"  Install ffmpeg with: sudo apt-get install ffmpeg")
            final_path = output_path.with_suffix('.mp3')
            if mp3_path != final_path:
                mp3_path.rename(final_path)
            print(f"✓ MP3 file available: {final_path}")
            print(f"  The application can use MP3 files directly.")

        print("\n✓ Sample audio created successfully!")
        return 0

    except Exception as e:
        print(f"✗ Error creating audio: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
