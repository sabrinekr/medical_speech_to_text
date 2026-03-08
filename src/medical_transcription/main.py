"""CLI interface for medical transcription."""

import sys
import json
import logging
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.json import JSON

from medical_transcription.core.audio_processor import AudioProcessor
from medical_transcription.core.transcriber import Transcriber
from medical_transcription.core.llm_extractor import LLMExtractor
from medical_transcription import __version__

console = Console()


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def app(
    audio_file: Path = typer.Argument(
        ...,
        exists=True,
        help="Path to audio file (WAV or MP3)"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSON file path (default: stdout)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output with progress details"
    ),
    show_version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version information"
    ),
):
    """Transcribe German medical dictation and extract structured summary."""

    # Handle version flag
    if show_version:
        console.print(f"Medical Transcription v{__version__}")
        return 0

    setup_logging(verbose)

    try:
        # Step 1: Process audio
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task1 = progress.add_task("Processing audio file...", total=None)

            processor = AudioProcessor()
            wav_path, is_temp = processor.convert_to_wav(audio_file)
            duration = processor.get_audio_duration(wav_path)

            progress.update(task1, completed=True)

            if verbose:
                console.print(f"[green]✓[/green] Audio processed: {duration:.1f}s duration")

            # Step 2: Transcribe
            task2 = progress.add_task("Transcribing audio (this may take a few minutes)...", total=None)

            transcriber = Transcriber()
            transcript_result = transcriber.transcribe(wav_path, language="de")
            transcript = transcript_result["transcript"]

            progress.update(task2, completed=True)

            if verbose:
                console.print(f"[green]✓[/green] Transcription complete: {len(transcript)} characters")
                console.print("\n[bold]Transcript:[/bold]")
                console.print(Panel(transcript, border_style="blue"))

            # Step 3: Extract structured summary
            task3 = progress.add_task("Extracting structured clinical summary...", total=None)

            extractor = LLMExtractor()
            summary = extractor.extract(transcript)

            progress.update(task3, completed=True)

            if verbose:
                console.print("[green]✓[/green] Clinical summary extracted")

            # Clean up temporary file
            if is_temp:
                wav_path.unlink()

        # Prepare output
        output_data = {
            "audio_file": str(audio_file),
            "duration_seconds": duration,
            "transcript": transcript,
            "clinical_summary": summary.model_dump()
        }

        # Write output
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            console.print(f"\n[green]✓[/green] Results saved to: {output}")
        else:
            # Print to stdout
            if not verbose:
                # If not verbose, just print JSON
                print(json.dumps(output_data, ensure_ascii=False, indent=2))
            else:
                # If verbose, show formatted output
                console.print("\n[bold]Clinical Summary:[/bold]")
                console.print(JSON(json.dumps(summary.model_dump(), ensure_ascii=False)))

        console.print("\n[bold green]✓ Processing complete![/bold green]")
        return 0

    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}", err=True)
        return 1
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}", err=True)
        return 1
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {e}", err=True)
        if verbose:
            import traceback
            console.print(traceback.format_exc(), err=True)
        return 1


if __name__ == "__main__":
    sys.exit(typer.run(app))
