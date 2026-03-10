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

from medical_transcription.agent import MedicalAgent
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
        help="Path to audio file (WAV, MP3, M4A, OGG, FLAC)"
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
    """Transcribe German medical dictation and extract structured summary using agentic AI with Ollama."""

    # Handle version flag
    if show_version:
        console.print(f"Medical Transcription v{__version__}")
        return 0

    setup_logging(verbose)

    try:
        # Initialize agent
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"🤖 Medical agent processing {audio_file.name}...",
                total=None
            )

            # Create agent (always uses Ollama)
            agent = MedicalAgent()

            if verbose:
                provider_info = agent.get_provider_info()
                console.print(
                    f"\n[cyan]Provider:[/cyan] {provider_info['provider']} "
                    f"({provider_info['model']})"
                )

            # Process audio end-to-end
            summary = agent.process(str(audio_file))

            progress.update(task, completed=True)

        # Prepare output
        output_data = {
            "audio_file": str(audio_file),
            "provider": agent.get_provider_info()["provider"],
            "model": agent.get_provider_info()["model"],
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
