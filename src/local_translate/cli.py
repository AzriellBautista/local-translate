"""Command-line interface for ``local-translate``."""

from __future__ import annotations

import sys
from collections.abc import Iterable

import ollama
import typer
from rich.console import Console
from rich.table import Table
from typer._click import ClickException

from .languages import (
    is_supported_language,
    list_supported_languages,
)
from .translator import (
    DEFAULT_TRANSLATION_MODEL,
    iter_translation_chunks,
    translate,
)

EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130

app = typer.Typer(
    add_completion=False,
    help="Translate text locally using Ollama.",
)


def _validate_language(code: str, option_name: str) -> str:
    normalized = code.strip()
    if not is_supported_language(normalized):
        raise typer.BadParameter(
            f"unsupported language code {code!r}; run --show-langs to list valid codes",
            param_hint=f"--{option_name}",
        )
    return normalized


def _show_languages() -> None:
    console = Console()
    table = Table(title="Supported languages", show_header=True, header_style="bold")
    table.add_column("Code", style="cyan", no_wrap=True)
    table.add_column("Language")
    for code, name in list_supported_languages():
        table.add_row(code, name)
    console.print(table)


def _configure_stdio() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _read_stdin() -> str:
    _configure_stdio()
    buffer = getattr(sys.stdin, "buffer", None)
    if buffer is not None:
        return buffer.read().decode("utf-8", errors="replace")
    return sys.stdin.read()


@app.command()
def run(
    text: str | None = typer.Argument(
        None, help="Text to translate. Read from STDIN when omitted."
    ),
    from_lang: str | None = typer.Option(
        None, "--from", "-F", help="Source language code."
    ),
    to_lang: str | None = typer.Option(
        None, "--to", "-T", help="Target language code."
    ),
    stream: bool = typer.Option(False, "--stream", help="Stream the translation."),
    show_langs: bool = typer.Option(
        False, "--show-langs", help="Show supported language codes and exit."
    ),
    model: str = typer.Option(
        DEFAULT_TRANSLATION_MODEL, "--model", help="Ollama model to use."
    ),
) -> None:
    """Translate TEXT from one language to another."""
    _configure_stdio()
    if show_langs:
        _show_languages()
        raise typer.Exit()

    if from_lang is None or to_lang is None:
        raise typer.BadParameter(
            "--from and --to are both required.",
            param_hint="--from/--to",
        )

    source_code = _validate_language(from_lang, "from")
    target_code = _validate_language(to_lang, "to")

    if text is None:
        text = _read_stdin()

    response = translate(
        text,
        source_code,
        target_code,
        model=model,
        stream=stream,
    )

    if stream:
        if isinstance(response, ollama.ChatResponse):
            raise RuntimeError("Expected a streaming response")
        for content in iter_translation_chunks(response):
            typer.echo(content, nl=False)
            sys.stdout.flush()
    else:
        if not isinstance(response, ollama.ChatResponse):
            raise RuntimeError("Expected a non-streaming response")
        typer.echo(response.message.content)


def main(argv: Iterable[str] | None = None) -> int:
    """Compatibility entry point accepting an optional argv list."""
    args = list(argv) if argv is not None else None
    _configure_stdio()
    try:
        app(args=args, standalone_mode=False)
    except typer.Exit as exc:
        return exc.exit_code if exc.exit_code is not None else EXIT_SUCCESS
    except ClickException as exc:
        exc.show()
        return exc.exit_code
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else EXIT_SUCCESS
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_SUCCESS


def cli(argv: Iterable[str] | None = None) -> None:
    """Console-script entry point."""
    sys.exit(main(argv))



__all__ = [
    "app",
    "cli",
    "main",
    "run",
]
