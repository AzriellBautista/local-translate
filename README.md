# local-translate

Local text translation from the command line, powered by [Ollama](https://ollama.com).

## Install

```bash
uv tool install git+https://github.com/AzriellBautista/local-translate
```

Requires a running Ollama instance with a translation model available
(default: `translategemma:4b-it-q4_K_M`).

## Usage

```bash
local-translate TEXT --from FROM --to TO [--stream] [--show-langs]
```

- `TEXT` — text to translate. If omitted, text is read from STDIN.
- `--from` / `-F` — source language code.
- `--to` / `-T` — target language code.
- `--stream` — stream the translation.
- `--show-langs` — list supported language codes.

## Examples

```bash
local-translate "hello" --from en --to es
```

Pipe text in:

```bash
python -c "import this" | local-translate --from en --to es
```

Stream the output:

```bash
local-translate "hello" --from en --to es --stream
```

List supported languages:

```bash
local-translate --show-langs
```