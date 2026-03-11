# modellaucher

Interactive launcher for serving local GGUF models with `llama.cpp` and family-aware defaults.

## Features

- Recursively scans `~/models` (or a custom directory) for `.gguf` models.
- Detects model family, size, quantization, shard count, and nearby `mmproj` files.
- Reads GGUF metadata to show each model's maximum supported context.
- Presents a cleaner interactive model menu with optional color output.
- Includes tuned presets for common families, including Unsloth-style Qwen3.5 thinking/non-thinking modes.
- Builds and runs the final `llama-server` command after confirmation.

## Requirements

- macOS/Linux shell with Python 3.10+.
- `llama.cpp` with `llama-server` available in `PATH` or passed via `--llama-server`.
- Local GGUF models stored under `~/models` by default.

## Quick start

```bash
chmod +x ./modelmenu
./modelmenu
```

You can also run the Python entry directly:

```bash
python3 ./modelmenu.py
```

## CLI options

```bash
./modelmenu --models-dir ~/models
./modelmenu --llama-server ~/llama.cpp/llama-server
./modelmenu --host 0.0.0.0 --port 8000
./modelmenu --list-only
./modelmenu --dry-run
./modelmenu --color auto
./modelmenu --color always
./modelmenu --color never
```

## Serving presets

### Qwen3.5 / Qwen3

- Non-thinking (general)
- Non-thinking (reasoning)
- Thinking (general)
- Thinking (precise coding)

When supported by your `llama-server` build, the launcher sets:

```text
--chat-template-kwargs {"enable_thinking":true|false}
```

### Other families

- DeepSeek-R1 gets reasoning-focused defaults.
- Other families (Llama, Mistral, Gemma, Phi, GPT-OSS, unknown) get balanced/coding/creative presets.

## Context guidance

During model selection, the launcher reads each GGUF and extracts max context metadata.
During runtime settings, the prompt shows guidance like:

```text
Context size (max 262,144) [16384]:
```

## Network access

Default host is `0.0.0.0`, so other LAN machines can connect.

Example:

- Server on Mac Studio: `--host 0.0.0.0 --port 8000`
- Client URL: `http://192.168.0.161:8000`

Only forward ports at your router if you explicitly want internet exposure.

## Notes

- If `llama-server` cannot be auto-detected, the launcher prompts for its path.
- Unsupported `llama-server` flags are skipped with a clear note, instead of hard-failing.
- `modelmenu` is a thin wrapper that runs `modelmenu.py`.
