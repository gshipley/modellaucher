#!/usr/bin/env python3
"""
Interactive launcher for serving local GGUF models with llama.cpp.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shlex
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)
SIZE_RE = re.compile(
    r"(?<!\d)(?P<main>\d+(?:\.\d+)?)b(?:[-_]?a(?P<active>\d+(?:\.\d+)?)b)?",
    re.IGNORECASE,
)
QUANT_PATTERNS = [
    re.compile(r"UD-Q\d+_[A-Z0-9_]+", re.IGNORECASE),
    re.compile(r"MXFP4_MOE", re.IGNORECASE),
    re.compile(r"MXFP4", re.IGNORECASE),
    re.compile(r"IQ\d+_[A-Z0-9_]+", re.IGNORECASE),
    re.compile(r"Q\d+_[A-Z0-9_]+", re.IGNORECASE),
    re.compile(r"BF16|F16|FP16|FP8", re.IGNORECASE),
]

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

GGUF_PRIMITIVE_SIZES = {
    GGUF_TYPE_UINT8: 1,
    GGUF_TYPE_INT8: 1,
    GGUF_TYPE_UINT16: 2,
    GGUF_TYPE_INT16: 2,
    GGUF_TYPE_UINT32: 4,
    GGUF_TYPE_INT32: 4,
    GGUF_TYPE_FLOAT32: 4,
    GGUF_TYPE_BOOL: 1,
    GGUF_TYPE_UINT64: 8,
    GGUF_TYPE_INT64: 8,
    GGUF_TYPE_FLOAT64: 8,
}

FAMILY_INFO = {
    "qwen3.5": (
        "Qwen3.5",
        "Hybrid reasoning model with thinking/non-thinking toggle.",
    ),
    "qwen3": (
        "Qwen3",
        "Reasoning-capable Qwen model; works best with explicit think mode control.",
    ),
    "qwen2.5": (
        "Qwen2.5",
        "Strong general/coding model family with reliable instruct behavior.",
    ),
    "deepseek-r1": (
        "DeepSeek R1",
        "Reasoning-focused model family; usually strongest with lower temperature.",
    ),
    "deepseek": (
        "DeepSeek",
        "General model family from DeepSeek.",
    ),
    "llama": (
        "Llama",
        "General-purpose instruct model family.",
    ),
    "mistral": (
        "Mistral",
        "Fast, strong instruct family for general and coding tasks.",
    ),
    "gemma": (
        "Gemma",
        "Instruction model family from Google.",
    ),
    "phi": (
        "Phi",
        "Compact Microsoft models that perform well at smaller sizes.",
    ),
    "gpt-oss": (
        "GPT-OSS",
        "Open MoE model family; balanced defaults are usually a strong starting point.",
    ),
    "nemotron": (
        "Nemotron",
        "NVIDIA Nemotron family; may require a patched llama.cpp build for some GGUF variants.",
    ),
    "unknown": (
        "Unknown",
        "No special preset detected; using conservative defaults.",
    ),
}

FAMILY_COLOR_CODES = {
    "qwen3.5": "1;34",
    "qwen3": "1;34",
    "qwen2.5": "1;34",
    "deepseek-r1": "1;35",
    "deepseek": "35",
    "llama": "33",
    "mistral": "36",
    "gemma": "32",
    "phi": "36",
    "gpt-oss": "1;95",
    "nemotron": "1;33",
    "unknown": "37",
}


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    description: str
    ctx_size: int
    temp: float
    top_p: float
    top_k: int
    min_p: float
    repeat_penalty: float = 1.0
    presence_penalty: Optional[float] = None
    enable_thinking: Optional[bool] = None
    recommended: bool = False


@dataclass(frozen=True)
class ModelEntry:
    path: Path
    family: str
    family_label: str
    size: Optional[str]
    quant: Optional[str]
    max_context: Optional[int]
    sharded_parts: Optional[int]
    mmproj: Optional[Path]
    description: str


@dataclass(frozen=True)
class HardwareProfile:
    key: str
    label: str
    description: str
    logical_cores: int
    performance_cores: Optional[int] = None
    memory_gb: Optional[int] = None
    unified_memory: bool = False


@dataclass(frozen=True)
class RuntimeDefaults:
    ctx_size: int
    threads: int
    use_gpu: bool
    n_gpu_layers: str
    flash_attn: Optional[str]
    parallel: Optional[int]
    notes: tuple[str, ...] = ()


def should_use_color(mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    if os.environ.get("NO_COLOR"):
        return False
    term = os.environ.get("TERM", "").lower()
    if term == "dumb":
        return False
    return sys.stdout.isatty()


def colorize(text: str, code: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


QWEN35_PRESETS = [
    Preset(
        key="qwen35_nonthink_general",
        label="Qwen3.5 non-thinking (general) [Unsloth]",
        description="Stable default for general chat/instruct usage.",
        ctx_size=16384,
        temp=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        repeat_penalty=1.0,
        presence_penalty=1.5,
        enable_thinking=False,
        recommended=True,
    ),
    Preset(
        key="qwen35_nonthink_reasoning",
        label="Qwen3.5 non-thinking (reasoning) [Unsloth]",
        description="Higher entropy non-thinking preset for harder reasoning prompts.",
        ctx_size=16384,
        temp=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        repeat_penalty=1.0,
        presence_penalty=1.5,
        enable_thinking=False,
    ),
    Preset(
        key="qwen35_think_general",
        label="Qwen3.5 thinking (general) [Unsloth]",
        description="Reasoning-enabled default with balanced creativity.",
        ctx_size=16384,
        temp=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        repeat_penalty=1.0,
        presence_penalty=1.5,
        enable_thinking=True,
    ),
    Preset(
        key="qwen35_think_coding",
        label="Qwen3.5 thinking (precise coding) [Unsloth]",
        description="Lower temperature thinking preset for exact coding tasks.",
        ctx_size=16384,
        temp=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        repeat_penalty=1.0,
        presence_penalty=0.0,
        enable_thinking=True,
    ),
]

DEEPSEEK_R1_PRESETS = [
    Preset(
        key="r1_reasoning",
        label="Reasoning focused (recommended)",
        description="Lower temperature preset tuned for chain-of-thought heavy tasks.",
        ctx_size=16384,
        temp=0.6,
        top_p=0.95,
        top_k=40,
        min_p=0.0,
        repeat_penalty=1.0,
        presence_penalty=0.0,
        recommended=True,
    ),
    Preset(
        key="r1_general",
        label="General chat",
        description="Balanced everyday chat preset.",
        ctx_size=16384,
        temp=0.7,
        top_p=0.9,
        top_k=40,
        min_p=0.0,
        repeat_penalty=1.05,
        presence_penalty=0.0,
    ),
]

NEMOTRON_PRESETS = [
    Preset(
        key="nemotron_recommended",
        label="Nemotron recommended (NVIDIA/Unsloth)",
        description="Recommended baseline: temperature=1.0, top_p=0.95.",
        ctx_size=16384,
        temp=1.0,
        top_p=0.95,
        top_k=40,
        min_p=0.0,
        repeat_penalty=1.0,
        presence_penalty=0.0,
        recommended=True,
    ),
    Preset(
        key="nemotron_precise",
        label="Nemotron precise coding",
        description="Lower entropy preset for deterministic coding outputs.",
        ctx_size=16384,
        temp=0.6,
        top_p=0.9,
        top_k=40,
        min_p=0.0,
        repeat_penalty=1.05,
        presence_penalty=0.0,
    ),
]

GENERIC_PRESETS = [
    Preset(
        key="balanced",
        label="Balanced chat (recommended)",
        description="Reliable default for most instruct models.",
        ctx_size=8192,
        temp=0.7,
        top_p=0.9,
        top_k=40,
        min_p=0.0,
        repeat_penalty=1.05,
        presence_penalty=0.0,
        recommended=True,
    ),
    Preset(
        key="precise_coding",
        label="Precise coding",
        description="Lower-temperature preset for deterministic coding answers.",
        ctx_size=8192,
        temp=0.2,
        top_p=0.9,
        top_k=40,
        min_p=0.0,
        repeat_penalty=1.1,
        presence_penalty=0.0,
    ),
    Preset(
        key="creative",
        label="Creative writing",
        description="Higher-temperature preset for brainstorming and ideation.",
        ctx_size=8192,
        temp=0.95,
        top_p=0.95,
        top_k=50,
        min_p=0.0,
        repeat_penalty=1.0,
        presence_penalty=0.0,
    ),
]


def detect_family(model_name: str) -> str:
    name = model_name.lower()
    squashed = re.sub(r"[^a-z0-9]", "", name)

    if "qwen3.5" in name or "qwen35" in squashed:
        return "qwen3.5"
    if "qwen3" in name:
        return "qwen3"
    if "qwen2.5" in name or "qwen25" in squashed:
        return "qwen2.5"
    if "deepseek-r1" in name or "deepseekr1" in squashed:
        return "deepseek-r1"
    if "deepseek" in name:
        return "deepseek"
    if "llama" in name:
        return "llama"
    if "mistral" in name or "mixtral" in name or "ministral" in name:
        return "mistral"
    if "gemma" in name:
        return "gemma"
    if "phi" in name:
        return "phi"
    if "nemotron" in name:
        return "nemotron"
    if "gpt-oss" in name or "gptoss" in squashed:
        return "gpt-oss"
    return "unknown"


def extract_size(model_name: str) -> Optional[str]:
    for match in SIZE_RE.finditer(model_name):
        main = match.group("main")
        active = match.group("active")
        if active:
            return f"{main}B-A{active}B"
        return f"{main}B"
    return None


def extract_quant(model_name: str) -> Optional[str]:
    for pattern in QUANT_PATTERNS:
        match = pattern.search(model_name)
        if match:
            return match.group(0).upper()
    return None


def _read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise EOFError("Unexpected end of file while parsing GGUF metadata.")
    return data


def _read_u32(handle) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _read_u64(handle) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _read_i32(handle) -> int:
    return struct.unpack("<i", _read_exact(handle, 4))[0]


def _read_i64(handle) -> int:
    return struct.unpack("<q", _read_exact(handle, 8))[0]


def _read_f32(handle) -> float:
    return struct.unpack("<f", _read_exact(handle, 4))[0]


def _read_f64(handle) -> float:
    return struct.unpack("<d", _read_exact(handle, 8))[0]


def _read_gguf_string(handle) -> str:
    length = _read_u64(handle)
    return _read_exact(handle, length).decode("utf-8", errors="replace")


def _read_gguf_value(handle, value_type: int):
    if value_type == GGUF_TYPE_UINT8:
        return _read_exact(handle, 1)[0]
    if value_type == GGUF_TYPE_INT8:
        return struct.unpack("<b", _read_exact(handle, 1))[0]
    if value_type == GGUF_TYPE_UINT16:
        return struct.unpack("<H", _read_exact(handle, 2))[0]
    if value_type == GGUF_TYPE_INT16:
        return struct.unpack("<h", _read_exact(handle, 2))[0]
    if value_type == GGUF_TYPE_UINT32:
        return _read_u32(handle)
    if value_type == GGUF_TYPE_INT32:
        return _read_i32(handle)
    if value_type == GGUF_TYPE_FLOAT32:
        return _read_f32(handle)
    if value_type == GGUF_TYPE_BOOL:
        return bool(_read_exact(handle, 1)[0])
    if value_type == GGUF_TYPE_STRING:
        return _read_gguf_string(handle)
    if value_type == GGUF_TYPE_ARRAY:
        element_type = _read_u32(handle)
        count = _read_u64(handle)
        if element_type in GGUF_PRIMITIVE_SIZES:
            _read_exact(handle, GGUF_PRIMITIVE_SIZES[element_type] * count)
            return None
        # Strings and nested arrays are rare in keys we care about, but supported here.
        for _ in range(count):
            _read_gguf_value(handle, element_type)
        return None
    if value_type == GGUF_TYPE_UINT64:
        return _read_u64(handle)
    if value_type == GGUF_TYPE_INT64:
        return _read_i64(handle)
    if value_type == GGUF_TYPE_FLOAT64:
        return _read_f64(handle)
    raise ValueError(f"Unsupported GGUF value type: {value_type}")


def read_max_context_from_gguf(model_path: Path) -> Optional[int]:
    try:
        with model_path.open("rb") as handle:
            if _read_exact(handle, 4) != b"GGUF":
                return None
            version = _read_u32(handle)
            if version == 1:
                _ = _read_u32(handle)  # tensor count
                kv_count = _read_u32(handle)
            else:
                _ = _read_u64(handle)  # tensor count
                kv_count = _read_u64(handle)

            context_candidates: list[int] = []
            for _ in range(kv_count):
                key = _read_gguf_string(handle)
                value_type = _read_u32(handle)
                value = _read_gguf_value(handle, value_type)
                if isinstance(value, int) and (key == "context_length" or key.endswith(".context_length")):
                    if value > 0:
                        context_candidates.append(value)

            if not context_candidates:
                return None
            return max(context_candidates)
    except Exception:
        return None


def parse_shard(filename: str) -> tuple[Optional[int], Optional[int]]:
    match = SHARD_RE.match(filename)
    if not match:
        return None, None
    return int(match.group("index")), int(match.group("total"))


def find_mmproj_file(model_file: Path) -> Optional[Path]:
    parent = model_file.parent
    for pattern in ("mmproj*.gguf", "*mmproj*.gguf", "*mm-project*.gguf"):
        candidates = sorted(parent.glob(pattern))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return None


def build_description(
    family: str,
    family_label: str,
    family_text: str,
    size: Optional[str],
    quant: Optional[str],
    sharded_parts: Optional[int],
    mmproj: Optional[Path],
) -> str:
    extras = []
    if size:
        extras.append(size)
    if quant:
        extras.append(quant)
    if sharded_parts:
        extras.append(f"{sharded_parts} shards")
    if mmproj:
        extras.append("mmproj")
    extras_text = f" ({', '.join(extras)})" if extras else ""
    return f"{family_label}{extras_text} - {family_text}"


def discover_models(model_root: Path) -> list[ModelEntry]:
    models: list[ModelEntry] = []
    if not model_root.exists():
        return models

    seen: set[Path] = set()
    for path in sorted(model_root.rglob("*.gguf")):
        if not path.is_file():
            continue
        if "mmproj" in path.name.lower():
            continue

        shard_index, shard_total = parse_shard(path.name)
        if shard_total is not None and shard_index != 1:
            continue

        resolved_path = path.resolve()
        if resolved_path in seen:
            continue
        seen.add(resolved_path)

        family = detect_family(path.stem)
        family_label, family_text = FAMILY_INFO[family]
        size = extract_size(path.stem)
        quant = extract_quant(path.stem)
        max_context = read_max_context_from_gguf(path)
        mmproj = find_mmproj_file(path)
        description = build_description(
            family=family,
            family_label=family_label,
            family_text=family_text,
            size=size,
            quant=quant,
            sharded_parts=shard_total,
            mmproj=mmproj,
        )
        models.append(
            ModelEntry(
                path=path,
                family=family,
                family_label=family_label,
                size=size,
                quant=quant,
                max_context=max_context,
                sharded_parts=shard_total,
                mmproj=mmproj,
                description=description,
            )
        )

    return sorted(models, key=lambda item: item.path.name.lower())


def presets_for_family(family: str) -> list[Preset]:
    if family in {"qwen3.5", "qwen3"}:
        return QWEN35_PRESETS
    if family == "deepseek-r1":
        return DEEPSEEK_R1_PRESETS
    if family == "nemotron":
        return NEMOTRON_PRESETS
    return GENERIC_PRESETS


def default_threads() -> int:
    cpus = os.cpu_count() or 4
    if cpus <= 4:
        return cpus
    if cpus <= 12:
        return cpus - 1
    return cpus - 2


def recommended_threads_for_hardware(hardware: HardwareProfile) -> tuple[int, Optional[str]]:
    if hardware.performance_cores:
        return hardware.performance_cores, (
            f"Threads default to the detected performance cores ({hardware.performance_cores}) instead of all logical cores."
        )

    if hardware.key == "apple-silicon-high-memory" and hardware.logical_cores >= 32:
        inferred_perf_cores = 24
        return inferred_perf_cores, (
            f"Threads default to an inferred performance-core count ({inferred_perf_cores}) for this M3 Ultra-class profile."
        )

    if hardware.key == "apple-silicon" and hardware.logical_cores >= 16:
        inferred_perf_cores = max(8, hardware.logical_cores - 4)
        return inferred_perf_cores, (
            f"Threads default to an inferred performance-core count ({inferred_perf_cores}) for Apple Silicon."
        )

    return default_threads(), None


def find_llama_server(explicit: Optional[str]) -> Optional[Path]:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())

    env_bin = os.environ.get("LLAMA_SERVER_BIN")
    if env_bin:
        candidates.append(Path(env_bin).expanduser())

    which_bin = shutil.which("llama-server")
    if which_bin:
        candidates.append(Path(which_bin))

    candidates.extend(
        [
            (Path.cwd() / "llama.cpp" / "llama-server"),
            (Path.home() / "llama.cpp" / "llama-server"),
            (Path.home() / "code" / "llama.cpp" / "llama-server"),
        ]
    )

    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def prompt_for_llama_server() -> Optional[Path]:
    while True:
        entered = input("Path to llama-server (or q to quit): ").strip()
        if entered.lower() in {"q", "quit", "exit"}:
            return None
        candidate = Path(entered).expanduser()
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate.resolve()
        print("Not an executable file. Try again.")


def supported_flags(llama_server: Path) -> set[str]:
    try:
        proc = subprocess.run(
            [str(llama_server), "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except Exception:
        return set()
    return set(re.findall(r"(--[a-z0-9][a-z0-9\-]*)", output))


def format_command(parts: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def choose_index(count: int, prompt: str) -> int:
    while True:
        raw = input(prompt).strip()
        if raw.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        try:
            idx = int(raw)
        except ValueError:
            print("Enter a number.")
            continue
        if 1 <= idx <= count:
            return idx
        print(f"Choose a value between 1 and {count}.")


def ask_text(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


def ask_int(prompt: str, default: int, minimum: int = 0) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Enter a whole number.")
            continue
        if value < minimum:
            print(f"Value must be >= {minimum}.")
            continue
        return value


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    default_label = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{default_label}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Enter y or n.")


def ask_choice(prompt: str, default: str, allowed: Iterable[str]) -> str:
    allowed_set = {value.lower() for value in allowed}
    allowed_text = "/".join(allowed)
    while True:
        raw = input(f"{prompt} [{default}]: ").strip().lower()
        if not raw:
            return default
        if raw in allowed_set:
            return raw
        print(f"Enter one of: {allowed_text}.")


def ask_n_gpu_layers(default: str) -> str:
    while True:
        raw = input(f"GPU layers [{default}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"auto", "all"}:
            return raw
        try:
            value = int(raw)
        except ValueError:
            print("Enter a whole number, 'auto', or 'all'.")
            continue
        if value < 0:
            print("Value must be >= 0.")
            continue
        return str(value)


def run_text_command(parts: list[str], timeout: float = 2.0) -> Optional[str]:
    try:
        proc = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except Exception:
        return None
    output = (proc.stdout or "").strip()
    if output:
        return output
    error_text = (proc.stderr or "").strip()
    return error_text or None


def detect_memory_gb() -> Optional[int]:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and phys_pages > 0:
            return int((page_size * phys_pages) / (1024**3))
    except (AttributeError, OSError, ValueError):
        pass

    if sys.platform == "darwin":
        output = run_text_command(["sysctl", "-n", "hw.memsize"])
        if output:
            try:
                return int(int(output) / (1024**3))
            except ValueError:
                return None
    return None


def detect_perf_cores() -> Optional[int]:
    if sys.platform != "darwin":
        return None
    output = run_text_command(["sysctl", "-n", "hw.perflevel0.physicalcpu"])
    if not output:
        return None
    try:
        value = int(output)
    except ValueError:
        return None
    return value if value > 0 else None


def build_hardware_profile(profile_key: str) -> HardwareProfile:
    logical_cores = os.cpu_count() or 4
    performance_cores = detect_perf_cores()
    memory_gb = detect_memory_gb()

    if profile_key == "apple-silicon-high-memory":
        memory_text = f"{memory_gb} GB unified memory" if memory_gb else "high-memory unified memory"
        return HardwareProfile(
            key=profile_key,
            label="Apple Silicon high-memory",
            description=f"M3 Ultra-class Mac profile tuned for very large GGUF and MoE models ({memory_text}).",
            logical_cores=logical_cores,
            performance_cores=performance_cores,
            memory_gb=memory_gb,
            unified_memory=True,
        )
    if profile_key == "apple-silicon":
        memory_text = f"{memory_gb} GB unified memory" if memory_gb else "unified memory"
        return HardwareProfile(
            key=profile_key,
            label="Apple Silicon",
            description=f"Mac profile tuned for Metal offload on unified-memory systems ({memory_text}).",
            logical_cores=logical_cores,
            performance_cores=performance_cores,
            memory_gb=memory_gb,
            unified_memory=True,
        )
    memory_text = f", {memory_gb} GB RAM" if memory_gb else ""
    return HardwareProfile(
        key="generic",
        label="Generic host",
        description=f"Portable defaults based on detected CPU count{memory_text}.",
        logical_cores=logical_cores,
        performance_cores=performance_cores,
        memory_gb=memory_gb,
        unified_memory=False,
    )


def detect_hardware_profile(selection: str) -> HardwareProfile:
    if selection != "auto":
        return build_hardware_profile(selection)

    is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
    memory_gb = detect_memory_gb()
    if is_apple_silicon and memory_gb is not None and memory_gb >= 256:
        return build_hardware_profile("apple-silicon-high-memory")
    if is_apple_silicon:
        return build_hardware_profile("apple-silicon")
    return build_hardware_profile("generic")


def parse_model_size_billions(size: Optional[str]) -> tuple[Optional[float], Optional[float]]:
    if not size:
        return None, None
    match = SIZE_RE.search(size)
    if not match:
        return None, None
    main = float(match.group("main"))
    active = match.group("active")
    return main, float(active) if active else None


def recommend_runtime_defaults(
    model: ModelEntry,
    preset: Preset,
    hardware: HardwareProfile,
    known_flags: set[str],
) -> RuntimeDefaults:
    total_b, active_b = parse_model_size_billions(model.size)
    effective_b = active_b if active_b is not None else total_b
    very_large_moe = (total_b is not None and total_b >= 200) or (effective_b is not None and effective_b >= 50)
    large_model = (total_b is not None and total_b >= 70) or (effective_b is not None and effective_b >= 20)

    supports_flash_attn = not known_flags or "--flash-attn" in known_flags
    supports_parallel = not known_flags or "--parallel" in known_flags

    ctx_size = preset.ctx_size
    threads, thread_note = recommended_threads_for_hardware(hardware)
    use_gpu = True
    n_gpu_layers = "99"
    flash_attn = "auto" if supports_flash_attn else None
    parallel = 1 if supports_parallel else None
    notes: list[str] = []

    if hardware.key == "apple-silicon-high-memory":
        n_gpu_layers = "all"
        if very_large_moe:
            ctx_size = max(ctx_size, 131072)
            parallel = 1 if supports_parallel else None
            notes.append("Large MoE detected: defaulting to a single slot to preserve KV headroom and steadier latency.")
        elif large_model:
            ctx_size = max(ctx_size, 65536)
            parallel = 2 if supports_parallel else None
        else:
            ctx_size = max(ctx_size, 32768)
            parallel = 4 if supports_parallel else None
        notes.append("Apple Silicon unified memory usually performs best with full Metal offload when the model fits.")
    elif hardware.key == "apple-silicon":
        n_gpu_layers = "all"
        if very_large_moe:
            ctx_size = max(ctx_size, 32768)
            parallel = 1 if supports_parallel else None
        elif large_model:
            ctx_size = max(ctx_size, 16384)
            parallel = 2 if supports_parallel else None
        else:
            parallel = 2 if supports_parallel else None
        notes.append("Apple Silicon defaults lean toward full Metal offload and conservative server slot counts.")
    else:
        if very_large_moe:
            parallel = 1 if supports_parallel else None
        elif large_model and supports_parallel and (hardware.memory_gb or 0) >= 128:
            parallel = 2

    if thread_note:
        notes.append(thread_note)

    if model.max_context:
        ctx_size = min(ctx_size, model.max_context)

    return RuntimeDefaults(
        ctx_size=ctx_size,
        threads=max(1, threads),
        use_gpu=use_gpu,
        n_gpu_layers=n_gpu_layers,
        flash_attn=flash_attn,
        parallel=parallel,
        notes=tuple(notes),
    )


def build_llama_server_cmd(
    llama_server: Path,
    model: ModelEntry,
    preset: Preset,
    host: str,
    port: int,
    alias: str,
    ctx_size: int,
    threads: int,
    n_gpu_layers: str,
    flash_attn: Optional[str],
    parallel: Optional[int],
    known_flags: set[str],
) -> tuple[list[str], list[str]]:
    cmd: list[str] = [str(llama_server)]
    skipped: list[str] = []

    def add(flag: str, value: Optional[str] = None) -> None:
        if known_flags and flag not in known_flags:
            skipped.append(flag)
            return
        cmd.append(flag)
        if value is not None:
            cmd.append(value)

    add("--model", str(model.path))
    if model.mmproj:
        add("--mmproj", str(model.mmproj))
    add("--host", host)
    add("--port", str(port))
    add("--alias", alias)
    add("--ctx-size", str(ctx_size))
    add("--threads", str(threads))
    add("--n-gpu-layers", str(n_gpu_layers))
    if flash_attn is not None:
        add("--flash-attn", flash_attn)
    if parallel is not None:
        add("--parallel", str(parallel))

    add("--temp", f"{preset.temp:g}")
    add("--top-p", f"{preset.top_p:g}")
    add("--top-k", str(preset.top_k))
    add("--min-p", f"{preset.min_p:g}")
    add("--repeat-penalty", f"{preset.repeat_penalty:g}")
    if preset.presence_penalty is not None:
        add("--presence-penalty", f"{preset.presence_penalty:g}")
    if preset.enable_thinking is not None:
        thinking = "true" if preset.enable_thinking else "false"
        add("--chat-template-kwargs", f'{{"enable_thinking":{thinking}}}')

    return cmd, skipped


def compact_model_name(filename: str) -> str:
    stem = re.sub(r"\.gguf$", "", filename, flags=re.IGNORECASE)
    stem = re.sub(r"-\d{5}-of-\d{5}$", "", stem)
    return stem


def format_context_window(value: Optional[int]) -> str:
    if value is None:
        return "unknown"
    return f"{value:,}"


def describe_hardware_profile(hardware: HardwareProfile) -> str:
    parts = [hardware.label]
    if hardware.memory_gb:
        memory_label = "unified memory" if hardware.unified_memory else "RAM"
        parts.append(f"{hardware.memory_gb} GB {memory_label}")
    if hardware.performance_cores:
        parts.append(f"{hardware.performance_cores} perf cores")
    elif hardware.logical_cores:
        parts.append(f"{hardware.logical_cores} logical cores")
    return " | ".join(parts)


def format_metric(
    label: str,
    value: str,
    width: int,
    value_code: str,
    use_color: bool,
) -> str:
    label_text = colorize(f"{label}=", "2;37", use_color)
    value_text = colorize(f"{value:<{width}}", value_code, use_color)
    return f"{label_text}{value_text}"


def print_models(models: list[ModelEntry], model_root: Path, use_color: bool) -> None:
    print()
    print(colorize("Discovered models:", "1;36", use_color))
    for i, model in enumerate(models, start=1):
        rel_path = model.path
        try:
            rel_path = model.path.resolve().relative_to(model_root.resolve())
        except ValueError:
            rel_path = model.path.resolve()

        size = model.size or "-"
        quant = model.quant or "-"
        shards = str(model.sharded_parts) if model.sharded_parts else "-"
        has_mmproj = "yes" if model.mmproj else "no"
        max_ctx = format_context_window(model.max_context)

        index_text = colorize(f"{i:>2}.", "1;35", use_color)
        name_text = colorize(compact_model_name(model.path.name), "1;97", use_color)
        print(f"{index_text} {name_text}")
        family_code = FAMILY_COLOR_CODES.get(model.family, "37")
        quant_code = "1;33" if model.quant else "37"
        max_ctx_code = "1;36" if model.max_context else "31"
        shards_code = "1;32" if model.sharded_parts else "37"
        mmproj_code = "1;32" if model.mmproj else "31"
        metrics = " ".join(
            [
                format_metric("family", model.family_label, 10, family_code, use_color),
                format_metric("size", size, 10, "37", use_color),
                format_metric("quant", quant, 12, quant_code, use_color),
                format_metric("max_ctx", max_ctx, 8, max_ctx_code, use_color),
                format_metric("shards", shards, 3, shards_code, use_color),
                format_metric("mmproj", has_mmproj, 3, mmproj_code, use_color),
            ]
        )
        path_label = colorize("path:", "2;37", use_color)
        path_text = colorize(str(rel_path), "37", use_color)
        note_label = colorize("note:", "2;37", use_color)
        note_text = colorize(FAMILY_INFO[model.family][1], "2;37", use_color)
        print(
            f"    {metrics}"
        )
        print(f"    {path_label} {path_text}")
        print(f"    {note_label} {note_text}")


def print_presets(presets: list[Preset], use_color: bool) -> None:
    print()
    print(colorize("Available serving presets:", "1;36", use_color))
    for i, preset in enumerate(presets, start=1):
        has_recommended_word = "recommended" in preset.label.lower()
        recommended = " [recommended]" if preset.recommended and not has_recommended_word else ""
        index_text = colorize(f"{i:>2}.", "1;35", use_color)
        label_code = "1;32" if preset.recommended else "37"
        label_text = colorize(f"{preset.label}{recommended}", label_code, use_color)
        description_text = colorize(preset.description, "2;37", use_color)
        print(f"{index_text} {label_text}")
        print(f"    {description_text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive launcher for serving local GGUF models with llama.cpp."
    )
    parser.add_argument(
        "--models-dir",
        default="~/models",
        help="Directory to recursively scan for GGUF models (default: ~/models).",
    )
    parser.add_argument(
        "--llama-server",
        default=None,
        help="Path to llama-server binary (optional).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Default host for llama-server.")
    parser.add_argument("--port", type=int, default=8000, help="Default port for llama-server.")
    parser.add_argument(
        "--hardware-profile",
        choices=("auto", "generic", "apple-silicon", "apple-silicon-high-memory"),
        default="auto",
        help="Hardware-tuned defaults profile (default: auto-detect).",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List discovered models and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print command but do not launch llama-server.",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Control colored output (default: auto).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    use_color = should_use_color(args.color)
    model_root = Path(args.models_dir).expanduser()
    model_label = colorize("Model directory:", "2;37", use_color)
    model_dir_text = colorize(str(model_root), "37", use_color)
    print(f"{model_label} {model_dir_text}")

    models = discover_models(model_root)
    if not models:
        print("No model files found. Add GGUF files under the model directory and retry.")
        return 1

    print_models(models, model_root, use_color)
    if args.list_only:
        return 0

    llama_server = find_llama_server(args.llama_server)
    if llama_server is None:
        print()
        print("Could not find llama-server automatically.")
        llama_server = prompt_for_llama_server()
        if llama_server is None:
            print("Aborted.")
            return 1

    print()
    print(f"Using llama-server: {llama_server}")
    known_flags = supported_flags(llama_server)
    hardware = detect_hardware_profile(args.hardware_profile)
    print(colorize(f"Hardware profile: {describe_hardware_profile(hardware)}", "2;37", use_color))
    print(colorize(hardware.description, "2;37", use_color))

    print()
    print("Pick a model number (or q to quit):")
    model_idx = choose_index(len(models), "Model> ")
    selected_model = models[model_idx - 1]

    presets = presets_for_family(selected_model.family)
    print_presets(presets, use_color)
    print()
    print("Pick a preset number (or q to quit):")
    preset_idx = choose_index(len(presets), "Preset> ")
    selected_preset = presets[preset_idx - 1]

    if selected_model.family == "nemotron":
        print()
        print(colorize("Nemotron compatibility note:", "1;33", use_color))
        print(
            colorize(
                "If model loading fails with tensor shape mismatch, use Unsloth's llama.cpp "
                "nvidia-fix branch for Nemotron Super GGUF files.",
                "33",
                use_color,
            )
        )

    runtime_defaults = recommend_runtime_defaults(
        model=selected_model,
        preset=selected_preset,
        hardware=hardware,
        known_flags=known_flags,
    )

    print()
    print(colorize("Recommended defaults for this model and host:", "1;36", use_color))
    print(
        colorize(
            f"ctx={format_context_window(runtime_defaults.ctx_size)} "
            f"threads={runtime_defaults.threads} "
            f"gpu_layers={runtime_defaults.n_gpu_layers}",
            "37",
            use_color,
        )
    )
    if runtime_defaults.flash_attn is not None:
        print(colorize(f"flash_attn={runtime_defaults.flash_attn}", "37", use_color))
    if runtime_defaults.parallel is not None:
        print(colorize(f"parallel={runtime_defaults.parallel}", "37", use_color))
    for note in runtime_defaults.notes:
        print(colorize(f"- {note}", "2;37", use_color))

    print()
    print("Runtime settings (Enter accepts default):")
    host = ask_text("Host", args.host)
    port = ask_int("Port", args.port, minimum=1)
    if selected_model.max_context:
        ctx_prompt = f"Context size (max {format_context_window(selected_model.max_context)})"
    else:
        ctx_prompt = "Context size"
    ctx_size = ask_int(ctx_prompt, runtime_defaults.ctx_size, minimum=256)
    threads = ask_int("Threads", runtime_defaults.threads, minimum=1)
    use_gpu = ask_yes_no("Enable GPU offload (--n-gpu-layers)", default=runtime_defaults.use_gpu)
    if use_gpu:
        n_gpu_layers = ask_n_gpu_layers(runtime_defaults.n_gpu_layers)
    else:
        n_gpu_layers = "0"
    flash_attn = runtime_defaults.flash_attn
    if flash_attn is not None:
        flash_attn = ask_choice("Flash attention (--flash-attn)", flash_attn, ("auto", "on", "off"))
    parallel = runtime_defaults.parallel
    if parallel is not None:
        parallel = ask_int("Parallel server slots (--parallel)", parallel, minimum=1)
    alias = ask_text("Model alias", selected_model.path.stem)

    cmd, skipped_flags = build_llama_server_cmd(
        llama_server=llama_server,
        model=selected_model,
        preset=selected_preset,
        host=host,
        port=port,
        alias=alias,
        ctx_size=ctx_size,
        threads=threads,
        n_gpu_layers=n_gpu_layers,
        flash_attn=flash_attn,
        parallel=parallel,
        known_flags=known_flags,
    )

    print()
    print("Generated command:")
    print(format_command(cmd))

    if skipped_flags:
        unique_skipped = ", ".join(sorted(set(skipped_flags)))
        print()
        print(f"Note: your llama-server build does not advertise these flags: {unique_skipped}")
        if "--chat-template-kwargs" in skipped_flags and selected_model.family.startswith("qwen"):
            print("For Qwen thinking control, use /think or /no_think inside prompts.")

    if args.dry_run:
        print()
        print("Dry run mode: command not executed.")
        return 0

    print()
    if not ask_yes_no("Launch server now", default=True):
        print("Aborted before launch.")
        return 0

    print()
    print("Launching llama-server (Ctrl+C to stop)...")
    run = subprocess.run(cmd, check=False)
    return run.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
