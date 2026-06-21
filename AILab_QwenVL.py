# ComfyUI-QwenVL
# This custom node integrates the Qwen-VL series, including the latest Qwen3-VL models,
# including Qwen2.5-VL and the latest Qwen3-VL, to enable advanced multimodal AI for text generation,
# image understanding, and video analysis.
#
# Models License Notice:
# - Qwen3-VL: Apache-2.0 License (https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
# - Qwen2.5-VL: Apache-2.0 License (https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
#
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-QwenVL

import gc
import json
import os
import platform
from contextlib import contextmanager
from enum import Enum
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image
from huggingface_hub import snapshot_download
try:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForVision2Seq
from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig

import folder_paths
from comfy.utils import ProgressBar

# SageAttention support
try:
    from sageattention.core import (
        sageattn_qk_int8_pv_fp16_cuda,
        sageattn_qk_int8_pv_fp8_cuda,
        sageattn_qk_int8_pv_fp8_cuda_sm90,
    )
    SAGE_ATTENTION_AVAILABLE = True
except ImportError:
    SAGE_ATTENTION_AVAILABLE = False

NODE_DIR = Path(__file__).parent
CONFIG_PATH = NODE_DIR / "hf_models.json"
SYSTEM_PROMPTS_PATH = NODE_DIR / "AILab_System_Prompts.json"
BYTES_PER_GIB = 1024 ** 3
NVFP4_F4_E2M1_MAX = 6.0
NVFP4_F8_E4M3_MAX = 448.0
HF_VL_MODELS: dict[str, dict] = {}
HF_TEXT_MODELS: dict[str, dict] = {}
HF_ALL_MODELS: dict[str, dict] = {}
SYSTEM_PROMPTS = {}
PRESET_PROMPTS: list[str] = ["Describe this image in detail."]

TOOLTIPS = {
    "model_name": "Pick the Qwen-VL checkpoint. First run downloads weights into models/LLM/Qwen-VL, so leave disk space.",
    "quantization": "Precision vs VRAM. FP16 gives the best quality if memory allows; 8-bit suits 8–16 GB GPUs; 4-bit fits 6 GB or lower but is slower.",
    "attention_mode": "auto tries flash-attn v2 when installed and falls back to SDPA. Only override when debugging attention backends.",
    "preset_prompt": "Built-in instruction describing how Qwen-VL should analyze the media input.",
    "custom_prompt": "Optional override—when filled it completely replaces the preset template.",
    "max_tokens": "Maximum number of new tokens to decode. Larger values yield longer answers but consume more time and memory.",
    "keep_model_loaded": "Keeps the model resident in VRAM/RAM after the run so the next prompt skips loading.",
    "seed": "Seed controlling sampling and frame picking; reuse it to reproduce results.",
    "use_torch_compile": "Enable torch.compile('reduce-overhead') on supported CUDA/Torch 2.1+ builds for extra throughput after the first compile.",
    "device": "Choose where to run the model: auto, cpu, mps, or cuda:x for multi-GPU systems.",
    "temperature": "Sampling randomness when num_beams == 1. 0.2–0.4 is focused, 0.7+ is creative.",
    "top_p": "Nucleus sampling cutoff when num_beams == 1. Lower values keep only top tokens; 0.9–0.95 allows more variety.",
    "num_beams": "Beam-search width. Values >1 disable temperature/top_p and trade speed for more stable answers.",
    "repetition_penalty": "Values >1 (e.g., 1.1–1.3) penalize repeated phrases; 1.0 leaves logits untouched.",
    "frame_count": "Number of frames extracted from video inputs before prompting Qwen-VL. More frames provide context but cost time.",
}

class Quantization(str, Enum):
    Q4 = "4-bit (VRAM-friendly)"
    Q8 = "8-bit (Balanced)"
    FP16 = "None (FP16)"

    @classmethod
    def get_values(cls):
        return [item.value for item in cls]

    @classmethod
    def from_value(cls, value):
        for item in cls:
            if item.value == value:
                return item
        raise ValueError(f"Unsupported quantization: {value}")

ATTENTION_MODES = ["auto", "sage", "flash_attention_2", "sdpa"]

def load_model_configs():
    global HF_VL_MODELS, HF_TEXT_MODELS, HF_ALL_MODELS, SYSTEM_PROMPTS, PRESET_PROMPTS
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        if "hf_vl_models" in data or "hf_text_models" in data:
            HF_VL_MODELS = data.get("hf_vl_models") or {}
            HF_TEXT_MODELS = data.get("hf_text_models") or {}
        else:
            HF_VL_MODELS = {k: v for k, v in data.items() if not k.startswith("_")}
            HF_TEXT_MODELS = {}
        SYSTEM_PROMPTS = data.get("_system_prompts", {})
        PRESET_PROMPTS = data.get("_preset_prompts", PRESET_PROMPTS)
    except Exception as exc:
        print(f"[QwenVL] Config load failed: {exc}")
        HF_VL_MODELS = {}
        HF_TEXT_MODELS = {}
        HF_ALL_MODELS = {}
        SYSTEM_PROMPTS = {}
    try:
        with open(SYSTEM_PROMPTS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        qwenvl_prompts = data.get("qwenvl") or {}
        preset_override = data.get("_preset_prompts") or []
        if isinstance(qwenvl_prompts, dict) and qwenvl_prompts:
            SYSTEM_PROMPTS = qwenvl_prompts
        if isinstance(preset_override, list) and preset_override:
            PRESET_PROMPTS = preset_override
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[QwenVL] System prompts load failed: {exc}")
    custom = NODE_DIR / "custom_models.json"
    if custom.exists():
        try:
            with open(custom, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
            custom_vl = data.get("hf_vl_models") or {}
            custom_text = data.get("hf_text_models") or {}
            legacy = data.get("hf_models", {}) or data.get("models", {})
            if isinstance(custom_vl, dict) and custom_vl:
                HF_VL_MODELS.update(custom_vl)
                print(f"[QwenVL] Loaded {len(custom_vl)} custom VL models")
            if isinstance(custom_text, dict) and custom_text:
                HF_TEXT_MODELS.update(custom_text)
                print(f"[QwenVL] Loaded {len(custom_text)} custom text models")
            if isinstance(legacy, dict) and legacy:
                HF_VL_MODELS.update(legacy)
                print(f"[QwenVL] Loaded {len(legacy)} custom legacy models")
        except Exception as exc:
            print(f"[QwenVL] custom_models.json skipped: {exc}")
    HF_ALL_MODELS = dict(HF_VL_MODELS)
    HF_ALL_MODELS.update(HF_TEXT_MODELS)

if not HF_ALL_MODELS:
    load_model_configs()

def get_device_info():
    gpu = {"available": False, "total_memory": 0, "free_memory": 0}
    device_type = "cpu"
    recommended = "cpu"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024**3
        gpu = {
            "available": True,
            "total_memory": total,
            "free_memory": total - (torch.cuda.memory_allocated(0) / 1024**3),
        }
        device_type = "nvidia_gpu"
        recommended = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device_type = "apple_silicon"
        recommended = "mps"
        gpu = {"available": True, "total_memory": 0, "free_memory": 0}
    sys_mem = psutil.virtual_memory()
    return {
        "gpu": gpu,
        "system_memory": {
            "total": sys_mem.total / 1024**3,
            "available": sys_mem.available / 1024**3,
        },
        "device_type": device_type,
        "recommended_device": recommended,
    }

def normalize_device_choice(device: str) -> str:
    device = (device or "auto").strip()
    if device == "auto":
        return "auto"

    if device.isdigit():
        device = f"cuda:{int(device)}"

    if device == "cuda":
        if not torch.cuda.is_available():
            print("[QwenVL] CUDA requested but not available, falling back to CPU")
            return "cpu"
        return "cuda"

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[QwenVL] CUDA requested but not available, falling back to CPU")
            return "cpu"
        if ":" in device:
            try:
                device_idx = int(device.split(":", 1)[1])
                if device_idx >= torch.cuda.device_count():
                    print(f"[QwenVL] CUDA device {device_idx} not available, using cuda:0")
                    return "cuda:0"
            except (ValueError, IndexError):
                print(f"[QwenVL] Invalid CUDA device format '{device}', using cuda:0")
                return "cuda:0"
        return device

    if device == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            print("[QwenVL] MPS requested but not available, falling back to CPU")
            return "cpu"
        return "mps"

    return device

def flash_attn_available():
    #if platform.system() != "Linux":
    #    return False
    if not torch.cuda.is_available():
        return False

    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        return False

    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False

    try:
        import importlib.metadata as importlib_metadata
        _ = importlib_metadata.version("flash_attn")
    except Exception:
        return False

    return True


def sage_attn_available():
    """Check if SageAttention is available and GPU supports it."""
    if not SAGE_ATTENTION_AVAILABLE:
        return False
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        return False
    return True


def get_sage_attention_config():
    """Get the appropriate SageAttention kernel based on GPU architecture."""
    if not sage_attn_available():
        return None, None, None

    major, minor = torch.cuda.get_device_capability()
    arch_code = major * 10 + minor

    attn_func = None
    pv_accum_dtype = "fp32"

    if arch_code >= 120:  # Blackwell
        pv_accum_dtype = "fp32+fp32"
        attn_func = sageattn_qk_int8_pv_fp8_cuda
        print(f"[QwenVL] SageAttention: Using SM120 (Blackwell) FP8 kernel")
    elif arch_code >= 90:  # Hopper
        pv_accum_dtype = "fp32+fp32"
        attn_func = sageattn_qk_int8_pv_fp8_cuda_sm90
        print(f"[QwenVL] SageAttention: Using SM90 (Hopper) FP8 kernel")
    elif arch_code == 89:  # Ada Lovelace
        pv_accum_dtype = "fp32+fp32"
        attn_func = sageattn_qk_int8_pv_fp8_cuda
        print(f"[QwenVL] SageAttention: Using SM89 (Ada) FP8 kernel")
    elif arch_code >= 80:  # Ampere
        pv_accum_dtype = "fp32"
        attn_func = sageattn_qk_int8_pv_fp16_cuda
        print(f"[QwenVL] SageAttention: Using SM80+ (Ampere) FP16 kernel")
    else:
        print(f"[QwenVL] SageAttention not supported on SM{arch_code}")
        return None, None, None

    return attn_func, "per_warp", pv_accum_dtype

def resolve_attention_mode(mode, force_sdpa=False):
    """Resolve attention mode with fallback logic.
    
    Args:
        mode: The requested attention mode
        force_sdpa: If True, always return SDPA (for FP8/BnB models)
    """
    if force_sdpa:
        return "sdpa"
    
    if mode == "sdpa":
        return "sdpa"
    if mode == "sage":
        if sage_attn_available():
            return "sage"
        print("[QwenVL] SageAttention forced but unavailable, falling back to SDPA")
        return "sdpa"
    if mode == "flash_attention_2":
        if flash_attn_available():
            return "flash_attention_2"
        print("[QwenVL] Flash-Attn forced but unavailable, falling back to SDPA")
        return "sdpa"
    
    # Auto mode: try sage → flash → sdpa
    if sage_attn_available():
        print("[QwenVL] Auto mode: Using SageAttention")
        return "sage"
    if flash_attn_available():
        print("[QwenVL] Auto mode: Using Flash Attention 2")
        return "flash_attention_2"
    print("[QwenVL] Auto mode: Using SDPA")
    return "sdpa"


def set_sage_attention(model):
    """
    Apply SageAttention patching to the model.
    Patches Qwen2Attention and Qwen3VLTextAttention modules to use SageAttention kernels.
    """
    if not sage_attn_available():
        raise ImportError("SageAttention library is not installed or GPU doesn't support it.")

    SAGE_ATTN_FUNC, QK_QUANT_GRAN, PV_ACCUM_DTYPE = get_sage_attention_config()
    if SAGE_ATTN_FUNC is None:
        raise RuntimeError("No compatible SageAttention kernel found for this GPU.")

    # Try to import different attention classes for different Qwen models
    attention_classes = []
    
    # Qwen2 models
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb as qwen2_apply_rotary
        attention_classes.append((Qwen2Attention, qwen2_apply_rotary))
    except ImportError:
        pass
    
    # Qwen3 models (Qwen3-VL, etc.)
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb as qwen3_apply_rotary
        attention_classes.append((Qwen3Attention, qwen3_apply_rotary))
    except ImportError:
        pass
    
    # Qwen3-VL specific
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention, apply_rotary_pos_emb as qwen3vl_apply_rotary
        attention_classes.append((Qwen3VLTextAttention, qwen3vl_apply_rotary))
    except ImportError:
        pass
    
    if not attention_classes:
        print("[QwenVL] Could not import any attention classes for SageAttention patching")
        return

    def make_sage_forward(AttentionClass, apply_rotary_pos_emb_func):
        def sage_attention_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple = None,
            attention_mask: torch.Tensor = None,
            past_key_values=None,
            cache_position: torch.LongTensor = None,
            position_ids: torch.LongTensor = None,
            **kwargs,
        ):
            original_dtype = hidden_states.dtype

            # Determine target dtype
            is_4bit = hasattr(self.q_proj, 'quant_state')
            if is_4bit:
                target_dtype = torch.bfloat16
            else:
                target_dtype = self.q_proj.weight.dtype

            if hidden_states.dtype != target_dtype:
                hidden_states = hidden_states.to(target_dtype)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            bsz, q_len = input_shape[0], input_shape[1] if len(input_shape) > 1 else hidden_states.size(1)

            # Handle q_norm and k_norm for Qwen3-VL
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            # Apply normalization if available (Qwen3-VL specific)
            if hasattr(self, 'q_norm'):
                query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            else:
                query_states = query_states.view(hidden_shape).transpose(1, 2)
            
            if hasattr(self, 'k_norm'):
                key_states = self.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
            else:
                key_states = key_states.view(hidden_shape).transpose(1, 2)
            
            value_states = value_states.view(hidden_shape).transpose(1, 2)

            # Apply rotary embeddings
            if position_embeddings is not None:
                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb_func(query_states, key_states, cos, sin)

            if past_key_values is not None:
                cache_kwargs = {"sin": sin if position_embeddings else None, "cos": cos if position_embeddings else None, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

            is_causal = attention_mask is None and q_len > 1

            attn_output = SAGE_ATTN_FUNC(
                query_states.to(target_dtype),
                key_states.to(target_dtype),
                value_states.to(target_dtype),
                tensor_layout="HND",
                is_causal=is_causal,
                qk_quant_gran=QK_QUANT_GRAN,
                pv_accum_dtype=PV_ACCUM_DTYPE,
            )

            if isinstance(attn_output, tuple):
                attn_output = attn_output[0]

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1)

            attn_output = self.o_proj(attn_output)

            if attn_output.dtype != original_dtype:
                attn_output = attn_output.to(original_dtype)

            return attn_output, None
        
        return sage_attention_forward

    # Apply patching to all supported attention modules
    patched_count = 0
    for AttentionClass, apply_rotary_func in attention_classes:
        sage_forward = make_sage_forward(AttentionClass, apply_rotary_func)
        for module in model.modules():
            if isinstance(module, AttentionClass):
                module.forward = sage_forward.__get__(module, AttentionClass)
                patched_count += 1

    if patched_count > 0:
        print(f"[QwenVL] SageAttention: Patched {patched_count} attention layers")
    else:
        print("[QwenVL] SageAttention: No compatible attention layers found to patch")

def ensure_model(model_name):
    info = HF_ALL_MODELS.get(model_name)
    if not info:
        raise ValueError(f"Model '{model_name}' not in config")
    repo_id = info["repo_id"]

    # Use ComfyUI's multi-path system if available
    llm_paths = folder_paths.get_folder_paths("LLM") if "LLM" in folder_paths.folder_names_and_paths else []
    if llm_paths:
        models_dir = Path(llm_paths[0]) / "Qwen-VL"
    else:
        # Fallback to default behavior
        models_dir = Path(folder_paths.models_dir) / "LLM" / "Qwen-VL"

    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / repo_id.split("/")[-1]

    # ✅ If already downloaded (has weights), use local without calling snapshot_download
    if target.exists() and target.is_dir():
        if any(target.glob("*.safetensors")) or any(target.glob("*.bin")):
            return str(target)

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", ".git*"],
    )
    return str(target)

def enforce_memory(model_name, quantization, device_info):
    info = HF_ALL_MODELS.get(model_name, {})
    requirements = info.get("vram_requirement", {})
    mapping = {
        Quantization.Q4: requirements.get("4bit", 0),
        Quantization.Q8: requirements.get("8bit", 0),
        Quantization.FP16: requirements.get("full", 0),
    }
    needed = mapping.get(quantization, 0)
    if not needed:
        return quantization
    if device_info["recommended_device"] in {"cpu", "mps"}:
        needed *= 1.5
        available = device_info["system_memory"]["available"]
    else:
        available = device_info["gpu"]["free_memory"]
    if needed * 1.2 > available:
        if quantization == Quantization.FP16:
            print("[QwenVL] Auto-switch to 8-bit due to VRAM pressure")
            return Quantization.Q8
        if quantization == Quantization.Q8:
            print("[QwenVL] Auto-switch to 4-bit due to VRAM pressure")
            return Quantization.Q4
        raise RuntimeError("Insufficient memory for 4-bit mode")
    return quantization

def is_fp8_model(model_name: str) -> bool:
    """Check if model name indicates it's a pre-quantized FP8 model."""
    fp8_indicators = ["-fp8", "_fp8", "-FP8", "_FP8"]
    return any(indicator in model_name for indicator in fp8_indicators)


def is_nvfp4_model(model_name: str) -> bool:
    """Check if model config identifies a compressed-tensors NVFP4 checkpoint."""
    info = HF_ALL_MODELS.get(model_name, {})
    quantization_format = str(info.get("quantization_format", "")).lower()
    repo_id = str(info.get("repo_id", "")).lower()
    model_key = str(model_name or "").lower()
    return quantization_format == "nvfp4" or "nvfp4" in repo_id or "nvfp4" in model_key


def is_prequantized_fp8_model(model_name: str) -> bool:
    """Existing quantized entries are FP8 unless a more specific format is declared."""
    info = HF_ALL_MODELS.get(model_name, {})
    return is_fp8_model(model_name) or (bool(info.get("quantized")) and not is_nvfp4_model(model_name))


def cuda_device_inventory() -> list[str]:
    if not torch.cuda.is_available():
        return []
    return [f"cuda:{idx}={torch.cuda.get_device_name(idx)}" for idx in range(torch.cuda.device_count())]


def require_nvfp4_cuda_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("[QwenVL] NVFP4 model requires CUDA; no CUDA device is available.")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("[QwenVL] NVFP4 model requires cuda:0; no CUDA devices were detected.")

    devices = cuda_device_inventory()
    print(f"[QwenVL] NVFP4 CUDA devices: {', '.join(devices)}")
    target_name = torch.cuda.get_device_name(0)
    if "5090" not in target_name:
        raise RuntimeError(
            "[QwenVL] NVFP4 model is pinned to cuda:0 / RTX 5090, "
            f"but cuda:0 is '{target_name}'. Set CUDA_DEVICE_ORDER=PCI_BUS_ID before launch "
            "or adjust the device guard."
        )
    torch.cuda.set_device(0)
    return "cuda:0"


def nvfp4_cuda0_memory_budget_gib() -> int:
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    reserve_bytes = max(8 * BYTES_PER_GIB, int(total_bytes * 0.20))
    usable_bytes = min(free_bytes - reserve_bytes, total_bytes - reserve_bytes)
    if usable_bytes < 8 * BYTES_PER_GIB:
        raise RuntimeError(
            "[QwenVL] NVFP4 model needs at least 8 GiB free on cuda:0 after reserve; "
            f"free={free_bytes / BYTES_PER_GIB:.1f} GiB, reserve={reserve_bytes / BYTES_PER_GIB:.1f} GiB."
        )
    return max(8, int(usable_bytes // BYTES_PER_GIB))


def nvfp4_transformers_load_kwargs() -> dict:
    target_device = require_nvfp4_cuda_device()
    cuda0_budget_gib = nvfp4_cuda0_memory_budget_gib()
    print(
        "[QwenVL] NVFP4 direct cuda:0 load: "
        f"{cuda0_budget_gib} GiB usable after reserve; CPU/disk offload disabled"
    )
    return {
        "target_device": target_device,
        "device_map": {"": target_device},
    }


def nvfp4_quantization_config(model_path: str):
    from transformers.utils.quantization_config import CompressedTensorsConfig

    with open(Path(model_path) / "config.json", "r", encoding="utf-8") as fh:
        raw_config = json.load(fh)
    raw_quantization = raw_config.get("quantization_config")
    if not raw_quantization:
        raise RuntimeError(f"[QwenVL] Missing quantization_config in NVFP4 model: {model_path}")
    config = CompressedTensorsConfig.from_dict(raw_quantization, run_compressed=True)
    config.run_compressed = True
    return config


def device_to_string(device) -> str:
    if isinstance(device, torch.device):
        return str(device)
    return str(device)


def cuda_index_from_device(device):
    if isinstance(device, int):
        return device
    try:
        parsed = torch.device(device)
    except Exception:
        return None
    if parsed.type != "cuda":
        return None
    return 0 if parsed.index is None else parsed.index


def summarize_device_map(model) -> dict[str, str]:
    device_map = getattr(model, "hf_device_map", None) or {}
    return {str(name): device_to_string(device) for name, device in device_map.items()}


def summarize_parameter_devices(model, sample_limit: int = 20) -> dict:
    devices = set()
    samples = []
    for name, parameter in model.named_parameters():
        device = device_to_string(parameter.device)
        devices.add(device)
        if len(samples) < sample_limit:
            samples.append({"name": name, "device": device})
    return {"devices": sorted(devices), "sample": samples}


def validate_nvfp4_device_placement(model) -> None:
    blocked = []
    for name, device in summarize_device_map(model).items():
        if cuda_index_from_device(device) == 1:
            blocked.append(f"device_map:{name}->{device}")

    for name, parameter in model.named_parameters():
        if parameter.device.type == "cuda" and parameter.device.index == 1:
            blocked.append(f"parameter:{name}->{parameter.device}")
            if len(blocked) >= 20:
                break

    if blocked:
        raise RuntimeError(
            "[QwenVL] NVFP4 model attempted to use cuda:1 / RTX 4090. "
            f"Blocked placements: {', '.join(blocked)}"
        )


def nvfp4_linear_forward_decompressed(self, input):
    from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
    from compressed_tensors.utils import get_direct_state_dict

    state_dict = get_direct_state_dict(self)
    decompressed = NVFP4PackedCompressor.decompress(state_dict, self.quantization_scheme)
    weight = decompressed["weight"].to(device=input.device, dtype=input.dtype)
    bias = decompressed.get("bias")
    if bias is not None:
        bias = bias.to(device=input.device, dtype=input.dtype)
    return torch.nn.functional.linear(input, weight, bias)


def nvfp4_tensor_scale_from_module_or_input(state_dict: dict, input_2d: torch.Tensor) -> torch.Tensor:
    force_dynamic = os.environ.get("QWENVL_NVFP4_DYNAMIC_INPUT_SCALE") == "1"
    input_global_scale = state_dict.get("input_global_scale")
    if input_global_scale is not None and not force_dynamic:
        input_global_scale = input_global_scale.to(
            device=input_2d.device, dtype=torch.float32, non_blocking=True
        ).reshape(1)
        return torch.reciprocal(input_global_scale)

    max_abs = input_2d.detach().abs().amax().to(dtype=torch.float32)
    scale = max_abs / (NVFP4_F8_E4M3_MAX * NVFP4_F4_E2M1_MAX)
    return torch.clamp(scale, min=torch.finfo(torch.float32).tiny).reshape(1)


def nvfp4_linear_forward_kernel(self, input):
    import comfy_kitchen as ck
    from compressed_tensors.utils import get_direct_state_dict

    if input.device.type != "cuda":
        raise RuntimeError("comfy_kitchen NVFP4 kernel requires CUDA input")
    if torch.cuda.get_device_capability(input.device)[0] < 10:
        raise RuntimeError("comfy_kitchen NVFP4 kernel requires Blackwell/SM100 or newer")

    state_dict = get_direct_state_dict(self)
    weight_packed = state_dict["weight_packed"].to(
        device=input.device, dtype=torch.uint8, non_blocking=True
    ).contiguous()
    weight_scale = state_dict["weight_scale"].to(device=input.device, non_blocking=True).contiguous()
    weight_scale = ck.to_blocked(weight_scale, flatten=False).contiguous()
    weight_global_scale = state_dict["weight_global_scale"].to(
        device=input.device, dtype=torch.float32, non_blocking=True
    ).reshape(1)
    weight_tensor_scale = torch.reciprocal(weight_global_scale)

    in_features = getattr(self, "in_features", weight_packed.shape[1] * 2)
    out_features = getattr(self, "out_features", weight_packed.shape[0])
    input_shape = input.shape
    input_2d = input.reshape(-1, input_shape[-1])
    if input_2d.shape[-1] != in_features:
        raise RuntimeError(
            f"NVFP4 Linear input mismatch: input has {input_2d.shape[-1]} features, "
            f"module expects {in_features}."
        )

    out_dtype = input.dtype if input.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    input_2d = input_2d.to(dtype=out_dtype).contiguous()
    input_tensor_scale = nvfp4_tensor_scale_from_module_or_input(state_dict, input_2d)
    input_packed, input_block_scale = ck.quantize_nvfp4(
        input_2d,
        input_tensor_scale,
        pad_16x=True,
        hi_first=False,
    )

    bias = state_dict.get("bias")
    if bias is not None:
        bias = bias.to(device=input.device, dtype=out_dtype, non_blocking=True)

    output = ck.scaled_mm_nvfp4(
        input_packed,
        weight_packed,
        tensor_scale_a=input_tensor_scale,
        tensor_scale_b=weight_tensor_scale,
        block_scale_a=input_block_scale,
        block_scale_b=weight_scale,
        bias=bias,
        out_dtype=out_dtype,
    )
    output = output[: input_2d.shape[0], :out_features]
    return output.reshape(*input_shape[:-1], out_features)


def nvfp4_linear_forward_dequant_kernel(self, input):
    import comfy_kitchen as ck
    from compressed_tensors.utils import get_direct_state_dict

    if input.device.type != "cuda":
        raise RuntimeError("comfy_kitchen NVFP4 dequant kernel requires CUDA input")

    state_dict = get_direct_state_dict(self)
    weight_packed = state_dict["weight_packed"].to(
        device=input.device, dtype=torch.uint8, non_blocking=True
    ).contiguous()
    weight_scale = state_dict["weight_scale"].to(device=input.device, non_blocking=True).contiguous()
    weight_scale = ck.to_blocked(weight_scale, flatten=False).contiguous()
    weight_global_scale = state_dict["weight_global_scale"].to(
        device=input.device, dtype=torch.float32, non_blocking=True
    ).reshape(1)
    weight_tensor_scale = torch.reciprocal(weight_global_scale)

    in_features = getattr(self, "in_features", weight_packed.shape[1] * 2)
    out_features = getattr(self, "out_features", weight_packed.shape[0])
    out_dtype = input.dtype if input.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

    weight = ck.dequantize_nvfp4(
        weight_packed,
        per_tensor_scale=weight_tensor_scale,
        block_scales=weight_scale,
        output_type=out_dtype,
        hi_first=False,
    )[:out_features, :in_features]

    bias = state_dict.get("bias")
    if bias is not None:
        bias = bias.to(device=input.device, dtype=out_dtype, non_blocking=True)

    return torch.nn.functional.linear(input.to(dtype=out_dtype), weight, bias)


def nvfp4_linear_forward(self, input):
    try:
        if os.environ.get("QWENVL_NVFP4_W4A4_KERNEL") == "1":
            return nvfp4_linear_forward_kernel(self, input)
        return nvfp4_linear_forward_dequant_kernel(self, input)
    except Exception as exc:
        if not getattr(self, "_nvfp4_kernel_warned", False):
            print(f"[QwenVL] NVFP4 kernel fallback for {self.__class__.__name__}: {exc}")
            self._nvfp4_kernel_warned = True
        return nvfp4_linear_forward_decompressed(self, input)


def enable_nvfp4_linear_forward(model) -> int:
    if hasattr(model, "ct_decompress_hook"):
        model.ct_decompress_hook.remove()
        delattr(model, "ct_decompress_hook")

    patched = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and hasattr(module, "weight_packed"):
            bound_forward = nvfp4_linear_forward.__get__(module, module.__class__)
            if hasattr(module, "_old_forward"):
                module._old_forward = bound_forward
            else:
                module.forward = bound_forward
            patched += 1
    print(f"[QwenVL] NVFP4: patched {patched} compressed Linear forwards for CUDA dequant execution")
    return patched


@contextmanager
def nvfp4_torchdynamo_disabled(enabled: bool):
    if not enabled:
        yield
        return

    env_names = ("TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE")
    previous_env = {name: os.environ.get(name) for name in env_names}
    previous_dynamo_disable = None
    dynamo_config = None
    try:
        import torch._dynamo as torch_dynamo

        dynamo_config = torch_dynamo.config
        previous_dynamo_disable = getattr(dynamo_config, "disable", None)
    except Exception:
        dynamo_config = None

    try:
        for name in env_names:
            os.environ[name] = "1"
        if dynamo_config is not None and previous_dynamo_disable is not None:
            dynamo_config.disable = True
        print("[QwenVL] NVFP4: TorchDynamo and torch.compile disabled for this run")
        yield
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if dynamo_config is not None and previous_dynamo_disable is not None:
            dynamo_config.disable = previous_dynamo_disable


def quantization_config(model_name, quantization):
    """Returns (quant_config, dtype, is_prequantized).
    
    Pre-quantized models are loaded without BitsAndBytes.
    """
    info = HF_ALL_MODELS.get(model_name, {})
    if info.get("quantized") or is_fp8_model(model_name) or is_nvfp4_model(model_name):
        return None, None, True
    if quantization == Quantization.Q4:
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return cfg, None, False
    if quantization == Quantization.Q8:
        return BitsAndBytesConfig(load_in_8bit=True), None, False
    return None, torch.float16 if torch.cuda.is_available() else torch.float32, False

class QwenVLBase:
    def __init__(self):
        self.device_info = get_device_info()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_signature = None
        self.last_model_path = None
        self.last_device_map = {}
        self.last_parameter_devices = {}
        self.current_is_nvfp4 = False
        print(f"[QwenVL] Node on {self.device_info['device_type']}")

    def clear(self):
        """Clear model from memory and free VRAM."""
        if self.model is not None:
            # Move model to CPU first to free GPU memory
            if not self.current_is_nvfp4:
                try:
                    self.model = self.model.cpu()
                except:
                    pass
            self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_signature = None
        self.current_is_nvfp4 = False
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def load_model(
        self,
        model_name,
        quant_value,
        attention_mode,
        use_compile,
        device_choice,
        keep_model_loaded,
    ):
        requested_quant = Quantization.from_value(quant_value)
        is_prequantized_nvfp4 = is_nvfp4_model(model_name)
        is_prequantized_fp8 = is_prequantized_fp8_model(model_name)
        is_prequantized = is_prequantized_nvfp4 or is_prequantized_fp8

        if is_prequantized:
            quant = requested_quant
            if requested_quant != Quantization.FP16:
                print(f"[QwenVL] {model_name} is pre-quantized; ignoring BitsAndBytes quantization selector")
        else:
            quant = enforce_memory(model_name, requested_quant, self.device_info)
        
        # Check if BitsAndBytes quantization is being used
        is_bnb_quantization = not is_prequantized and quant in [Quantization.Q4, Quantization.Q8]
        
        # Determine if we need to force SDPA (for pre-quantized or BitsAndBytes models)
        force_sdpa = is_prequantized or is_bnb_quantization
        
        # Resolve attention mode with force_sdpa flag
        attn_impl = resolve_attention_mode(attention_mode, force_sdpa=force_sdpa)
        
        # Additional info messages for forced SDPA
        if force_sdpa and attention_mode in ["auto", "sage", "flash_attention_2"]:
            if is_prequantized_nvfp4:
                print("[QwenVL] NVFP4 compressed-tensors model detected - forcing SDPA attention")
            elif is_prequantized_fp8:
                print("[QwenVL] FP8 model detected - forcing SDPA attention")
            elif is_bnb_quantization:
                print("[QwenVL] BitsAndBytes quantization detected - forcing SDPA attention")
        
        print(f"[QwenVL] Attention backend selected: {attn_impl}")
        
        device_requested = self.device_info["recommended_device"] if device_choice == "auto" else device_choice
        device = normalize_device_choice(device_requested)
        effective_use_compile = bool(use_compile) and not is_prequantized_nvfp4
        if use_compile and is_prequantized_nvfp4:
            print("[QwenVL] NVFP4: torch.compile disabled for this model")
        signature = (model_name, quant.value, attn_impl, device, effective_use_compile)
        
        # Check if we need to reload (model, quantization, or attention changed)
        if keep_model_loaded and self.model is not None and self.current_signature == signature:
            return
        
        # Clear model and VRAM before loading new configuration
        if self.model is not None:
            print("[QwenVL] Clearing previous model from memory before loading new configuration...")
        self.clear()
        self.last_device_map = {}
        self.last_parameter_devices = {}
        model_path = ensure_model(model_name)
        self.last_model_path = model_path
        quant_config, dtype, _ = quantization_config(model_name, quant)
        
        # Handle attention mode for loading
        # SageAttention requires loading with SDPA first, then patching
        actual_attn_impl = attn_impl
        if attn_impl == "sage":
            actual_attn_impl = "sdpa"
        
        # Build load kwargs
        load_kwargs = {
            "attn_implementation": actual_attn_impl,
            "use_safetensors": True,
        }
        
        if is_prequantized_nvfp4:
            nvfp4_load_settings = nvfp4_transformers_load_kwargs()
            target_device = nvfp4_load_settings.pop("target_device")
            if device not in {"auto", "cuda", target_device}:
                print(f"[QwenVL] NVFP4: ignoring requested device '{device}' and using {target_device}")

            load_kwargs.update(nvfp4_load_settings)
            load_kwargs["dtype"] = "auto"
            load_kwargs["quantization_config"] = nvfp4_quantization_config(model_path)

            print(f"[QwenVL] Loading NVFP4 compressed-tensors model directly on {target_device}...")
            try:
                self.current_is_nvfp4 = True
                self.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs).eval()
                enable_nvfp4_linear_forward(self.model)
                validate_nvfp4_device_placement(self.model)
                self.last_device_map = summarize_device_map(self.model)
                self.last_parameter_devices = summarize_parameter_devices(self.model)
                print(f"[QwenVL] NVFP4 model loaded on {target_device}")
            except Exception:
                self.clear()
                raise
        elif is_prequantized_fp8:
            # For pre-quantized FP8 models: load directly to device, don't use device_map
            # This prevents meta tensor issues
            
            # Determine target device based on user choice and availability
            if device == "auto":
                # Auto-select best available device
                if torch.cuda.is_available():
                    target_device = "cuda:0"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    target_device = "mps"
                else:
                    target_device = "cpu"
            else:
                # Use user-specified device
                target_device = device
            
            # For FP8 models, we need to disable accelerate's device_map and load manually
            # to avoid meta tensor issues
            load_kwargs["device_map"] = None
            load_kwargs["torch_dtype"] = "auto"  # Let transformers detect FP8 dtype from config
            
            print(f"[QwenVL] Loading FP8 model to {target_device}...")
            
            # Load model on CPU first (without device_map to avoid meta tensors)
            self.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs)
            
            # Check if model has meta tensors and materialize them
            has_meta = any(param.device.type == "meta" for param in self.model.parameters())
            if has_meta:
                print("[QwenVL] Model has meta tensors, materializing...")
                # Materialize meta tensors on CPU first
                self.model = self.model.to_empty(device="cpu")
                # Now load the state dict properly using HuggingFace's built-in method
                # This handles both single-file and sharded checkpoints correctly
                print(f"[QwenVL] Loading weights from {model_path}")
                try:
                    # Use HuggingFace's official sharded checkpoint loading
                    from transformers.modeling_utils import load_sharded_checkpoint
                    
                    print(f"[QwenVL] Loading weights from {model_path}")
                    
                    # Check if this is a sharded checkpoint
                    index_file = os.path.join(model_path, "model.safetensors.index.json")
                    if os.path.exists(index_file):
                        # Sharded checkpoint - use HF's official loading function
                        print("[QwenVL] Detected sharded checkpoint, using HF load_sharded_checkpoint...")
                        load_sharded_checkpoint(self.model, model_path, strict=True)
                        print("[QwenVL] All shards loaded successfully")
                    else:
                        # Single-file checkpoint - use HF's standard loading
                        from transformers.modeling_utils import load_state_dict
                        from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
                        
                        if os.path.exists(os.path.join(model_path, SAFE_WEIGHTS_NAME)):
                            state_dict_path = os.path.join(model_path, SAFE_WEIGHTS_NAME)
                        elif os.path.exists(os.path.join(model_path, WEIGHTS_NAME)):
                            state_dict_path = os.path.join(model_path, WEIGHTS_NAME)
                        else:
                            raise RuntimeError(f"Could not find model weights in {model_path}")
                        
                        print(f"[QwenVL] Loading weights from {state_dict_path}")
                        state_dict = load_state_dict(state_dict_path)
                        
                        # Load state dict into the model - use strict=True for FP8 models
                        # to ensure all scale factors are loaded
                        try:
                            self.model.load_state_dict(state_dict, strict=True)
                            print("[QwenVL] All weights loaded successfully")
                        except RuntimeError as e:
                            # If strict loading fails, try non-strict and warn
                            print(f"[QwenVL] Strict loading failed: {e}")
                            print("[QwenVL] Attempting non-strict loading...")
                            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                            if missing_keys:
                                print(f"[QwenVL] Warning: Missing keys: {missing_keys}")
                            if unexpected_keys:
                                print(f"[QwenVL] Info: Unexpected keys (not loaded): {unexpected_keys}")
                except Exception as e:
                    print(f"[QwenVL] Error loading weights: {e}")
                    raise
            
            # Now move to target device
            print(f"[QwenVL] Moving FP8 model to {target_device}")
            self.model = self.model.to(target_device)
            self.model.eval()
            print(f"[QwenVL] FP8 model loaded on {target_device}")
        else:
            # For regular models: use device_map and dtype
            load_kwargs["device_map"] = device if device != "auto" else "auto"
            if dtype:
                load_kwargs["dtype"] = dtype
            
            if quant_config:
                load_kwargs["quantization_config"] = quant_config
            
            # Show appropriate attention info in loading message
            if attn_impl == "sage":
                print(f"[QwenVL] Loading {model_name} ({quant.value}, base=sdpa, will_patch=sage)")
            else:
                print(f"[QwenVL] Loading {model_name} ({quant.value}, attn={actual_attn_impl})")
            
            self.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs).eval()
        
        # Apply SageAttention patching if selected
        if attn_impl == "sage":
            try:
                set_sage_attention(self.model)
                print("[QwenVL] SageAttention enabled")
            except Exception as exc:
                print(f"[QwenVL] SageAttention patching failed: {exc}")
        
        self.model.config.use_cache = True
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = True
        if effective_use_compile and device.startswith("cuda") and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[QwenVL] torch.compile enabled")
            except Exception as exc:
                print(f"[QwenVL] torch.compile skipped: {exc}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.current_signature = signature

    @staticmethod
    def tensor_to_pil(tensor):
        if tensor is None:
            return None
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(array)

    @torch.no_grad()
    def generate(
        self,
        prompt_text,
        image,
        video,
        frame_count,
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
    ):
        conversation = [{"role": "user", "content": []}]
        if image is not None:
            conversation[0]["content"].append({"type": "image", "image": self.tensor_to_pil(image)})
        if video is not None:
            frames = [self.tensor_to_pil(frame) for frame in video]
            if len(frames) > frame_count:
                idx = np.linspace(0, len(frames) - 1, frame_count, dtype=int)
                frames = [frames[i] for i in idx]
            if frames:
                conversation[0]["content"].append({"type": "video", "video": frames})
        conversation[0]["content"].append({"type": "text", "text": prompt_text})
        chat = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        images = [item["image"] for item in conversation[0]["content"] if item["type"] == "image"]
        video_frames = [frame for item in conversation[0]["content"] if item["type"] == "video" for frame in item["video"]]
        videos = [video_frames] if video_frames else None
        processed = self.processor(text=chat, images=images or None, videos=videos, return_tensors="pt")
        model_device = torch.device("cuda:0") if self.current_is_nvfp4 else next(self.model.parameters()).device
        model_inputs = {
            key: value.to(model_device) if torch.is_tensor(value) else value
            for key, value in processed.items()
        }
        stop_tokens = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, "eot_id") and self.tokenizer.eot_id is not None:
            stop_tokens.append(self.tokenizer.eot_id)
        kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "num_beams": num_beams,
            "eos_token_id": stop_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if num_beams == 1:
            kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
        else:
            kwargs["do_sample"] = False
        outputs = self.model.generate(**model_inputs, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        input_len = model_inputs["input_ids"].shape[-1]
        text = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
        return text.strip()

    def run(self, model_name, quantization, preset_prompt, custom_prompt, image, video, frame_count, max_tokens, temperature, top_p, num_beams, repetition_penalty, seed, keep_model_loaded, attention_mode, use_torch_compile, device):
        # Create progress bar with 3 stages: setup, model loading, generation
        pbar = ProgressBar(3)
        
        torch.manual_seed(seed)
        prompt = SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip()
        
        pbar.update_absolute(1, 3, None)
        
        with nvfp4_torchdynamo_disabled(is_nvfp4_model(model_name)):
            self.load_model(
                model_name,
                quantization,
                attention_mode,
                use_torch_compile,
                device,
                keep_model_loaded,
            )
            
            pbar.update_absolute(2, 3, None)
            
            try:
                text = self.generate(
                    prompt,
                    image,
                    video,
                    frame_count,
                    max_tokens,
                    temperature,
                    top_p,
                    num_beams,
                    repetition_penalty,
                )
                
                pbar.update_absolute(3, 3, None)
                
                return (text,)
            finally:
                if not keep_model_loaded:
                    self.clear()

class AILab_QwenVL(QwenVLBase):
    @classmethod
    def INPUT_TYPES(cls):
        models = list(HF_VL_MODELS.keys())
        default_model = models[0] if models else "Qwen3-VL-4B-Instruct"
        prompts = PRESET_PROMPTS or ["Describe this image in detail."]
        preferred_prompt = "🖼️ Detailed Description"
        default_prompt = preferred_prompt if preferred_prompt in prompts else prompts[0]
        return {
            "required": {
                "model_name": (models, {"default": default_model, "tooltip": TOOLTIPS["model_name"]}),
                "quantization": (Quantization.get_values(), {"default": Quantization.FP16.value, "tooltip": TOOLTIPS["quantization"]}),
                "attention_mode": (ATTENTION_MODES, {"default": "auto", "tooltip": TOOLTIPS["attention_mode"]}),
                "preset_prompt": (prompts, {"default": default_prompt, "tooltip": TOOLTIPS["preset_prompt"]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": TOOLTIPS["custom_prompt"]}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 2048, "tooltip": TOOLTIPS["max_tokens"]}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "tooltip": TOOLTIPS["keep_model_loaded"]}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1, "tooltip": TOOLTIPS["seed"]}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "🧪AILab/QwenVL"

    def process(self, model_name, quantization, preset_prompt, custom_prompt, attention_mode, max_tokens, keep_model_loaded, seed, image=None, video=None):
        return self.run(model_name, quantization, preset_prompt, custom_prompt, image, video, 16, max_tokens, 0.6, 0.9, 1, 1.2, seed, keep_model_loaded, attention_mode, False, "auto")

class AILab_QwenVL_Advanced(QwenVLBase):
    @classmethod
    def INPUT_TYPES(cls):
        models = list(HF_VL_MODELS.keys())
        default_model = models[0] if models else "Qwen3-VL-4B-Instruct"
        prompts = PRESET_PROMPTS or ["Describe this image in detail."]
        preferred_prompt = "🖼️ Detailed Description"
        default_prompt = preferred_prompt if preferred_prompt in prompts else prompts[0]

        num_gpus = torch.cuda.device_count()
        gpu_list = [f"cuda:{i}" for i in range(num_gpus)]
        device_options = ["auto", "cpu", "mps"] + gpu_list

        return {
            "required": {
                "model_name": (models, {"default": default_model, "tooltip": TOOLTIPS["model_name"]}),
                "quantization": (Quantization.get_values(), {"default": Quantization.FP16.value, "tooltip": TOOLTIPS["quantization"]}),
                "attention_mode": (ATTENTION_MODES, {"default": "auto", "tooltip": TOOLTIPS["attention_mode"]}),
                "use_torch_compile": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["use_torch_compile"]}),
                "device": (device_options, {"default": "auto", "tooltip": TOOLTIPS["device"]}),
                "preset_prompt": (prompts, {"default": default_prompt, "tooltip": TOOLTIPS["preset_prompt"]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": TOOLTIPS["custom_prompt"]}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 4096, "tooltip": TOOLTIPS["max_tokens"]}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "tooltip": TOOLTIPS["temperature"]}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "tooltip": TOOLTIPS["top_p"]}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": TOOLTIPS["num_beams"]}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 2.0, "tooltip": TOOLTIPS["repetition_penalty"]}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64, "tooltip": TOOLTIPS["frame_count"]}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "tooltip": TOOLTIPS["keep_model_loaded"]}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1, "tooltip": TOOLTIPS["seed"]}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "🧪AILab/QwenVL"

    def process(self, model_name, quantization, attention_mode, use_torch_compile, device, preset_prompt, custom_prompt, max_tokens, temperature, top_p, num_beams, repetition_penalty, frame_count, keep_model_loaded, seed, image=None, video=None):
        return self.run(model_name, quantization, preset_prompt, custom_prompt, image, video, frame_count, max_tokens, temperature, top_p, num_beams, repetition_penalty, seed, keep_model_loaded, attention_mode, use_torch_compile, device)

NODE_CLASS_MAPPINGS = {
    "AILab_QwenVL": AILab_QwenVL,
    "AILab_QwenVL_Advanced": AILab_QwenVL_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_QwenVL": "QwenVL",
    "AILab_QwenVL_Advanced": "QwenVL (Advanced)",
}
