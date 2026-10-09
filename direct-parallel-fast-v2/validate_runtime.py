from __future__ import annotations

import json
import hashlib
import os
import platform
import subprocess
import time
from pathlib import Path


ROOT = Path(os.getenv("MASHHAD_LTX_ROOT", "/workspace/LTX25-DIRECT"))
STATE_DIR = ROOT / ".mashhad"
RECEIPT = STATE_DIR / "runtime-compatibility.json"
MODEL_RECEIPT = STATE_DIR / "models-bf16.receipt.json"
EXPECTED_MODEL_NAMES = (
    "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "ltx-2.5-video-vae-bf16.safetensors",
    "ltx-2.5-audio-vae-bf16.safetensors",
    "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
)


def driver_version() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).splitlines()[0].strip()
    except Exception:
        return None


def edge_fingerprint(path: Path) -> str:
    edge_bytes = 1024 * 1024
    digest = hashlib.blake2b(digest_size=20)
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        digest.update(handle.read(edge_bytes))
        if file_size > edge_bytes:
            handle.seek(max(0, file_size - edge_bytes))
            digest.update(handle.read(edge_bytes))
    digest.update(str(file_size).encode("ascii"))
    return digest.hexdigest()


def validate_model_receipt(model_dir: Path) -> None:
    from safetensors import safe_open

    try:
        receipt = json.loads(MODEL_RECEIPT.read_text(encoding="utf-8"))
        records = receipt["files"]
    except (OSError, KeyError, json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError("Trusted BF16 model receipt is missing or invalid") from exc
    if not isinstance(records, dict) or len(records) != len(EXPECTED_MODEL_NAMES):
        raise RuntimeError("Trusted BF16 model receipt does not describe exactly five files")
    by_name = {
        str(item.get("file_name")): item
        for item in records.values()
        if isinstance(item, dict) and item.get("file_name")
    }
    if set(by_name) != set(EXPECTED_MODEL_NAMES):
        raise RuntimeError("Trusted BF16 model receipt names do not match this release")
    weight_files = {
        path.name
        for path in model_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".safetensors", ".ckpt", ".pt", ".bin"}
    }
    if weight_files != set(EXPECTED_MODEL_NAMES):
        raise RuntimeError(f"Model directory must contain only the five BF16 weights; found: {sorted(weight_files)}")
    for name, record in by_name.items():
        path = model_dir / name
        stat = path.stat()
        if stat.st_size != int(record.get("size") or -1):
            raise RuntimeError(f"Model size changed: {name}")
        # Normal resume trusts an unchanged receipt. If metadata changed, perform
        # cheap edge + safetensors validation, never a full multi-GB SHA scan.
        if stat.st_mtime_ns != int(record.get("mtime_ns") or -1):
            if edge_fingerprint(path) != record.get("edge_fingerprint"):
                raise RuntimeError(f"Model edge fingerprint changed: {name}")
            try:
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    if not list(handle.keys()):
                        raise RuntimeError(f"Empty safetensors file: {name}")
            except Exception as exc:
                raise RuntimeError(f"Safetensors validation failed: {name}") from exc


def main() -> None:
    import natten
    import torch
    import ltx_core
    import ltx_pipelines

    if not torch.__version__.startswith("2.9.1"):
        raise RuntimeError(f"Expected base torch 2.9.1, got {torch.__version__}")
    if not (torch.version.cuda or "").startswith("12.8"):
        raise RuntimeError(f"Expected CUDA 12.8 torch build, got {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not pass PyTorch BF16 compatibility")
    if not bool(getattr(natten, "HAS_LIBNATTEN", False)):
        raise RuntimeError("NATTEN imported, but its compiled CUDA kernel library is unavailable")

    device = torch.device("cuda:0")
    left = torch.randn((64, 64), device=device, dtype=torch.bfloat16)
    right = torch.randn((64, 64), device=device, dtype=torch.bfloat16)
    product = left @ right
    torch.cuda.synchronize(device)
    if not torch.isfinite(product).all().item():
        raise RuntimeError("BF16 CUDA smoke test produced non-finite values")
    from natten.functional import na3d

    query = torch.randn((1, 2, 4, 4, 4, 16), device=device, dtype=torch.bfloat16)
    attention = na3d(query, query, query, kernel_size=(2, 3, 3), scale=16**-0.5)
    torch.cuda.synchronize(device)
    if attention.shape != query.shape or not torch.isfinite(attention).all().item():
        raise RuntimeError("NATTEN CUDA smoke test failed on this GPU")
    del left, right, product, query, attention
    torch.cuda.empty_cache()

    model_dir = Path(os.getenv("LTX_MODELS_DIR", str(ROOT / "models")))
    missing = [name for name in EXPECTED_MODEL_NAMES if not (model_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing Direct BF16 model files: {missing}")
    validate_model_receipt(model_dir)

    properties = torch.cuda.get_device_properties(0)
    result = {
        "schema": 1,
        "validated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "natten": getattr(natten, "__version__", "unknown"),
        "ltx_core": getattr(ltx_core, "__version__", "installed"),
        "ltx_pipelines": getattr(ltx_pipelines, "__version__", "installed"),
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "vram_bytes": properties.total_memory,
        "driver": driver_version(),
        "bf16": True,
        "int8_convrot": "not-tested-not-enabled",
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = RECEIPT.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, RECEIPT)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
