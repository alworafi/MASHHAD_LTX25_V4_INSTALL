from __future__ import annotations

import base64
import binascii
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT = Path(os.getenv("MASHHAD_LTX_ROOT", "/workspace/LTX25-DIRECT"))
MODEL_DIR = Path(os.getenv("LTX_MODELS_DIR", str(ROOT / "models")))
INPUT_DIR = Path(os.getenv("WORKER_INPUT_DIR", str(ROOT / "inputs")))
OUTPUT_DIR = Path(os.getenv("WORKER_OUTPUT_DIR", str(ROOT / "outputs")))
STATE_DIR = ROOT / ".mashhad"

MODEL_FILES = {
    "transformer": MODEL_DIR / "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoder": MODEL_DIR / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video_vae": MODEL_DIR / "ltx-2.5-video-vae-bf16.safetensors",
    "audio_vae": MODEL_DIR / "ltx-2.5-audio-vae-bf16.safetensors",
    "spatial_upsampler": MODEL_DIR / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
}


class Asset(BaseModel):
    filename: str
    content_type: str = "application/octet-stream"
    base64: str


class GenerateRequest(BaseModel):
    job_id: str = Field(pattern=r"^[0-9a-f-]{36}$")
    prompt: str = Field(min_length=3, max_length=4000)
    seed: int = Field(ge=0, le=2_147_483_647)
    resolution: str = "1024x576"
    width: int | None = None
    height: int | None = None
    fps: float = Field(default=24.0, ge=1.0, le=60.0)
    num_frames: int = Field(default=121, ge=17, le=481)
    duration: float | None = Field(default=None, ge=0.5, le=30.0)
    audio_enabled: bool = True
    first_frame: Asset | None = None
    last_frame: Asset | None = None
    reference_image: Asset | None = None
    first_frame_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    last_frame_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    reference_strength: float = Field(default=0.45, ge=0.0, le=1.0)
    lora: str | None = None
    lora_strength: float | None = None


class PipelineManager:
    """Own exactly one official BF16 pipeline for the life of this Pod process."""

    def __init__(self) -> None:
        self.pipeline: Any | None = None
        self.model_state = "unloaded"
        self.model_error: str | None = None
        self.loaded_at: float | None = None
        self.last_generation_completed_at: float | None = None
        self.offload_mode: str | None = None
        self.system_ram_gb = _system_ram_gb()
        self.cache_weights_in_ram = False
        self.warm_allocator = False
        self._generation_verified = False
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._queued = 0
        self._active_job: str | None = None
        self._cancel_requested: set[str] = set()

    def health(self) -> dict[str, Any]:
        missing = [str(path) for path in MODEL_FILES.values() if not path.is_file()]
        available = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if available else None
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if available else None
        with self._state_lock:
            pipeline_ready = self.pipeline is not None and self.model_state not in {"failed", "unloaded"}
            allocated_gb = round(torch.cuda.memory_allocated(0) / 1024**3, 2) if available else 0.0
            model_loaded_to_gpu = bool(self._active_job and allocated_gb >= 1.0)
            return {
                "service_ready": True,
                "pipeline_ready": pipeline_ready,
                # A lazy pipeline object is not proof that all official model
                # components have loaded and executed successfully.
                "model_ready": self._generation_verified,
                "model_loaded_to_gpu": model_loaded_to_gpu,
                "model_state": self.model_state,
                "model_error": self.model_error,
                "pipeline_initialized_at_epoch": self.loaded_at,
                "last_generation_completed_at_epoch": self.last_generation_completed_at,
                "active_job": self._active_job,
                "queue_depth": self._queued,
                "concurrency": 1,
                "pipeline": "ltx_pipelines.distilled.DistilledPipeline",
                "precision": "bf16",
                "quantization": None,
                "int8_convrot": "not-tested-not-enabled",
                "offload_mode": self.offload_mode,
                "model_residency": "active-gpu-stage" if model_loaded_to_gpu else "on-demand-component-loading",
                "pipeline_reused_between_requests": True,
                "weights_cached_in_ram": self.cache_weights_in_ram and self._generation_verified,
                "cuda_allocator_warm": self.warm_allocator,
                "system_ram_gb": self.system_ram_gb,
                "cuda_memory_allocated_gb": allocated_gb,
                "model_directory": str(MODEL_DIR),
                "model_files": len(MODEL_FILES),
                "missing_files": missing,
                "cuda": {"available": available, "device": device, "vram_gb": vram},
            }

    def _selected_offload(self):
        from ltx_pipelines.utils.types import OffloadMode

        requested = os.getenv("MASHHAD_DIRECT_OFFLOAD", "auto").strip().lower()
        if requested == "auto":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            # Official weights are ~28 GB before activation overhead. Keep margin.
            requested = "none" if vram_gb >= 44 else "cpu"
        try:
            selected = OffloadMode(requested)
        except ValueError as exc:
            raise RuntimeError("MASHHAD_DIRECT_OFFLOAD must be auto, none, cpu, or disk") from exc
        self.offload_mode = selected.value
        return selected

    def load(self) -> None:
        if self.pipeline is not None:
            return
        with self._load_lock:
            if self.pipeline is not None:
                return
            with self._state_lock:
                self.model_state = "loading"
                self.model_error = None
            try:
                missing = [str(path) for path in MODEL_FILES.values() if not path.is_file()]
                if missing:
                    raise FileNotFoundError("Missing model files: " + ", ".join(missing))
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is unavailable")

                from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
                from ltx_core.loader.registry import ModelRegistry
                from ltx_pipelines.distilled import DistilledPipeline
                from ltx_pipelines.utils.model_paths import ModelPaths

                self.cache_weights_in_ram = _cache_weights_enabled(self.system_ram_gb)
                self.warm_allocator = os.getenv("MASHHAD_DIRECT_WARM_ALLOCATOR", "1").strip().lower() not in {
                    "0", "false", "no",
                }
                registry = ModelRegistry(cache_weights=self.cache_weights_in_ram, cache_models=True)

                paths = ModelPaths.from_split(
                    transformer_path=str(MODEL_FILES["transformer"]),
                    text_encoder_path=str(MODEL_FILES["text_encoder"]),
                    video_vae_path=str(MODEL_FILES["video_vae"]),
                    audio_vae_path=str(MODEL_FILES["audio_vae"]),
                )
                pipeline = DistilledPipeline(
                    model_paths=paths,
                    spatial_upsampler_path=str(MODEL_FILES["spatial_upsampler"]),
                    loras=(),
                    quantization=None,
                    offload_mode=self._selected_offload(),
                    registry=registry,
                    alloc_trim_strategy=(
                        AllocatorTrimStrategy.DEFER if self.warm_allocator else AllocatorTrimStrategy.TRIM
                    ),
                )
                self.pipeline = pipeline
                with self._state_lock:
                    self.model_state = "pipeline-initialized"
                    self.loaded_at = time.time()
            except Exception as exc:
                with self._state_lock:
                    self.model_state = "failed"
                    self.model_error = f"{type(exc).__name__}: {exc}"[:4000]
                raise

    def cancel(self, job_id: str) -> bool:
        with self._state_lock:
            known = job_id == self._active_job or self._queued > 0
            self._cancel_requested.add(job_id)
            return known

    def generate(self, request: GenerateRequest) -> dict[str, Any]:
        if request.lora:
            raise HTTPException(
                422,
                "LoRA is reserved for a separately verified release; this BF16 build does not claim untested LoRA support.",
            )
        width, height = _dimensions(request)
        num_frames = _frames(request)
        job_dir = INPUT_DIR / request.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{request.job_id}.mp4"
        images = _conditionings(request, job_dir, num_frames)

        with self._state_lock:
            self._queued += 1
        try:
            with self._generation_lock:
                with self._state_lock:
                    self._queued -= 1
                    self._active_job = request.job_id
                if request.job_id in self._cancel_requested:
                    raise HTTPException(409, "Generation cancelled")
                self.load()
                with self._state_lock:
                    self.model_state = "loading-and-generating"

                from ltx_core.model.video_vae import get_video_chunks_number
                from ltx_pipelines.utils.media_io import encode_video

                started = time.perf_counter()
                result = self.pipeline(
                    prompt=request.prompt,
                    seed=request.seed,
                    height=height,
                    width=width,
                    frame_rate=request.fps,
                    images=images,
                    num_frames=num_frames,
                )
                if request.job_id in self._cancel_requested:
                    raise HTTPException(409, "Generation cancelled")
                encode_video(
                    video=result.video,
                    fps=request.fps,
                    audio=result.audio if request.audio_enabled else None,
                    output_path=str(output_path),
                    video_chunks_number=get_video_chunks_number(result.num_frames, result.tiling_config),
                )
                elapsed = time.perf_counter() - started
                if request.job_id in self._cancel_requested:
                    output_path.unlink(missing_ok=True)
                    raise HTTPException(409, "Generation cancelled")
                with self._state_lock:
                    self._generation_verified = True
                    self.last_generation_completed_at = time.time()
                    self.model_state = "verified-idle"
                return {
                    "status": "completed",
                    "output_path": str(output_path),
                    "generation_seconds": round(elapsed, 3),
                    "metrics": {
                        "engine": "direct-persistent",
                        "pipeline": "ltx_pipelines.distilled",
                        "precision": "bf16",
                        "quantization": None,
                        "int8_convrot": "not-tested-not-enabled",
                        "offload_mode": self.offload_mode,
                        "fps": request.fps,
                        "num_frames": result.num_frames,
                        "audio_enabled": request.audio_enabled,
                        "pipeline_reused": True,
                        "weights_cached_in_ram": self.cache_weights_in_ram,
                        "cuda_allocator_warm": self.warm_allocator,
                    },
                }
        finally:
            with self._state_lock:
                if self._active_job == request.job_id:
                    self._active_job = None
                if self.model_state == "loading-and-generating":
                    self.model_state = "verified-idle" if self._generation_verified else "pipeline-initialized"
                self._cancel_requested.discard(request.job_id)


def _system_ram_gb() -> float | None:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return round(pages * page_size / 1024**3, 1)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _cache_weights_enabled(system_ram_gb: float | None) -> bool:
    requested = os.getenv("MASHHAD_DIRECT_CACHE_WEIGHTS", "auto").strip().lower()
    if requested in {"1", "true", "yes", "on"}:
        return True
    if requested in {"0", "false", "no", "off"}:
        return False
    if requested != "auto":
        raise RuntimeError("MASHHAD_DIRECT_CACHE_WEIGHTS must be auto, 1, or 0")
    minimum = float(os.getenv("MASHHAD_DIRECT_CACHE_WEIGHTS_MIN_RAM_GB", "96"))
    return system_ram_gb is not None and system_ram_gb >= minimum


def _dimensions(request: GenerateRequest) -> tuple[int, int]:
    if request.width is not None or request.height is not None:
        if request.width is None or request.height is None:
            raise HTTPException(422, "width and height must be supplied together")
        width, height = request.width, request.height
    else:
        try:
            width, height = (int(item) for item in request.resolution.lower().split("x", 1))
        except (AttributeError, ValueError) as exc:
            raise HTTPException(422, "Invalid resolution") from exc
    if width < 256 or height < 256 or width > 2560 or height > 2560 or width % 64 or height % 64:
        raise HTTPException(422, "Width and height must be multiples of 64 between 256 and 2560")
    if width * height > 3_000_000:
        raise HTTPException(422, "Resolution is too large")
    return width, height


def _frames(request: GenerateRequest) -> int:
    if request.duration is None:
        frames = request.num_frames
    else:
        frames = int(request.duration * request.fps)
        frames = max(17, (frames // 8) * 8 + 1)
    if frames > 481:
        raise HTTPException(422, "Requested duration exceeds the 481-frame limit")
    return frames


def _asset(asset: Asset | None, directory: Path, stem: str) -> Path | None:
    if asset is None:
        return None
    suffix = Path(asset.filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(422, "Unsupported conditioning image type")
    try:
        content = base64.b64decode(asset.base64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(422, "Invalid conditioning image payload") from exc
    if len(content) > 12 * 1024 * 1024:
        raise HTTPException(413, "Conditioning image exceeds 12 MiB")
    destination = directory / f"{stem}{suffix}"
    destination.write_bytes(content)
    return destination


def _conditionings(request: GenerateRequest, directory: Path, num_frames: int) -> list[Any]:
    from ltx_pipelines.utils.types import ImageConditioningInput

    images: list[Any] = []
    first = _asset(request.first_frame, directory, "first")
    last = _asset(request.last_frame, directory, "last")
    reference = _asset(request.reference_image, directory, "reference")
    if first:
        images.append(ImageConditioningInput(str(first), 0, request.first_frame_strength))
    if last:
        images.append(ImageConditioningInput(str(last), num_frames - 1, request.last_frame_strength))
    if reference:
        images.append(ImageConditioningInput(str(reference), num_frames // 2 if first else 0, request.reference_strength))
    return images


INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)
manager = PipelineManager()
app = FastAPI(title="Mashhad LTX-2.5 Direct Persistent API", version="2.0.0")


@app.on_event("startup")
def preload() -> None:
    if os.getenv("MASHHAD_DIRECT_PRELOAD", "1").strip().lower() not in {"0", "false", "no"}:
        threading.Thread(target=_preload_safely, daemon=True, name="ltx-model-preload").start()


def _preload_safely() -> None:
    try:
        manager.load()
    except Exception:
        # Health reports the exact failure without taking down the API process.
        return


@app.get("/health")
def health() -> dict[str, Any]:
    return manager.health()


@app.post("/model/load")
def load_model() -> dict[str, Any]:
    try:
        manager.load()
    except Exception as exc:
        raise HTTPException(503, str(exc)) from exc
    return manager.health()


@app.post("/generate")
def generate(request: GenerateRequest) -> dict[str, Any]:
    try:
        return manager.generate(request)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"{type(exc).__name__}: {exc}"[:4000]) from exc


@app.delete("/generation/{job_id}")
def cancel(job_id: str) -> dict[str, Any]:
    return {"job_id": job_id, "cancel_requested": manager.cancel(job_id)}


@app.get("/outputs/{filename}")
def output(filename: str):
    from fastapi.responses import FileResponse

    if Path(filename).name != filename or Path(filename).suffix.lower() not in {".mp4", ".mov", ".webm"}:
        raise HTTPException(404)
    path = OUTPUT_DIR / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="video/mp4", filename=filename)


@app.on_event("shutdown")
def cleanup() -> None:
    # Inputs can always be reconstructed from the site request; model files are never touched.
    if os.getenv("MASHHAD_CLEAN_INPUTS_ON_STOP", "0") == "1":
        shutil.rmtree(INPUT_DIR, ignore_errors=True)
