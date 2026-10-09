from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open


MODEL_REPO = "Lightricks/LTX-2.5"
MODEL_FILES = (
    "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "vae/ltx-2.5-video-vae-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
)

ROOT = Path(os.getenv("MASHHAD_LTX_ROOT", "/workspace/LTX25-DIRECT"))
MODEL_DIR = Path(os.getenv("LTX_MODELS_DIR", str(ROOT / "models")))
TEMP_DIR = ROOT / ".downloads"
STATE_DIR = ROOT / ".mashhad"
RECEIPT_PATH = STATE_DIR / "models-bf16.receipt.json"
REPORT_PATH = STATE_DIR / "models-bf16.download-report.json"
TOKEN = (os.getenv("HF_TOKEN") or "").strip() or None
REVISION = (os.getenv("LTX_MODEL_REVISION") or "main").strip()
WORKERS = max(1, min(5, int(os.getenv("LTX_DOWNLOAD_WORKERS", "5"))))
FULL_VERIFY = os.getenv("MASHHAD_FULL_MODEL_VERIFY", "0") == "1"
EDGE_BYTES = 1024 * 1024


def duration(value: float) -> str:
    seconds = max(0, int(round(value)))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def size(value: int) -> str:
    return f"{value / 1_000_000_000:.2f} GB"


def edge_fingerprint(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=20)
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        digest.update(handle.read(EDGE_BYTES))
        if file_size > EDGE_BYTES:
            handle.seek(max(0, file_size - EDGE_BYTES))
            digest.update(handle.read(EDGE_BYTES))
    digest.update(str(file_size).encode("ascii"))
    return digest.hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safetensors_valid(path: Path) -> bool:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return bool(handle.keys())
    except Exception:
        return False


def load_receipt() -> dict[str, Any]:
    try:
        value = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def cached_valid(path: Path, expected: int, previous: dict[str, Any] | None) -> bool:
    if not path.is_file() or path.stat().st_size != expected:
        return False
    if not previous or previous.get("size") != expected:
        return safetensors_valid(path)
    stat = path.stat()
    if previous.get("mtime_ns") == stat.st_mtime_ns and previous.get("edge_fingerprint"):
        return True
    fingerprint = edge_fingerprint(path)
    if fingerprint != previous.get("edge_fingerprint"):
        return False
    return safetensors_valid(path)


def clean_temp() -> None:
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)


def main() -> None:
    started = time.monotonic()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    previous = load_receipt()
    previous_files = previous.get("files") if isinstance(previous.get("files"), dict) else {}

    api = HfApi(token=TOKEN)
    info = api.model_info(MODEL_REPO, revision=REVISION, files_metadata=True)
    metadata = {item.rfilename: item.size for item in info.siblings}
    missing = [name for name in MODEL_FILES if not isinstance(metadata.get(name), int) or metadata[name] <= 0]
    if missing:
        raise RuntimeError(f"Official model manifest is missing required files: {missing}")

    def prepare(item: tuple[int, str]) -> dict[str, Any]:
        index, remote_path = item
        expected = int(metadata[remote_path])
        target = MODEL_DIR / Path(remote_path).name
        prior = previous_files.get(remote_path) if isinstance(previous_files, dict) else None
        file_started = time.monotonic()
        started_epoch = time.time()
        print(
            "MASHHAD_DOWNLOAD_EVENT "
            + json.dumps({
                "status": "started",
                "index": index,
                "total_files": len(MODEL_FILES),
                "file_name": target.name,
                "started_at": started_epoch,
                "size_bytes": expected,
            }, separators=(",", ":")),
            flush=True,
        )
        if cached_valid(target, expected, prior if isinstance(prior, dict) else None):
            print(f"CACHED {target.name} | {size(expected)}", flush=True)
            status = "cached"
        else:
            target.unlink(missing_ok=True)
            for attempt in range(1, 4):
                try:
                    downloaded = Path(
                        hf_hub_download(
                            repo_id=MODEL_REPO,
                            filename=remote_path,
                            revision=REVISION,
                            local_dir=str(TEMP_DIR),
                            token=TOKEN,
                            force_download=attempt > 1,
                        )
                    )
                    if downloaded.stat().st_size != expected or not safetensors_valid(downloaded):
                        raise RuntimeError("downloaded file failed size or safetensors validation")
                    os.replace(downloaded, target)
                    status = "downloaded"
                    break
                except Exception as exc:
                    print(f"RETRY {attempt}/3 {remote_path}: {exc}", flush=True)
                    if attempt == 3:
                        raise
                    time.sleep(3 * attempt)
            seconds = time.monotonic() - file_started
            print(
                f"DOWNLOADED {target.name} | {size(expected)} | {duration(seconds)} | "
                f"{expected / max(seconds, 0.001) / 1_000_000:.1f} MB/s",
                flush=True,
            )
        stat = target.stat()
        completed_epoch = time.time()
        seconds = time.monotonic() - file_started
        record: dict[str, Any] = {
            "remote_path": remote_path,
            "file_name": target.name,
            "size": expected,
            "mtime_ns": stat.st_mtime_ns,
            "edge_fingerprint": edge_fingerprint(target),
            "status": status,
            "index": index,
            "total_files": len(MODEL_FILES),
            "started_at": started_epoch,
            "completed_at": completed_epoch,
            "seconds": round(seconds, 3),
            "average_speed_bps": round(expected / max(seconds, 0.001), 3),
        }
        if FULL_VERIFY:
            record["sha256"] = sha256(target)
        print(
            "MASHHAD_DOWNLOAD_EVENT "
            + json.dumps({
                "status": "completed",
                "index": index,
                "total_files": len(MODEL_FILES),
                "file_name": target.name,
                "started_at": started_epoch,
                "completed_at": completed_epoch,
                "size_bytes": expected,
                "downloaded_bytes": expected,
                "average_speed_bps": record["average_speed_bps"],
                "cache_status": status,
            }, separators=(",", ":")),
            flush=True,
        )
        return record

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            records = list(pool.map(prepare, enumerate(MODEL_FILES, start=1)))
    finally:
        clean_temp()

    total_seconds = time.monotonic() - started
    receipt = {
        "schema": 1,
        "model_repo": MODEL_REPO,
        "requested_revision": REVISION,
        "resolved_revision": getattr(info, "sha", None),
        "verification": "sha256" if FULL_VERIFY else "size+safetensors+edge-fingerprint",
        "files": {item["remote_path"]: item for item in records},
        "total_size": sum(int(metadata[name]) for name in MODEL_FILES),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    temporary_receipt = RECEIPT_PATH.with_suffix(".tmp")
    temporary_receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_receipt, RECEIPT_PATH)
    REPORT_PATH.write_text(
        json.dumps({**receipt, "total_seconds": round(total_seconds, 3)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"ALL 5 BF16 MODELS READY IN ONE DIRECTORY | {size(receipt['total_size'])} | "
        f"{duration(total_seconds)} | {MODEL_DIR}",
        flush=True,
    )
    print("All 5 requested models ready", flush=True)


if __name__ == "__main__":
    main()
