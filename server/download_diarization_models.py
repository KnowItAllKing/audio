from __future__ import annotations

import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path

from .diarization import default_sherpa_model_paths


SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
SEGMENTATION_ARCHIVE_SHA256 = "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488"
SEGMENTATION_MODEL_SHA256 = "d582f4b4c6b48205de7e0643c57df0df5615a3c176189be3fc461e9d18827b5d"
EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)
EMBEDDING_SHA256 = "1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str) -> None:
    if destination.is_file() and _sha256(destination) == expected_sha256:
        print(f"Already verified: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "Cadence model installer"})
    print(f"Downloading {url}")
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    actual_sha256 = _sha256(temporary)
    if actual_sha256 != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"checksum mismatch for {url}: expected {expected_sha256}, got {actual_sha256}"
        )
    os.replace(temporary, destination)


def _extract_segmentation(archive: Path, cache_root: Path) -> None:
    with tarfile.open(archive, mode="r:bz2") as bundle:
        for member in bundle.getmembers():
            target = (cache_root / member.name).resolve()
            if not target.is_relative_to(cache_root.resolve()):
                raise RuntimeError(f"unsafe path in model archive: {member.name}")
        bundle.extractall(cache_root)


def main() -> None:
    segmentation_model, embedding_model = default_sherpa_model_paths()
    cache_root = embedding_model.parent
    cache_root.mkdir(parents=True, exist_ok=True)

    archive = cache_root / "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
    _download(SEGMENTATION_URL, archive, SEGMENTATION_ARCHIVE_SHA256)
    if not segmentation_model.is_file() or _sha256(segmentation_model) != SEGMENTATION_MODEL_SHA256:
        _extract_segmentation(archive, cache_root)
    if _sha256(segmentation_model) != SEGMENTATION_MODEL_SHA256:
        raise RuntimeError(f"invalid extracted segmentation model: {segmentation_model}")

    _download(EMBEDDING_URL, embedding_model, EMBEDDING_SHA256)
    print(f"Diarization models ready in {cache_root}")


if __name__ == "__main__":
    main()
