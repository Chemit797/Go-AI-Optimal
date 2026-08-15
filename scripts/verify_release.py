#!/usr/bin/env python3
"""Verify every released model artifact against weights/manifest.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=root / "weights/manifest.json")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for artifact in payload["artifacts"]:
        path = root / artifact["path"]
        if not path.is_file():
            failures.append(f"missing: {artifact['path']}")
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        if actual_size != int(artifact["size_bytes"]):
            failures.append(
                f"size: {artifact['path']} expected={artifact['size_bytes']} actual={actual_size}"
            )
        if actual_hash != artifact["sha256"]:
            failures.append(
                f"sha256: {artifact['path']} expected={artifact['sha256']} actual={actual_hash}"
            )
    if failures:
        raise SystemExit("Release verification failed:\n" + "\n".join(failures))
    print(
        json.dumps(
            {
                "status": "ok",
                "manifest": str(manifest_path),
                "artifacts": len(payload["artifacts"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
