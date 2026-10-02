#!/usr/bin/env python3
"""Dry-run Apple's on-device Foundation Model against the renamer's prompts.

Renames nothing. For each file, prints the filename the on-device model
suggests, using the prompt from the matching folder in config.yaml.

Usage: try_apple_fm.py [--config PATH] [--folder N] FILE [FILE ...]
"""

import argparse
import asyncio
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import apple_fm_sdk as fm
import pymupdf
import yaml

INSTRUCTIONS = (
    "You are a file naming assistant. Analyze the attached image and follow "
    "the user's rules exactly to produce a new filename."
)


@fm.generable()
class Suggestion:
    analysis: str = fm.guide(description="Short explanation of how each part of the filename was chosen")
    filename: str = fm.guide(description="The new filename, following the rules exactly")


def created_vars(path: str) -> dict:
    stat = os.stat(path)
    ts = datetime.fromtimestamp(getattr(stat, "st_birthtime", stat.st_mtime))
    return {
        "created_date": ts.strftime("%Y%m%d"),
        "created_datetime": ts.strftime("%Y%m%d-%H%M%S"),
        "created_datetime_short": ts.strftime("%Y%m%d-%H%M"),
        "created_iso": ts.isoformat(),
        "created_year": ts.strftime("%Y"),
        "created_month": ts.strftime("%m"),
        "created_day": ts.strftime("%d"),
        "created_time": ts.strftime("%H%M%S"),
    }


def pick_folder(folders: list, file_path: str, index: int | None) -> dict:
    if index is not None:
        return folders[index]
    resolved = Path(file_path).resolve()
    for folder in folders:
        folder_path = Path(folder["path"].replace("\\ ", " ")).expanduser().resolve()
        if folder_path in resolved.parents:
            return folder
    sys.exit(f"No folder in config matches {file_path}; pass --folder N (0-based).")


def to_image(file_path: str, tmp_dir: str) -> Path:
    if not file_path.lower().endswith(".pdf"):
        return Path(file_path)
    with pymupdf.open(file_path) as doc:
        out = Path(tmp_dir) / (Path(file_path).stem + ".png")
        doc[0].get_pixmap(dpi=150).save(out)
        return out


async def suggest(file_path: str, folder: dict, tmp_dir: str) -> None:
    prompt = folder["prompt"].format(**created_vars(file_path))
    image = fm.ImageAttachment(to_image(file_path, tmp_dir))
    session = fm.LanguageModelSession(instructions=INSTRUCTIONS)

    print(f"\n=== {file_path}")
    start = time.monotonic()
    try:
        result = await session.respond([prompt, image], generating=Suggestion)
    except fm.FoundationModelsError as e:
        print(f"  ERROR ({type(e).__name__}): {e}")
        return
    elapsed = time.monotonic() - start
    print(f"  filename: {result.filename}")
    print(f"  analysis: {result.analysis}")
    print(f"  time:     {elapsed:.1f}s")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "config.yaml"))
    parser.add_argument("--folder", type=int, help="0-based index of the folder whose prompt to use")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    available, reason = fm.SystemLanguageModel().is_available()
    if not available:
        sys.exit(f"On-device model unavailable: {reason}")

    with open(args.config) as f:
        folders = yaml.safe_load(f)["folders"]

    with tempfile.TemporaryDirectory(prefix="try_apple_fm_") as tmp_dir:
        for file_path in args.files:
            await suggest(file_path, pick_folder(folders, file_path, args.folder), tmp_dir)


if __name__ == "__main__":
    asyncio.run(main())
