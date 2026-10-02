#!/usr/bin/env python3
"""Dry-run the on-device model on scans using fields instead of a free-text name.

The model fills date, classification (from a fixed list), company, amount and
description; code assembles DATE__CLASS__company__amount-or-description.
Renames nothing.

Usage: try_apple_fm_scans.py FILE [FILE ...]
"""

import asyncio
import re
import sys
import tempfile
import time
from pathlib import Path

import apple_fm_sdk as fm
import pymupdf

INSTRUCTIONS = """You read scanned documents and extract fields for filing.
Classification rules, applied in order:
- TAXES: only IRS/tax forms (1040, 1099, W-2, 1095), giving receipts or yearly donation totals.
- STATEMENT: periodic account statements from banks, credit cards, credit unions or investment accounts.
- BILL: mentions amount due, payment due, payment received or payment made.
- RECEIPT: a payment or sales receipt.
- GENERAL: anything else, including letters and notices."""

PROMPT = "Extract the fields for this scanned document."


@fm.generable()
class ScanFields:
    document_date: str = fm.guide(
        description="Date printed on the document as YYYYMMDD, or 'none' if no date is printed",
        regex=r"(\d{8}|none)",
    )
    classification: str = fm.guide(
        description="Document type", anyOf=["TAXES", "STATEMENT", "BILL", "RECEIPT", "GENERAL"]
    )
    company: str = fm.guide(description="Organization or person that issued the document")
    amount: str = fm.guide(
        description="Amount due or paid as dollars.cents, or 'none' if there is no amount",
        regex=r"(\d+\.\d{2}|none)",
    )
    description: str = fm.guide(description="What the document is, in at most three words")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower().replace("'", "")).strip("-") or "unknown"


def build_name(fields: ScanFields, ext: str) -> str:
    parts = [fields.document_date if fields.document_date != "none" else "nodate",
             fields.classification, slug(fields.company)]
    if fields.classification in ("BILL", "RECEIPT") and fields.amount != "none":
        parts.append(fields.amount)
    elif fields.classification != "STATEMENT":
        parts.append(slug(fields.description))
    return "__".join(parts) + ext


async def suggest(file_path: str, tmp_dir: str) -> None:
    with pymupdf.open(file_path) as doc:
        image_path = Path(tmp_dir) / (Path(file_path).stem + ".png")
        doc[0].get_pixmap(dpi=150).save(image_path)

    session = fm.LanguageModelSession(instructions=INSTRUCTIONS)
    print(f"\n=== {Path(file_path).name}")
    start = time.monotonic()
    try:
        fields = await session.respond([PROMPT, fm.ImageAttachment(image_path)], generating=ScanFields)
    except fm.FoundationModelsError as e:
        print(f"  ERROR ({type(e).__name__}): {e}")
        return
    print(f"  fields:   date={fields.document_date} class={fields.classification} "
          f"company={fields.company!r} amount={fields.amount} desc={fields.description!r}")
    print(f"  filename: {build_name(fields, Path(file_path).suffix)}")
    print(f"  time:     {time.monotonic() - start:.1f}s")


async def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    available, reason = fm.SystemLanguageModel().is_available()
    if not available:
        sys.exit(f"On-device model unavailable: {reason}")
    with tempfile.TemporaryDirectory(prefix="try_apple_fm_") as tmp_dir:
        for file_path in sys.argv[1:]:
            await suggest(file_path, tmp_dir)


if __name__ == "__main__":
    asyncio.run(main())
