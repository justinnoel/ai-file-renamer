#!/usr/bin/env python3
"""Dry-run: Vision OCR first, then the on-device model extracts fields from the text.

Same fields and filename assembly as try_apple_fm_scans.py, but the model reads
OCR text instead of the image. Renames nothing.

Usage: try_apple_fm_ocr.py [--show-text] FILE [FILE ...]
"""

import asyncio
import sys
import time
from pathlib import Path

import apple_fm_sdk as fm
import pymupdf
import Quartz
import Vision

sys.path.insert(0, str(Path(__file__).parent))
from try_apple_fm_scans import INSTRUCTIONS, ScanFields, build_name

MAX_CHARS = 6000


def ocr_page(pdf_path: str) -> str:
    """Recognize text on page 1 with Vision's accurate recognizer, top to bottom."""
    with pymupdf.open(pdf_path) as doc:
        png = doc[0].get_pixmap(dpi=200).tobytes("png")
    provider = Quartz.CGDataProviderCreateWithCFData(png)
    image = Quartz.CGImageCreateWithPNGDataProvider(provider, None, False, Quartz.kCGRenderingIntentDefault)

    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, None)
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError(f"Vision OCR failed: {error}")

    # Vision's origin is bottom-left; sort by descending y, then x, for reading order.
    lines = sorted(
        request.results(),
        key=lambda o: (-round(o.boundingBox().origin.y, 2), o.boundingBox().origin.x),
    )
    return "\n".join(o.topCandidates_(1)[0].string() for o in lines)


async def suggest(file_path: str, show_text: bool) -> None:
    print(f"\n=== {Path(file_path).name}")
    start = time.monotonic()
    text = ocr_page(file_path)
    ocr_time = time.monotonic() - start
    if show_text:
        print("  --- OCR text ---\n" + text + "\n  ----------------")

    model = fm.SystemLanguageModel(
        guardrails=fm.SystemLanguageModelGuardrails.PERMISSIVE_CONTENT_TRANSFORMATIONS
    )
    session = fm.LanguageModelSession(model=model)
    prompt = f"{INSTRUCTIONS}\n\nExtract the fields for this scanned document. Its text, read by OCR:\n\n{text[:MAX_CHARS]}"
    try:
        fields = await session.respond(prompt, generating=ScanFields)
    except fm.FoundationModelsError as e:
        print(f"  ERROR ({type(e).__name__}): {e}")
        return
    print(f"  fields:   date={fields.document_date} class={fields.classification} "
          f"company={fields.company!r} amount={fields.amount} desc={fields.description!r}")
    print(f"  filename: {build_name(fields, Path(file_path).suffix)}")
    print(f"  time:     {time.monotonic() - start:.1f}s (OCR {ocr_time:.1f}s, {len(text)} chars)")


async def main() -> None:
    args = sys.argv[1:]
    show_text = "--show-text" in args
    files = [a for a in args if a != "--show-text"]
    if not files:
        sys.exit(__doc__)
    available, reason = fm.SystemLanguageModel().is_available()
    if not available:
        sys.exit(f"On-device model unavailable: {reason}")
    for file_path in files:
        await suggest(file_path, show_text)


if __name__ == "__main__":
    asyncio.run(main())
