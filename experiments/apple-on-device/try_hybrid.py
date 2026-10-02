#!/usr/bin/env python3
"""Dry-run the hybrid: Vision OCR -> keyword classification in code -> on-device
model extracts date, company, amount and description only.

A refusal is retried once with a different prompt; a second refusal is counted
as "would fall back to Claude" (Claude is NOT called). Each result is scored
against the existing filename, which Claude produced earlier.
Renames nothing.

Usage: try_hybrid.py FILE [FILE ...]
"""

import asyncio
import os
from datetime import datetime
import re
import sys
import time
from pathlib import Path

import apple_fm_sdk as fm

sys.path.insert(0, str(Path(__file__).parent))
from try_apple_fm_ocr import ocr_page
from try_apple_fm_scans import slug

# --- Classification: config.yaml's scan rules, as keyword checks ----------------

TAX_PATTERNS = [
    r"\bform\s*(1040|1099|w-?2|1095(-b)?)\b", r"\b(1040|1099(-[a-z]+)?|w-2|1095-?[abc]?)\b",
    r"tax year", r"tax statement", r"supplemental tax form", r"department of the treasury",
    r"internal revenue service", r"united states treasury", r"giving receipt",
    r"total (charitable )?(donations|contributions|giving)",
]
EXCLUDED_FROM_TAXES_AND_STATEMENT = [  # utilities, healthcare, vehicle finance
    r"\butility\b", r"\belectric", r"\benergy\b", r"\bwater\b", r"\bgas service",
    r"oncology", r"\bmedical\b", r"\bclinic\b", r"hospital", r"healthcare", r"\bpatient\b",
    r"pharmacy", r"\bphysician", r"health ?care associates of texas",
    r"kia finance", r"auto financ", r"motor (credit|finance)",
]
STATEMENT_PATTERNS = [
    r"\bbank\b", r"credit union", r"credit card", r"card ?services", r"\bvisa\b", r"mastercard",
    r"american express", r"brokerage", r"investment", r"mortgage",
]
STATEMENT_CONFIRM = [r"statement (date|period)", r"account statement", r"statement", r"closing date"]
BILL_PATTERNS = [r"amount due", r"payment received", r"payment made", r"payment due"]
RECEIPT_PATTERNS = [r"\breceipt\b", r"\bsubtotal\b", r"\bchange due\b", r"\bcash\b", r"\btender", r"thank you for shopping"]


def any_match(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def classify(text: str) -> str:
    t = text.lower()
    excluded = any_match(EXCLUDED_FROM_TAXES_AND_STATEMENT, t)
    if not excluded and any_match(TAX_PATTERNS, t):
        return "TAXES"
    if not excluded and any_match(STATEMENT_PATTERNS, t) and any_match(STATEMENT_CONFIRM, t):
        return "STATEMENT"
    if any_match(BILL_PATTERNS, t):
        return "BILL"
    if any_match(RECEIPT_PATTERNS, t):
        return "RECEIPT"
    return "GENERAL"


def tax_year(text: str) -> str | None:
    for p in [r"tax year\s*:?\s*(20\d\d)", r"(20\d\d)\s+form\s+(1040|1099|w-?2|1095)",
              r"form\s+(?:1040|1099\S*|w-?2|1095\S*)\s+(20\d\d)", r"for (?:calendar )?year\s+(20\d\d)"]:
        m = re.search(p, text.lower())
        if m:
            return m.group(1)
    return None


# --- Extraction: on-device model, no rules ---------------------------------------

@fm.generable()
class Extracted:
    document_date: str = fm.guide(description="Date printed on the document as YYYYMMDD, or 'none'")
    company: str = fm.guide(description="Company or organization that sent or issued the document, not the person it is addressed to")
    amount: str = fm.guide(description="Total amount due or paid as dollars.cents, or 'none'")
    description: str = fm.guide(description="What the document is, in at most three words")


PROMPTS = [lambda t: t, lambda t: "Fill in the fields for this document:\n\n" + t]


async def extract(text: str) -> tuple[Extracted | None, int]:
    for attempt, make_prompt in enumerate(PROMPTS, start=1):
        try:
            return await fm.LanguageModelSession().respond(make_prompt(text[:6000]), generating=Extracted), attempt
        except fm.FoundationModelsError:
            continue
    return None, len(PROMPTS)


def build_name(cls: str, date: str, f: Extracted, ext: str) -> str:
    parts = [date, cls, slug(f.company)]
    if cls in ("BILL", "RECEIPT") and re.fullmatch(r"\d+\.\d{2}", f.amount.replace(",", "").lstrip("$")):
        parts.append(f.amount.replace(",", "").lstrip("$"))
    elif cls not in ("STATEMENT",):
        parts.append(slug(f.description))
    return "__".join(parts) + ext


def created_date(path: str) -> str:
    """config.yaml's fallback when no date is printed: the file's creation time."""
    st = os.stat(path)
    return datetime.fromtimestamp(getattr(st, "st_birthtime", st.st_mtime)).strftime("%Y%m%d-%H%M")


# --- Scoring against Claude's earlier filename -----------------------------------

def reference(name: str) -> dict:
    parts = Path(name).name.split(".pdf")[0].removesuffix(".ext").split("__")
    return {
        "date": parts[0][:8] if parts else "",
        "class": parts[1] if len(parts) > 1 else "",
        "company": parts[2] if len(parts) > 2 else "",
        "amount": parts[3] if len(parts) > 3 and re.fullmatch(r"\d+\.\d{2}", parts[3]) else None,
    }


def company_match(a: str, b: str) -> bool:
    a, b = slug(a), slug(b)
    return a == b or a in b or b in a or a.split("-")[0] == b.split("-")[0]


async def main() -> None:
    files = sys.argv[1:]
    if not files:
        sys.exit(__doc__)
    totals = {"date": 0, "class": 0, "company": 0, "amount": 0, "amount_n": 0, "retried": 0, "claude": 0}
    start_all = time.monotonic()
    for path in files:
        ref = reference(path)
        text = ocr_page(path)
        cls = classify(text)
        fields, attempts = await extract(text)
        if attempts > 1 and fields:
            totals["retried"] += 1
        if not fields:
            totals["claude"] += 1
            print(f"CLAUDE   {Path(path).name}\n         refused twice -> would fall back to Claude (type by code: {cls})")
            continue
        printed = re.sub(r"\D", "", fields.document_date)
        date = (tax_year(text) if cls == "TAXES" else None) or (printed if len(printed) == 8 else created_date(path))
        got_amount = fields.amount.replace(",", "").lstrip("$")
        score = {
            "date": date[:8] == ref["date"],
            "class": cls == ref["class"],
            "company": company_match(fields.company, ref["company"]),
        }
        for k, v in score.items():
            totals[k] += v
        amount_mark = ""
        if ref["amount"]:
            totals["amount_n"] += 1
            totals["amount"] += got_amount == ref["amount"]
            amount_mark = f" amount{'✓' if got_amount == ref['amount'] else '✗'}"
        marks = " ".join(f"{k}{'✓' if v else '✗'}" for k, v in score.items()) + amount_mark
        print(f"{'OK  ' if all(score.values()) else 'DIFF'}     {Path(path).name}\n"
              f"         -> {build_name(cls, date, fields, '.pdf')}   [{marks}]{' (retried)' if attempts > 1 else ''}")
    n = len(files)
    answered = n - totals["claude"]
    print(f"\n{n} files in {time.monotonic() - start_all:.0f}s. Answered on-device: {answered} "
          f"({totals['retried']} needed the retry). Would go to Claude: {totals['claude']}.")
    print(f"Of the {answered} answered, matching Claude's earlier name: date {totals['date']}, "
          f"type {totals['class']}, company {totals['company']}, amount {totals['amount']} of {totals['amount_n']}.")


if __name__ == "__main__":
    asyncio.run(main())
