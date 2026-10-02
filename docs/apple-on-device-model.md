# Apple's on-device model as a provider

Tested 2026-10-02 on macOS 27.0.1. Question: can Apple's on-device model
replace sending files to Anthropic, OpenAI or a local LLM?

**Short answer:** yes for screenshots; not yet for scans. A hybrid (Apple OCR,
keyword classification in code, the model for fields only) got close for scans
but needs work before it is trustworthy. Status: parked, not built.

## What Apple provides

- **Foundation Models framework** — the on-device model behind Apple
  Intelligence (roughly 3B parameters). macOS 27 added image input.

- **Python SDK** — `pip install apple-fm-sdk`
  ([GitHub](https://github.com/apple/python-apple-fm-sdk),
  [docs](https://apple.github.io/python-apple-fm-sdk/)). Needs an Apple Silicon
  Mac, Python 3.10+, Xcode and Apple Intelligence turned on. Supports image
  attachments (`fm.ImageAttachment`) and guided generation (`@fm.generable`
  classes with `fm.guide(...)`, including `anyOf` choices and `regex`).

- **Vision framework** — separate, on-device text recognition (OCR), usable from
  Python through `pyobjc-framework-Vision`. Much better at digits than the
  model's own reading of an image.

Nothing leaves the Mac; there is no API key and no per-call cost.

## What was tested

The test scripts are in `experiments/apple-on-device/`. None of them rename
anything. Scan results were scored against the filenames Claude produced for
the same files earlier.

| Approach | Script | Result |
| --- | --- | --- |
| Image + the folder's real prompt from `config.yaml` | `try_apple_fm.py` | Screenshots fine; scans unusable |
| Image + structured fields, filename built in code | `try_apple_fm_scans.py` | Misreads digits and dates |
| Vision OCR text + structured fields + type rules in the prompt | `try_apple_fm_ocr.py` | Numbers right, but 7 of 10 refused |
| Hybrid: OCR, type from keywords in code, model for fields only | `try_hybrid.py` | 33 of 37 answered; type is the weak spot |

### 1. Screenshots — usable

About 4–6 seconds each. Names were accurate but generic: a mortgage payment
confirmation became `payment-summary-digital-interface` rather than naming the
mortgage, the amount and "scheduled".

### 2. Scans with the real prompt — unusable

The scans prompt in `config.yaml` is a long decision tree. The small model got
the format, the dates and the classification wrong (it called a government
agency a credit card company).

### 3. Scans with structured fields from the image — unsafe

Fields: date, type (fixed list of five), company, amount, description. Only 1
of 10 matched exactly. Amounts were misread in a believable way (`15.54` read
as `15.50`, `25.97` as `25.00`), dates were years off, and the recipient's own
name was returned as the company.

### 4. Vision OCR first — accurate, but refuses

Reading the text with Vision and giving the model text instead of the image
fixed the numbers. But whenever the classification rules were in the
instructions or the prompt, 7 of 10 scans were refused with
`LanguageModelError:3 - The model refused to answer`. Findings:

- Same files refuse on every run — not random.

- `SystemLanguageModelGuardrails.PERMISSIVE_CONTENT_TRANSFORMATIONS` changed
  nothing, so it is not the safety filter.

- Removing the `regex` constraints changed nothing.

- With rules present, even a one-line prompt (`Total: $15.54`, or `Hello`) was
  refused. With no rules at all, 6 of the 7 answered.

- The SDK raises a generic `GenerationError` (status 255) with no refusal
  explanation, so the cause could not be pinned down.

### 5. Hybrid — closest

Vision OCR → type chosen in code from the keyword rules in `config.yaml` →
model extracts date, company, amount and description with **no rules in the
prompt** → on refusal, retry once with different wording, then count it as
"would go to Claude" (Claude was not actually called).

Sample: 37 scans — all 11 bills, 5 statements and 5 tax documents, plus 8
receipts and 8 general documents. About 2 seconds per scan, 69 seconds total.

| Measure | Result |
| --- | --- |
| Answered on-device | 33 of 37 (the retry never helped) |
| Refused → would go to Claude | 4 (3 tax documents, 1 bank bill) |
| Amount matches Claude | 16 of 17 |
| Company matches Claude | 27 of 33 |
| Date matches Claude | 25 of 33 |
| Type matches Claude | 22 of 33 |

Notes on the misses:

- **Type** is the main weakness. Seven medical bills were typed GENERAL or
  RECEIPT because they say "balance due" or "patient responsibility", which are
  not in the bill phrases in `config.yaml` ("amount due", "payment received",
  "payment made", "payment due"). Claude infers these; keywords cannot.

- **Date:** four misses cannot be judged on old files. When no date is printed,
  the fallback is the file's creation time, and these files were all re-created
  on the same day when they were copied. On new scans the fallback behaves as
  it does for Claude. Real misses: one date changed between runs, one was not
  found, one was a due date versus a statement date, and one is the tax-year
  rule (`2025`) versus the full date Claude kept.

- **Company:** some misses are arguable (a mortgage servicer's legal or former
  name versus its brand). Real misses: OCR typos (a dropped first letter) and a
  person's name taken from an invoice.

- **Refusals** fall on the most sensitive documents (tax forms), so those would
  still go to Claude unless refused files are left un-renamed instead.

- The existing names are Claude's output, not checked truth. One Claude name
  was dated six days in the future; the on-device model read the date the other
  way round, which suggests Claude swapped month and day there.

## If this is picked up again

1. Add an `apple` provider, chosen per folder in `config.yaml`. Use the model
   directly for screenshots.

2. For scans, use the hybrid. Extend the bill phrases first (for example
   "balance due", "patient responsibility", "amount enclosed", "total due") and
   re-test on scans that were *not* used to pick those phrases.

3. Decide what happens to refused files: send them to Claude, or leave them
   un-renamed so nothing leaves the Mac.

4. Re-check refusals on newer macOS releases; the cause was not found.

## Running the scripts

The project's `venv` points at a Python 3.12 that is no longer installed, so
the scripts use their own environment:

```sh
python3 -m venv /Users/jn/Apps/ai-file-renamer/experiments/apple-on-device/venv
/Users/jn/Apps/ai-file-renamer/experiments/apple-on-device/venv/bin/pip install -r /Users/jn/Apps/ai-file-renamer/experiments/apple-on-device/requirements.txt
```

Then pass one or more files, for example a screenshot:

```sh
/Users/jn/Apps/ai-file-renamer/experiments/apple-on-device/venv/bin/python /Users/jn/Apps/ai-file-renamer/experiments/apple-on-device/try_apple_fm.py "/Users/jn/Desktop/Screenshots/CleanShot 2026-10-01 at 7.45.26 AM@2x.png"
```

`try_apple_fm_scans.py`, `try_apple_fm_ocr.py` and `try_hybrid.py` take PDF
scans the same way: the script path, then one or more quoted PDF paths.

`try_apple_fm.py` picks the prompt by matching the file's folder against
`config.yaml`; pass `--folder 0` (screenshots) or `--folder 1` (scans) for files
elsewhere. Results print to the terminal only — do not commit them, as they
contain personal and financial details.
