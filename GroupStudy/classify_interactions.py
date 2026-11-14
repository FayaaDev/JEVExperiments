# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers>=5.17,<6", "huggingface-hub", "openpyxl"]
# ///
"""Run with: uv run classify_interactions.py (all inference stays local)."""
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from openpyxl import load_workbook

MODEL = Path("/Volumes/WD/LLMs/jevar")
SOURCE = Path(__file__).with_name("interaction-raw.xlsx")
OUTPUT = SOURCE.with_name("interaction-useful-ai.xlsx")
ROUTES = {
    "AI question or help request": "Asks how to use AI, asks for advice or recommendations, or reports an unsolved problem.",
    "Practical AI answer or tip": "Gives instructions, a solution, configuration, code, workflow, or an explanation users can apply to AI tools.",
    "AI mention, opinion or news": "Mentions AI tools, opinions, announcements, jokes, or links without practical explanation.",
    "Unrelated comment": "Greetings, thanks, social chat or topics unrelated to artificial intelligence.",
}
INSTRUCTION = "Which route should this message go to?"
LABELS = {"Useful AI information", "AI-related but not actionable", "Non-AI-related"}


def decision(results, visible_text, threshold):
    useful = any(r["route"] == "Practical AI answer or tip" and r["confidence"] >= threshold for r in results)
    if useful and visible_text:
        return "Useful AI information"
    if any(r["route"] != "Unrelated comment" for r in results):
        return "AI-related but not actionable"
    return "Non-AI-related"


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Will not overwrite {OUTPUT}")
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    sys.path.insert(0, str(MODEL))
    from jev_ar_route import JevAR

    torch.set_num_threads(8)
    router = JevAR(str(MODEL), device="mps" if torch.backends.mps.is_available() else "cpu")
    for text, expected in [
        ("How can I use ChatGPT to analyze data?", "AI-related but not actionable"),
        ("I use Claude and Codex.", "AI-related but not actionable"),
        ("Claude keeps ignoring my instructions. Does anyone know why?", "AI-related but not actionable"),
        ("To keep Claude Code following your project rules, put build commands and coding conventions in CLAUDE.md at the repository root.", "Useful AI information"),
        ("Thanks everyone, have a nice weekend!", "Non-AI-related"),
    ]:
        actual = router.route(text, ROUTES, INSTRUCTION)
        print(f"Sanity check: {text!r}: {actual}", flush=True)
        assert decision([actual], text, router.threshold) == expected, (text, expected, actual)
    assert decision([], "", router.threshold) == "Non-AI-related"
    assert decision([{"route": "Practical AI answer or tip", "confidence": 1}], "", router.threshold) != "Useful AI information"

    # Leave room for the complete route list and instruction, without losing long comments.
    prefix, _ = router._encode("", INSTRUCTION, ROUTES)
    budget = router.cfg.get("max_len", 640) - len(prefix)
    assert budget > 0
    book = load_workbook(SOURCE)
    sheet = book.active
    assert sheet.cell(1, 1).value == "ZTEXT" and sheet.max_column == 1
    headers = ["AI classification", "Useful AI information", "Model confidence", "Useful AI probability", "Needs review", "Review reason", "Chunks"]
    for column, header in enumerate(headers, 2):
        sheet.cell(1, column, header)
    counts = Counter()
    review_count = 0
    cache = {}
    for row in range(2, sheet.max_row + 1):
        text = str(sheet.cell(row, 1).value or "").strip()
        if text not in cache:
            ids = router.tok(text.replace(router.tok.mask_token, " "), add_special_tokens=False)["input_ids"]
            chunks = [router.tok.decode(ids[i:i + budget], skip_special_tokens=True) for i in range(0, len(ids), budget)]
            results = [router.route(chunk, ROUTES, INSTRUCTION) for chunk in chunks]
            visible_text = re.sub(r"https?://\S+", "", text).strip()
            label = decision(results, visible_text, router.threshold)
            confidence = max((r["confidence"] for r in results), default=0.0)
            useful_probability = max((r["probabilities"]["Practical AI answer or tip"] for r in results), default=0.0)
            reasons = []
            if any(r["confidence"] < router.threshold for r in results):
                reasons.append("Low model confidence")
            if len({r["route"] for r in results}) > 1:
                reasons.append("Mixed passage classifications; check full comment")
            if re.search(r"https?://\S+", text):
                reasons.append("Linked content not inspected; classified visible text only")
            if not text:
                reasons.append("Empty comment")
            cache[text] = [label, label == "Useful AI information", confidence, useful_probability, bool(reasons), "; ".join(reasons), len(chunks)]
        values = cache[text]
        for column, value in enumerate(values, 2):
            sheet.cell(row, column, value)
        counts[values[0]] += 1
        review_count += values[4]
        if (row - 1) % 100 == 0:
            print(f"Classified {row - 1}/{sheet.max_row - 1}", flush=True)
    sheet.freeze_panes = "B2"
    sheet.auto_filter.ref = sheet.dimensions
    for column in ("D", "E"):
        for cell in sheet[column][1:]:
            cell.number_format = "0.00%"
    useful = book.create_sheet("Useful AI comments")
    useful.append(["Source row"] + [cell.value for cell in sheet[1]])
    for row in sheet.iter_rows(min_row=2):
        if row[2].value:
            useful.append([row[0].row] + [cell.value for cell in row])
    useful.freeze_panes = "C2"
    useful.auto_filter.ref = useful.dimensions
    book.save(OUTPUT)
    check = load_workbook(OUTPUT, read_only=True).active
    original = load_workbook(SOURCE, read_only=True).active
    assert check.max_row == original.max_row
    assert [r[0] for r in check.values] == [r[0] for r in original.values]
    assert all(r[1] in LABELS and r[2] == (r[1] == "Useful AI information") for r in list(check.values)[1:])
    saved_useful = load_workbook(OUTPUT, read_only=True)["Useful AI comments"]
    assert saved_useful.max_row - 1 == counts["Useful AI information"]
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
    print(json.dumps({"output": str(OUTPUT), "counts": dict(counts), "needs_review": review_count, "model": str(MODEL), "device": str(router.device), "source_sha256": source_hash}, indent=2))


if __name__ == "__main__":
    main()
