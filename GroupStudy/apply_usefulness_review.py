# /// script
# dependencies = ["openpyxl"]
# ///
"""Combine Jev-AR predictions with a stricter content review; preserve both."""
import hashlib
import json
from pathlib import Path

from openpyxl import load_workbook

ROOT = Path(__file__).parent
OUTPUT = ROOT / "interaction-useful-ai-reviewed.xlsx"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    useful, borderline = {}, {}
    reviewed = 0
    for part in range(1, 4):
        data = json.loads((ROOT / f"usefulness-review-{part}.json").read_text())
        assert data["reviewed_count"] == 366
        reviewed += data["reviewed_count"]
        for key, target in [("useful", useful), ("borderline", borderline)]:
            for entry in data[key]:
                assert entry["row"] not in target and entry["reason"].strip()
                assert 2 + (part - 1) * 366 <= entry["row"] <= 1 + part * 366
                target[entry["row"]] = entry["reason"]
    assert reviewed == 1098 and not useful.keys() & borderline.keys()
    book = load_workbook(ROOT / "interaction-useful-ai.xlsx")
    sheet = book.active
    sheet.insert_cols(2, 3)
    for column, header in enumerate(["Useful AI information (reviewed)", "Practical value / review note", "Review outcome"], 2):
        sheet.cell(1, column, header)
    sheet.cell(1, 5, "Jev-AR classification")
    sheet.cell(1, 6, "Jev-AR useful decision")
    sheet.cell(1, 7, "Jev-AR confidence (not review confidence)")
    sheet.cell(1, 8, "Jev-AR practical-advice probability")
    sheet.cell(1, 9, "Model uncertainty or uninspected link")
    for row in range(2, sheet.max_row + 1):
        outcome = "Not useful"
        if row in useful:
            outcome = "Useful"
        elif row in borderline:
            outcome = "Borderline; excluded"
        sheet.cell(row, 2, row in useful)
        sheet.cell(row, 3, useful.get(row) or borderline.get(row) or "No self-contained actionable AI information in the visible comment.")
        sheet.cell(row, 4, outcome)
    del book["Useful AI comments"]
    selected = book.create_sheet("Useful AI comments")
    selected.append(["Source row"] + [c.value for c in sheet[1]])
    for row in sorted(useful):
        selected.append([row] + [c.value for c in sheet[row]])
    for tab in (sheet, selected):
        tab.freeze_panes = "B2"
        tab.auto_filter.ref = tab.dimensions
    notes = book.create_sheet("Method")
    for row in [
        ["Classification", "Local Jev-AR predictions followed by AI-agent content review of all 1,098 comments."],
        ["Useful", "Concrete AI instructions, configuration, workflow, solution, practical explanation, or a specific resource with a stated applicable purpose."],
        ["Excluded", "Mere tool mentions, opinions, unanswered questions, unsolved complaints, bare links, greetings, context-dependent fragments, and non-AI advice."],
        ["Links", "Linked content was not opened. A resource qualifies only if the visible comment explains its practical AI use."],
        ["Caution", "Review is an AI judgment, not human verification. Jev-AR confidence scores do not measure confidence in the final reviewed decision."],
        ["Source SHA-256", hashlib.sha256((ROOT / "interaction-raw.xlsx").read_bytes()).hexdigest()],
    ]:
        notes.append(row)
    book.save(OUTPUT)
    saved = load_workbook(OUTPUT, read_only=True)
    original = load_workbook(ROOT / "interaction-raw.xlsx", read_only=True).active
    assert [r[0] for r in saved.active.values] == [r[0] for r in original.values]
    assert sum(r[1] is True for r in list(saved.active.values)[1:]) == len(useful)
    assert {r[0] for r in list(saved["Useful AI comments"].values)[1:]} == useful.keys()
    print(json.dumps({"output": str(OUTPUT), "useful": len(useful), "borderline_excluded": len(borderline), "other_not_useful": reviewed - len(useful) - len(borderline), "reviewed": reviewed}, indent=2))


if __name__ == "__main__":
    main()
