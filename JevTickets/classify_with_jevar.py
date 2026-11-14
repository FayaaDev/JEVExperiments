# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers>=5.17,<6", "huggingface-hub", "openpyxl"]
# ///
"""Classify an Excel text column with local Jev-AR and configurable routes."""
import argparse
import json
import os
import sys
from collections import Counter
from copy import copy
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.utils import column_index_from_string, get_column_letter


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object with instruction and routes")
    instruction, routes = config.get("instruction"), config.get("routes")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a nonempty string")
    if not isinstance(routes, dict) or len(routes) < 2:
        raise ValueError("routes must contain at least two labels and descriptions")
    if any(not label.strip() or not isinstance(description, str) for label, description in routes.items()):
        raise ValueError("Route labels must be nonempty and descriptions must be strings")
    return instruction, routes


def classify_workbook(source, output, column, instruction, routes, router):
    source, output = Path(source), Path(output)
    if source.suffix.lower() != ".xlsx" or output.suffix.lower() != ".xlsx":
        raise ValueError("Input and output must be .xlsx files")
    if output.exists() or source.resolve() == output.resolve():
        raise FileExistsError(f"Will not overwrite {output}")
    book = load_workbook(source)
    sheet = book.active
    matches = [cell.column for cell in sheet[1] if cell.value == column]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one header named {column!r} on the active sheet")
    text_column = matches[0]
    prefix, _ = router._encode("", instruction, routes)
    budget = router.cfg.get("max_len", 640) - len(prefix)
    if budget <= 0:
        raise ValueError("Instruction and routes leave no room for input text")

    result_columns = []
    for header in ("القسم المسؤول", "درجة الثقة"):
        matches = [cell.column for cell in sheet[1] if cell.value == header]
        if len(matches) > 1:
            raise ValueError(f"Expected at most one header named {header!r}")
        result_column = matches[0] if matches else sheet.max_column + 1
        sheet.cell(1, result_column, header)._style = copy(sheet.cell(1, text_column)._style)
        sheet.column_dimensions[get_column_letter(result_column)]._style = copy(
            sheet.column_dimensions[get_column_letter(text_column)]._style)
        result_columns.append(result_column)
    counts, cache = Counter(), {}
    review_count = 0
    for row in range(2, sheet.max_row + 1):
        progress = f"[{row - 1}/{sheet.max_row - 1}] Ticket (Excel row {row})"
        print(f"{progress}: Processing...", file=sys.stderr, flush=True)
        cell = sheet.cell(row, text_column)
        reason = ""
        if cell.data_type == "f":
            reason = "Formula cell; provide plain text"
        elif cell.value is None or isinstance(cell.value, str) and not cell.value.strip():
            reason = "Empty text"
        elif not isinstance(cell.value, str):
            reason = "Non-text cell"
        if reason:
            values = [None, None, True, reason]
        else:
            text = cell.value.strip()
            if text not in cache:
                tokens = router.tok(text.replace(router.tok.mask_token, " "), add_special_tokens=False)["input_ids"]
                if len(tokens) > budget:
                    # shortcut: no generic chunk aggregation; long inputs need a workflow-specific policy.
                    values = [None, None, True, f"Text exceeds {budget}-token budget; not classified"]
                else:
                    result = router.route(text, routes, instruction)
                    values = [result["route"], result["confidence"], result["escalate"], "Low model confidence" if result["escalate"] else ""]
                cache[text] = values
            values = cache[text]
        for result_column, value in zip(result_columns, values[:2]):
            target = sheet.cell(row, result_column)
            target._style = copy(cell._style)
            target.value = value
            if isinstance(value, str):
                target.data_type = "s"
        sheet.cell(row, result_columns[1]).number_format = "0.00%"
        counts[values[0] or "Not classified"] += 1
        review_count += bool(values[2])
        if values[0] is None:
            status = f"Skipped — {values[3]}"
        else:
            status = f"Completed — {values[0]} ({values[1]:.2%})"
            if values[2]:
                status += f" — Needs review: {values[3]}"
        print(f"{progress}: {status}", file=sys.stderr, flush=True)
    removed_columns = [cell.column for cell in sheet[1]
                       if cell.value in ("نوع الطلب", "الأولوية", "الإجراء التالي")]
    if removed_columns:
        dimensions = [(column_index_from_string(key), copy(dimension))
                      for key, dimension in sheet.column_dimensions.items()]
        for index in reversed(removed_columns):
            sheet.delete_cols(index)
        sheet.column_dimensions.clear()
        for index, dimension in dimensions:
            if index not in removed_columns:
                new_index = index - sum(removed < index for removed in removed_columns)
                dimension.index = get_column_letter(new_index)
                dimension.min = dimension.max = new_index
                sheet.column_dimensions[dimension.index] = dimension
    # Exclusive creation also prevents overwriting a file created during inference.
    with output.open("xb") as stream:
        book.save(stream)
    return {"output": str(output), "sheet": sheet.title, "counts": dict(counts), "needs_review": review_count}


def ticket_arguments(directory):
    output = directory / "tickets-routed.xlsx"
    number = 2
    while output.exists():
        output = directory / f"tickets-routed-{number}.xlsx"
        number += 1
    return [str(directory / "raw.xlsx"), str(output),
            "--column", "نص الرسالة",
            "--config", str(directory / "ticket-routing.json")]


def tickets_main():
    main(ticket_arguments(Path(__file__).resolve().parent) + sys.argv[1:])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input .xlsx workbook; reads its active sheet")
    parser.add_argument("output", type=Path, help="New .xlsx workbook")
    parser.add_argument("--column", required=True, help="Exact text-column header in row 1")
    parser.add_argument("--config", required=True, type=Path, help="JSON containing instruction and routes")
    parser.add_argument("--model", type=Path, default=Path("/Volumes/WD/LLMs/jevar"), help="Local Jev-AR folder containing jev_ar_route.py and weights")
    args = parser.parse_args(argv)
    instruction, routes = load_config(args.config)
    if not args.model.is_dir():
        parser.error(f"Local model folder not found: {args.model}")
    if args.output.exists():
        parser.error(f"Will not overwrite {args.output}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("Loading Jev Arabic Model", file=sys.stderr, flush=True)
    import torch

    sys.path.insert(0, str(args.model.resolve()))
    from jev_ar_route import JevAR

    torch.set_num_threads(8)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    router = JevAR(str(args.model.resolve()), device=device)
    print("Model Loaded", file=sys.stderr, flush=True)
    summary = classify_workbook(args.input, args.output, args.column, instruction, routes, router)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
