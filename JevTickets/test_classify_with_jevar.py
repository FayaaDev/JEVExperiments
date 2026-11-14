import json
import io
import unittest
from contextlib import redirect_stderr
from copy import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

from classify_with_jevar import classify_workbook, load_config, ticket_arguments, tickets_main


INSTRUCTION = "Classify the text"
ROUTES = {"قبول": "Accepted", "مراجعة": "Review"}


class FakeTokenizer:
    mask_token = "[MASK]"

    def __init__(self):
        self.calls = []

    def __call__(self, text, *, add_special_tokens):
        self.calls.append((text, add_special_tokens))
        return {"input_ids": text.split()}


class FakeRouter:
    def __init__(self, max_len=5):
        self.cfg = {"max_len": max_len}
        self.tok = FakeTokenizer()
        self.encode_calls = []
        self.calls = []

    def _encode(self, text, instruction, routes):
        self.encode_calls.append((text, instruction, routes))
        return [101, 102], None

    def route(self, text, routes, instruction):
        self.calls.append((text, routes, instruction))
        review = text == "uncertain"
        return {
            "route": "مراجعة" if review else "قبول",
            "confidence": 0.55 if review else 0.9,
            "probabilities": {"قبول": 0.45 if review else 0.9,
                              "مراجعة": 0.55 if review else 0.1},
            "escalate": review,
        }


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / "source.xlsx"
        self.output = self.directory / "output.xlsx"
        self.router = FakeRouter()

    def save_workbook(self, rows):
        book = Workbook()
        sheet = book.active
        sheet.title = "Input"
        for row in rows:
            sheet.append(row)
        other = book.create_sheet("Other")
        other.append(["Keep me", "=1+2"])
        book.save(self.source)
        book.close()

    def classify(self):
        return classify_workbook(self.source, self.output, "Text",
                                 INSTRUCTION, ROUTES, self.router)

    def open_book(self, path):
        book = load_workbook(path)
        self.addCleanup(book.close)
        return book

    def test_load_config_success(self):
        path = self.directory / "config.json"
        path.write_text(json.dumps({"instruction": INSTRUCTION, "routes": ROUTES}),
                        encoding="utf-8")
        self.assertEqual(load_config(path), (INSTRUCTION, ROUTES))

    def test_tickets_defaults_and_existing_outputs(self):
        expected = [str(self.directory / "raw.xlsx"),
                    str(self.directory / "tickets-routed.xlsx"),
                    "--column", "نص الرسالة",
                    "--config", str(self.directory / "ticket-routing.json")]
        self.assertEqual(ticket_arguments(self.directory), expected)
        for name in ("tickets-routed.xlsx", "tickets-routed-2.xlsx"):
            (self.directory / name).write_bytes(b"Keep me")
        expected[1] = str(self.directory / "tickets-routed-3.xlsx")
        self.assertEqual(ticket_arguments(self.directory), expected)
        self.assertEqual((self.directory / "tickets-routed.xlsx").read_bytes(), b"Keep me")

    def test_tickets_command_forwards_defaults_and_overrides(self):
        with patch("classify_with_jevar.sys.argv", ["tickets", "--column", "Message"]), \
                patch("classify_with_jevar.main") as run:
            tickets_main()
        directory = Path(__file__).resolve().parent
        run.assert_called_once_with(ticket_arguments(directory) + ["--column", "Message"])

    def test_load_config_rejects_invalid_shapes_labels_and_instruction(self):
        invalid = [None, [], "config", 42, {}]
        invalid.extend({"instruction": value, "routes": ROUTES}
                       for value in (None, "", "  ", 1, [], {}))
        invalid.extend({"instruction": INSTRUCTION, "routes": value}
                       for value in (None, [], "routes", {}, {"one": "Only route"},
                                     {"": "Empty label", "ok": "Valid"},
                                     {"  ": "Blank label", "ok": "Valid"},
                                     {"one": None, "two": "Valid"},
                                     {"one": 7, "two": "Valid"}))
        path = self.directory / "config.json"
        for config in invalid:
            with self.subTest(config=config):
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_predictions_preserve_workbook_and_cache_repeated_text(self):
        rows = [["Text", "ID", "Formula"],
                ["hello", 1, "=B2+1"],
                [" hello ", 2, "=B3+1"],
                ["uncertain", 3, "=B4+1"],
                ["one [MASK] two three", 4, "=B5+1"]]
        self.save_workbook(rows)
        original_bytes = self.source.read_bytes()
        summary = self.classify()
        self.assertEqual(self.source.read_bytes(), original_bytes)
        book = self.open_book(self.output)
        sheet = book.active
        self.assertEqual(book.sheetnames, ["Input", "Other"])
        self.assertEqual(list(sheet.iter_rows(min_col=1, max_col=3, values_only=True)),
                         [tuple(row) for row in rows])
        self.assertEqual(list(book["Other"].values), [("Keep me", "=1+2")])
        self.assertEqual(sheet["C2"].data_type, "f")
        self.assertEqual(book["Other"]["B1"].data_type, "f")
        self.assertEqual([cell.value for cell in sheet[1]][3:],
                         ["القسم المسؤول", "درجة الثقة"])
        for row in (2, 3, 5):
            self.assertEqual(sheet.cell(row, 4).value, "قبول")
            self.assertEqual(sheet.cell(row, 5).value, 0.9)
            self.assertEqual(sheet.cell(row, 5).number_format, "0.00%")
        self.assertEqual(sheet["D4"].value, "مراجعة")
        self.assertEqual(sheet["E4"].value, 0.55)
        self.assertEqual(self.router.encode_calls, [("", INSTRUCTION, ROUTES)])
        self.assertEqual(self.router.calls,
                         [(text, ROUTES, INSTRUCTION)
                          for text in ("hello", "uncertain", "one [MASK] two three")])
        self.assertEqual(self.router.tok.calls,
                         [("hello", False), ("uncertain", False),
                          ("one   two three", False)])
        self.assertEqual(summary, {"output": str(self.output), "sheet": "Input",
                                   "counts": {"قبول": 3, "مراجعة": 1}, "needs_review": 1})

    def test_existing_department_and_confidence_columns_are_filled_without_extras(self):
        self.save_workbook([["Text", "درجة الثقة", "Keep", "القسم المسؤول"],
                            ["hello", 0.1, "=1+2", "Old department"],
                            [None, 0.8, "Keep me", "Old department"]])
        source_bytes = self.source.read_bytes()
        self.classify()
        sheet = self.open_book(self.output).active
        self.assertEqual(sheet.max_column, 4)
        self.assertEqual(list(sheet.values),
                         [("Text", "درجة الثقة", "Keep", "القسم المسؤول"),
                          ("hello", 0.9, "=1+2", "قبول"),
                          (None, None, "Keep me", None)])
        self.assertEqual(sheet["D2"].data_type, "s")
        self.assertEqual(sheet["B2"].number_format, "0.00%")
        self.assertEqual(self.source.read_bytes(), source_bytes)

    def test_duplicate_output_headers_rejected(self):
        for header in ("القسم المسؤول", "درجة الثقة"):
            with self.subTest(header=header):
                self.save_workbook([["Text", header, header], ["hello"]])
                with self.assertRaisesRegex(ValueError, "at most one header"):
                    self.classify()
                self.assertFalse(self.output.exists())

    def test_removed_columns_and_prediction_formatting(self):
        self.save_workbook([
            ["نوع الطلب", "Text", "القسم المسؤول", "الأولوية", "الإجراء التالي", "درجة الثقة", "Keep"],
            ["Remove", "hello", None, "Remove", "Remove", None, "Keep me"],
            ["Remove", None, None, "Remove", "Remove", None, "Keep this too"],
        ])
        source = self.open_book(self.source)
        sheet = source.active
        sheet.sheet_view.rightToLeft = True
        sheet.row_dimensions[2].height = 35
        for row in range(1, 4):
            cell = sheet.cell(row, 2)
            cell.font = Font(name="Sakkal Majalla", size=20, bold=row == 1)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.fill = PatternFill("solid", fgColor="336699" if row == 1 else "EEEEEE")
            cell.border = Border(bottom=Side(style="thin", color="000000"))
        for column, width in zip("ABCDEFG", (10, 52, 20, 11, 28, 14, 22)):
            sheet.column_dimensions[column].width = width
        source.save(self.source)
        original_bytes = self.source.read_bytes()
        self.classify()
        result = self.open_book(self.output).active
        self.assertEqual(list(result.values), [
            ("Text", "القسم المسؤول", "درجة الثقة", "Keep"),
            ("hello", "قبول", 0.9, "Keep me"),
            (None, None, None, "Keep this too"),
        ])
        self.assertEqual([result.column_dimensions[c].width for c in "ABCD"],
                         [52, 20, 14, 22])
        self.assertTrue(result.sheet_view.rightToLeft)
        self.assertEqual(result.row_dimensions[2].height, 35)
        for row in range(1, 4):
            for column in (2, 3):
                expected = copy(sheet.cell(row, 2)._style)
                if row > 1 and column == 3:
                    expected.numFmtId = result.cell(row, column)._style.numFmtId
                    self.assertEqual(result.cell(row, column).number_format, "0.00%")
                self.assertEqual(result.cell(row, column)._style, expected)
                self.assertEqual(result.cell(row, column).font.name, "Sakkal Majalla")
        self.assertEqual(self.source.read_bytes(), original_bytes)
        self.assertEqual(list(self.open_book(self.output)["Other"].values),
                         [("Keep me", "=1+2")])

    def test_live_progress_for_predictions_cached_rows_and_skips(self):
        self.save_workbook([["Text"], ["hello"], ["hello"], ["uncertain"],
                            [None], ["one two three four"]])
        progress = io.StringIO()
        route = self.router.route

        def check_processing(text, routes, instruction):
            self.assertTrue(progress.getvalue().splitlines()[-1].endswith("Processing..."))
            return route(text, routes, instruction)

        with redirect_stderr(progress), patch.object(self.router, "route", side_effect=check_processing):
            self.classify()
        lines = progress.getvalue().splitlines()
        self.assertEqual(len(lines), 10)
        for index in range(5):
            self.assertEqual(lines[index * 2],
                             f"[{index + 1}/5] Ticket (Excel row {index + 2}): Processing...")
        self.assertIn("Completed — قبول (90.00%)", lines[1])
        self.assertIn("Completed — قبول (90.00%)", lines[3])
        self.assertIn("Completed — مراجعة (55.00%) — Needs review: Low model confidence", lines[5])
        self.assertIn("Skipped — Empty text", lines[7])
        self.assertIn("Skipped — Text exceeds 3-token budget; not classified", lines[9])
        self.assertEqual(len(self.router.calls), 2)

    def test_invalid_cells_are_unclassified_and_flagged(self):
        self.save_workbook([["Text", "ID"], [None, 1], ["   ", 2],
                            [123, 3], [True, 4], ["=1+2", 5],
                            ["one two three four", 6], ["one two three four", 7]])
        progress = io.StringIO()
        with redirect_stderr(progress):
            summary = self.classify()
        sheet = self.open_book(self.output).active
        reasons = ["Empty text", "Empty text", "Non-text cell", "Non-text cell",
                   "Formula cell; provide plain text",
                   "Text exceeds 3-token budget; not classified",
                   "Text exceeds 3-token budget; not classified"]
        for row, reason in enumerate(reasons, 2):
            with self.subTest(row=row):
                self.assertEqual([sheet.cell(row, col).value for col in (3, 4)],
                                 [None, None])
                self.assertIn(f"Skipped — {reason}", progress.getvalue())
        self.assertEqual(sheet.max_column, 4)
        self.assertEqual(sheet["A6"].value, "=1+2")
        self.assertEqual(sheet["A6"].data_type, "f")
        self.assertEqual(self.router.calls, [])
        self.assertEqual(self.router.tok.calls, [("one two three four", False)])
        self.assertEqual(summary["counts"], {"Not classified": 7})
        self.assertEqual(summary["needs_review"], 7)

    def test_duplicate_or_missing_input_header_rejected_without_output(self):
        for headers in (["Text", "Text"], ["Other", "ID"]):
            with self.subTest(headers=headers):
                self.save_workbook([headers, ["hello", 1]])
                with self.assertRaisesRegex(ValueError, "exactly one header"):
                    self.classify()
                self.assertFalse(self.output.exists())
                self.assertEqual(self.router.calls, [])

    def test_existing_output_and_source_are_never_overwritten(self):
        self.save_workbook([["Text"], ["hello"]])
        self.output.write_bytes(b"Existing output must remain untouched")
        original_bytes = self.source.read_bytes()
        with self.assertRaises(FileExistsError):
            self.classify()
        self.assertEqual(self.output.read_bytes(), b"Existing output must remain untouched")
        with self.assertRaises(FileExistsError):
            classify_workbook(self.source, self.source, "Text", INSTRUCTION,
                              ROUTES, self.router)
        self.assertEqual(self.source.read_bytes(), original_bytes)
        self.assertEqual(self.router.encode_calls, [])
        self.assertEqual(self.router.calls, [])

    def test_no_token_budget_rejected_without_output(self):
        self.save_workbook([["Text"], ["hello"]])
        self.router.cfg["max_len"] = 2
        with self.assertRaisesRegex(ValueError, "no room"):
            self.classify()
        self.assertFalse(self.output.exists())
        self.assertEqual(self.router.calls, [])


if __name__ == "__main__":
    unittest.main()
