# Reusable Jev-AR classifier

`classify_with_jevar.py` runs the local model directly. It does not use the AI-specific usefulness rules or saved reviews from the earlier scripts.

## Run

For the included `raw.xlsx` customer tickets, run from this folder:

```sh
uv run tickets
```

This uses the `نص الرسالة` column and the current `ticket-routing.json` beside
the classifier. Edit that JSON to change the question or teams for the next run.
Results go to `tickets-routed.xlsx`, or `tickets-routed-2.xlsx`,
`tickets-routed-3.xlsx`, etc. if earlier outputs exist. Predictions fill
`القسم المسؤول` with the plain route name and `درجة الثقة` with a percentage.
The ticket config uses Arabic department names. Generated workbooks omit
`نوع الطلب`, `الأولوية`, and `الإجراء التالي`. Use `uv run tickets --help` for options;
`--column`, `--config`, and `--model` can override the defaults.

Before classification, stderr prints `Loading Jev Arabic Model`, then `Model Loaded`
once initialization succeeds. During classification, each ticket prints `Processing...`
followed by its completed prediction, confidence, and any review flag. Skipped tickets show their reason;
cached predictions also show completion. Progress includes the ticket count and
Excel row number, but not customer message text. Progress goes to stderr, leaving
the final JSON summary on stdout. The workbook is saved only after all rows finish.

Customer tickets, with a `Message` column on the workbook's active sheet:

```sh
uv run classify_with_jevar.py tickets.xlsx tickets-routed.xlsx --column Message --config ticket-routing.json
```

Customer feedback, with a `Comment` column:

```sh
uv run classify_with_jevar.py comments.xlsx comments-classified.xlsx --column Comment --config customer-comments.json
```

To use the existing workbook with the customer-comment categories:

```sh
uv run classify_with_jevar.py ../GroupStudy/interaction-raw.xlsx interaction-comment-types.xlsx --column ZTEXT --config customer-comments.json
```

The model defaults to `/Volumes/WD/LLMs/jevar`. Set `--model /path/to/jevar` to change it. That folder must include `jev_ar_route.py`, the tokenizer, configuration, and weights.

## Create another workflow

Copy an example JSON file and change:

- `instruction`: the classification question.
- `routes`: output labels mapped to short descriptions. Supply at least two.

No code change or retraining is needed. Keep labels distinct and descriptions short; the model validates its option-token limits. The classifier chooses one label per row, not multiple labels.

## Output and limits

The new workbook fills `القسم المسؤول` and `درجة الثقة` on the active sheet,
adding those columns only if missing. Existing values in these two columns are
replaced in the new workbook; other columns and sheets are preserved. No extra
Jev columns or JSON probability cells are generated. Source files and previous
outputs are never overwritten or retroactively changed.

The three columns `نوع الطلب`, `الأولوية`, and `الإجراء التالي` are removed from
the output's active sheet. Retained column widths move with their columns.
Prediction headers and cells copy the message column's row-specific formatting
(including Sakkal Majalla font, size, alignment, wrapping, fills, and borders).
Confidence cells use a percentage number format. Row heights and sheet direction
are retained.

Empty, non-text, formula, and overlength inputs leave both output cells blank and
show a skip reason in terminal progress. Low-confidence review warnings appear in
the terminal, with a review count in the final summary, not as extra workbook
columns. Retain the terminal log if those warnings are needed later.
Long text is not silently truncated or combined across chunks: different workflows
require different aggregation rules. Shorten the input or implement a
workflow-specific long-text policy.

The model's built-in confidence threshold determines low-confidence flags. Confidence is not measured accuracy for your workflow. Test representative examples before using predictions to assign real tickets. Links are treated as text; their contents are not fetched. This script only labels a workbook; it does not send or assign tickets.

All model inference stays local. Dependency installation may need network access. The model's CC BY-NC 4.0 license requires separate permission for commercial use, such as a commercial customer-support service.
