# GroupStudy

WhatsApp interaction classification and usefulness review tools.

- `GroupStudy.py`: CSV classifier using the configured AI provider. Input messages are sent to that provider when run.
- `classify_interactions.py`: local Jev-AR workbook classifier.
- `apply_usefulness_review.py`: applies local review selections to classified workbooks.
- `leaderboarded-theme.css`: reusable theme styling.

Excel files, CSV exports, review JSON, caches, logs, identifying HTML reports,
and credentials remain local and are excluded by `.gitignore`.
The scripts require those local input files; this repository does not include them.

Review new files for identifying content before adding them: `.gitignore` matches
filenames, not personal information inside arbitrary files.
