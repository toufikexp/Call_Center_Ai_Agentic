# Runbook — Updating the classification taxonomy

The taxonomy lives in `src/config/classification_schema.json`. No code changes
are required for adding/removing/renaming categories or sub-categories.

## What lives in the schema

```jsonc
{
  "primary_categories":       [ "...", ... ],
  "category_descriptions":    { "<category>": "<one-line description>" },
  "category_subcategories":   { "<category>": [ "<subcat>", ... ] },
  "subcategory_descriptions": { "<category>": { "<subcat>": "<one-line description>" } },
  "default_subcategory":      { "<category>": "<subcat fallback>" }
}
```

All keys must reference categories that appear in `primary_categories`.
`OTHER` should remain in `primary_categories` — it's the safe default the
classifier falls back to.

## Add a new primary category

1. Append the category name to `primary_categories`.
2. Add a description under `category_descriptions`.
3. Add an entry under `category_subcategories` with at least one sub-category
   (or `["N/A"]` if there are no meaningful sub-categories).
4. Add subcategory descriptions under `subcategory_descriptions`.
5. Add a `default_subcategory` mapping for the new category.
6. Re-run a small batch and inspect classifier behavior. The new category
   description is included verbatim in the classification prompt — keep it
   short and unambiguous.

## Rename a category or sub-category

Be aware: existing JSON files in `data/results/` retain the old name. If you
need historical consistency, write a migration in `scripts/` (none exist
today) to rewrite past records.

1. Update every occurrence in the schema:
   - `primary_categories`
   - `category_descriptions` (key)
   - `category_subcategories` (key + values inside)
   - `subcategory_descriptions` (key + inner keys)
   - `default_subcategory` (key + value)
2. Re-run a smoke test.

## Remove a category

1. Delete it from all sections of the schema.
2. Make sure no test fixture or downstream consumer references the old name.
3. The classifier will no longer be allowed to choose it; existing JSON
   results are unaffected.

## Validation

The loader (`Settings._load_classification_schema`) silently falls back to
`{}` on parse error. To catch typos early, validate the JSON after editing:

```bash
python -c "import json; json.load(open('src/config/classification_schema.json')); print('OK')"
```

Then run a single-call smoke test and check `data/results/<file>.json` for a
sensible `subject` / `sub_subject` pair.

## Notes

- Subcategory matching is case-insensitive and tolerant of whitespace around
  dashes (see `normalize_subcat` in `classification.py`). Slight punctuation
  drift in the schema won't break matching.
- The classifier prompt is rebuilt from the schema on every call — restarts
  are not required, but a process running an in-memory `Settings` will need a
  fresh `get_settings()` call to see the new schema. Easiest: restart the
  process.
