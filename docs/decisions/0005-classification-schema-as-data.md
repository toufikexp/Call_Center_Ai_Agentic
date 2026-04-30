# ADR 0005 — Classification taxonomy as JSON data, not Python code

- Status: Accepted

## Context

The taxonomy (primary categories, sub-categories, descriptions, defaults) is
business data that changes at a different cadence from the code. Encoding it
as Python literals in `config.py` mixes business and engineering concerns and
makes diff-review of taxonomy changes painful.

## Decision

The taxonomy lives in `src/config/classification_schema.json`. At startup,
`Settings.create_default()` calls `_load_classification_schema()` which reads
the JSON and unpacks it into a `ClassificationSettings` model.

The schema path can be overridden with `CLASSIFICATION_SCHEMA_PATH` for
experimentation.

JSON shape (top-level keys):

```
primary_categories         : [str, ...]
category_descriptions      : { category: str }
category_subcategories     : { category: [subcat, ...] }
subcategory_descriptions   : { category: { subcat: str } }
default_subcategory        : { category: subcat }
```

## Consequences

- **Pro:** non-engineers can review and edit the taxonomy without touching
  Python.
- **Pro:** alternative taxonomies (e.g. for a different operator) are a
  drop-in replacement via `CLASSIFICATION_SCHEMA_PATH`.
- **Pro:** clear contract — `ClassificationSettings` is the only thing
  services ever consume.
- **Con:** typos in JSON aren't caught until startup. `create_default()`
  silently falls back to defaults if the JSON is malformed (see
  `_load_classification_schema`). Be careful editing.
