# polars-schema-index

**A Polars plugin for flattening nested columns with stable numeric indexing.**

`polars-schema-index` provides a systematic way to explode/unnest nested Polars DataFrames (does not yet support LazyFrames) without overwriting columns that share the same name. It achieves this by:

- Attaching a custom `schema_index` namespace to your DataFrame.
- Renaming columns that do not end in digits with a numbered suffix.
- Iteratively flattening `Struct` columns (and optionally exploding `list[struct]` columns first), so every nested field becomes a separate top-level column.

## Installation

```bash
pip install polars-schema-index[polars]
```

On older CPUs run:

```python
pip install polars-schema-index[polars-lts-cpu]
```

## Usage

```python
import polars as pl
from polars_schema_index import flatten_nested_data

# Example: flatten a deeply nested JSON structure
df = pl.read_ndjson(
    source=b'''{
        "body": [
            {
                "type": "If",
                "test": {
                    "type": "Compare",
                    "left": {
                        "type": "Name",
                        "id": "x",
                        "ctx": { "type": "Load" }
                    },
                    "ops": [{ "type": "IsNot" }],
                    "comparators": [{ "type": "Constant", "value": null }]
                },
                "body": [{ "type": "Pass" }],
                "orelse": []
            }
        ],
        "type_ignores": []
    }
    '''.replace(b"\n", b"")
)
flattened = flatten_nested_data(df)
print(flattened)
```

This gives a DataFrame with all nested fields expanded into uniquely suffixed, monotonically
increasing numbered columns:

```python
┌────────────────┬────────┬────────────┬─────────┬───┬─────────┬──────────┬──────────┬─────────┐
│ type_ignores_1 ┆ type_2 ┆ orelse_5   ┆ type_6  ┆ … ┆ type_14 ┆ type_15  ┆ value_16 ┆ type_17 │
│ ---            ┆ ---    ┆ ---        ┆ ---     ┆   ┆ ---     ┆ ---      ┆ ---      ┆ ---     │
│ list[null]     ┆ str    ┆ list[null] ┆ str     ┆   ┆ str     ┆ str      ┆ null     ┆ str     │
╞════════════════╪════════╪════════════╪═════════╪═══╪═════════╪══════════╪══════════╪═════════╡
│ []             ┆ If     ┆ []         ┆ Compare ┆ … ┆ IsNot   ┆ Constant ┆ null     ┆ Load    │
└────────────────┴────────┴────────────┴─────────┴───┴─────────┴──────────┴──────────┴─────────┘
```

### What It Solves

- **No more silent overwrites** of common keys (like `"type"`) when unnesting.
- **Stable numeric suffixes** for each column, so even if you run multiple flatten passes, names remain unique.
- **Optional exploding of list-of-struct columns** before flattening them.

### Key Functions

1. **`flatten_nested_data(df, explode_lists=True, max_passes=1000)`**
   Iteratively flattens all `Struct` columns in a DataFrame or LazyFrame, and explodes any `list[struct]` columns (if `explode_lists=True`). Continues until no `Struct` columns remain (or `max_passes` is reached).

2. **`df.schema_index.append_unnest_relabel(df, column=...)`**
   Moves one column to the end via `.permute`, unnest it, then relabel newly created columns with numeric suffixes.

### Note

- **Column Renaming**: The library appends numeric suffixes to *all columns* that lack them, even if they are already scalar columns. That ensures flattening never creates collisions, but it does mean your top-level columns will also gain suffixes.
- **LazyFrame Support**: By default, the plugin is registered for `DataFrame`. If you want to use this on LazyFrames, you can register a similar namespace for `LazyFrame` or manually attach the plugin’s logic. I may end up supporting both.

## Contributing

1. **Issues & Discussions**: Please open a GitHub issue for bugs, feature requests, or questions.
2. **Pull Requests**: PRs are welcome! Add tests under `tests/`, update the docs, and ensure you run `pytest` locally.

## License

This project is licensed under the MIT License.
