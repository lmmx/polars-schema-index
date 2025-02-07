import polars as pl
import pytest

from polars_schema_index import flatten_nested_data


def test_basic_struct_flatten():
    """Test flattening a single struct column."""
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "nested": [
                {"foo": 10, "bar": 100},
                {"foo": 20, "bar": 200},
            ],
        },
    )
    result = flatten_nested_data(df)

    # Expect top-level columns with numeric suffix
    # e.g. 'id', 'foo_0', 'bar_1' or something similar.
    # Check the total number of columns and validate data.
    assert result.shape == (2, 3), "Flattened struct should expand from 2 -> 3 columns"
    assert [
        "id_0",
        "foo_1",
        "bar_2",
    ] == result.columns, "Missing expected 'foo' or 'bar' columns after flattening"
    assert result.select(pl.col("*")).to_dicts() == [
        {"id_0": 1, "foo_1": 10, "bar_2": 100},
        {"id_0": 2, "foo_1": 20, "bar_2": 200},
    ], "Data integrity check failed after flattening struct"


def test_multi_level_struct():
    """Test flattening a multi-level nested struct."""
    df = pl.DataFrame(
        {"level1": [{"x": 1, "level_two": {"y": 10, "level_three": {"z": 100}}}]},
    )
    result = flatten_nested_data(df)
    # Should expand out all nested levels
    # For 1 row, we expect at least 3 columns: x_#, y_#, z_#
    assert result.shape[0] == 1, "Row count should remain the same (just 1 row)"
    assert len(result.columns) >= 3, "Should have at least 3 columns after flattening"
    # Validate final data
    row = result.select(pl.all()).to_dicts()[0]
    # The precise column names will vary depending on suffix assignment,
    # but we expect the values 1, 10, and 100 in separate columns.
    vals = list(row.values())
    assert (
        1 in vals and 10 in vals and 100 in vals
    ), "Missing data from multi-level flatten"


def test_list_of_structs():
    """Test flattening a list[struct] column (explode first, then flatten)."""
    df = pl.DataFrame(
        {
            "row_id": [1],
            "logs": [
                [
                    {"event": "login", "timestamp": 1000},
                    {"event": "logout", "timestamp": 2000},
                ],
            ],
        },
    )
    result = flatten_nested_data(df, explode_lists=True)
    # We started with 1 row, but exploding the list of 2 events -> 2 rows
    assert result.shape[0] == 2, "Exploding list[struct] should produce 2 rows"
    # Expect at least row_id + 2 columns for event/timestamp
    assert result.shape[1] >= 3, "Should have row_id plus flattened event/timestamp"
    # Check data
    assert result.to_dicts() == [
        {"row_id_0": 1, "event_1": "login", "timestamp_2": 1000},
        {"row_id_0": 1, "event_1": "logout", "timestamp_2": 2000},
    ]


def test_no_struct_columns():
    """Test a DataFrame with no struct columns (no change expected)."""
    df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
    result = flatten_nested_data(df)
    assert (
        result.shape == df.shape
    ), "Shape should remain unchanged with no struct columns"
    assert result.columns == ["x_0", "y_1"]
    assert result.to_dicts() == [{"x_0": 1, "y_1": 3}, {"x_0": 2, "y_1": 4}]


@pytest.mark.xfail(reason="LazyFrame won't work yet")
def test_flatten_lazyframe():
    """Test flattening on a LazyFrame."""
    lf = pl.DataFrame(
        {
            "session": [42, 43],
            "nested": [
                {"key": "alpha", "val": 10},
                {"key": "beta", "val": 20},
            ],
        },
    ).lazy()
    # Flatten
    result_lf = flatten_nested_data(lf)
    # Collect and validate
    result = result_lf.collect()
    # Expect columns session, plus 2 columns for nested
    assert result.shape[1] == 3, "LazyFrame flattening didn't produce 3 columns"
    row_dicts = result.to_dicts()
    assert row_dicts == [
        {"session": 42, "nested_0": "alpha", "nested_1": 10},
        {"session": 43, "nested_0": "beta", "nested_1": 20},
    ] or [
        {"session": 42, "key_0": "alpha", "val_1": 10},
        {"session": 43, "key_0": "beta", "val_1": 20},
    ], "LazyFrame data integrity check failed after flattening"


def test_deep_ast_example():
    """
    Test the typical AST-like JSON structure with repeated 'type' keys that we
    flatten. This is similar to the 'if x is not None: pass' example.
    """
    nested_json = b"""
    {
        "body": [{
          "type": "If",
          "test": {
            "type": "Compare",
            "left": {
              "type": "Name",
              "id": "x",
              "ctx": {"type": "Load"}
            },
            "ops": [{"type": "IsNot"}],
            "comparators": [{"type": "Constant", "value": null}]
          },
          "body": [{"type": "Pass"}],
          "orelse": []
        }],
        "type_ignores": []
    }"""

    df = pl.read_ndjson(source=nested_json.replace(b" ", b"").replace(b"\n", b""))
    flattened = flatten_nested_data(df)

    # We expect multiple columns, at least:
    # "body_#" columns for the 'If', 'Compare', 'Name', 'IsNot', 'Constant', etc.
    assert flattened.shape[0] == 1, "Should remain 1 row"
    # Check that we do indeed have multiple columns
    assert flattened.shape[1] > 2, "Should have more than 2 columns after flattening"

    # It's tough to know exact column names because of suffix assignment,
    # but we can check that certain tokens appear somewhere in the final columns
    colnames = " ".join(flattened.columns)
    assert "type" in colnames, "Should still see a 'type' in one of the columns"
    # Check presence of 'Pass', 'If', 'Compare', etc. in the data
    row_data = str(flattened.to_dicts()[0])
    for expected_val in ["If", "Compare", "Name", "IsNot", "Pass", "Constant"]:
        assert (
            expected_val in row_data
        ), f"Missing '{expected_val}' from flattened AST data"
