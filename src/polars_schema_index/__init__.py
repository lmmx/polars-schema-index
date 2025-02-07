import polars as pl
import polars.selectors as cs
import polars_permute  # noqa: F401
from polars.api import register_dataframe_namespace

try:
    from pysnooper import snoop
except ImportError:

    def snoop(func):
        """No-op decorator if pysnooper is not available"""
        return func


__all__ = ("flatten_nested_data",)


@register_dataframe_namespace("schema_index")
class SchemaIndexPlugin:
    def __init__(self, df: pl.DataFrame, verbose: bool = False):
        self._df = df
        self._verbose = verbose
        self._index_size: int = 0

    @property
    def index_size(self) -> int | None:
        """Get or set the schema index size."""
        return self._index_size

    @index_size.setter
    def index_size(self, value: int | None) -> None:
        self._index_size = value

    # ----------------------------------------------------------------------
    # 1) Collect schema and schema_size, and build schema_df
    # ----------------------------------------------------------------------
    def collect_schema(self) -> dict[str, pl.DataType]:
        """Collect the schema from the DataFrame or LazyFrame."""
        return self._df.collect_schema()

    @property
    def schema_size(self) -> int:
        """Return the size of the collected schema."""
        return len(self.collect_schema())

    @property
    def schema_df(self) -> pl.DataFrame:
        """Return a DataFrame representation of the schema (column -> type)."""
        return pl.DataFrame(schema=self.collect_schema())

    # ----------------------------------------------------------------------
    # 2) Split columns into numbered and unnumbered
    # ----------------------------------------------------------------------
    @property
    def numbered_cols(self) -> pl.DataFrame:
        """Return a DataFrame of columns that end with digits."""
        schema = self.schema_df
        if self._verbose:
            print("Loaded schema:", schema)
        numbered = schema.select(cs.ends_with(*"0123456789"))
        if self._verbose:
            print("Found numbered:", numbered)
        return numbered

    @property
    def unnumbered_cols(self) -> pl.DataFrame:
        """Return a DataFrame of columns that do not end with digits."""
        return self.schema_df.select(~cs.ends_with(*"0123456789"))

    # ----------------------------------------------------------------------
    # 3) Extract integer suffixes and return their maximum
    # ----------------------------------------------------------------------
    @property
    def idx_max(self) -> int:
        """
        Extract digit suffixes from numbered columns and return
        the largest integer found.
        """
        if not self.numbered_cols.columns:
            return -1
        int_idxs = (
            pl.Series(self.numbered_cols.columns)
            .str.extract(r"_(\d+)$")
            .drop_nulls()
            .cast(pl.Int64)
        )
        if self._verbose:
            print("Computed max:", int_idxs.max())
        return int_idxs.max()

    # ----------------------------------------------------------------------
    # 4) Compute the offset
    # ----------------------------------------------------------------------
    @property
    def idx_offset(self) -> int:
        """
        Increment the highest suffix present on existing columns.
        """
        return self.idx_max + 1

    # ----------------------------------------------------------------------
    # 5) Build flat_index for unnumbered columns
    # ----------------------------------------------------------------------
    @property
    @snoop()
    def flat_index(self) -> dict[str, str]:
        """
        Map each unnumbered column name to a suffix-appended name,
        using the idx_offset to avoid collisions.
        """
        unnumbered = self.unnumbered_cols.columns

        return {
            col_name: (
                col_name
                if col_name[-1].isnumeric()
                else f"{col_name}_{idx + self.idx_offset}"
            )
            for idx, col_name in enumerate(unnumbered)
        }

    def rename_new_columns_with_gaps(
        self,
        data: pl.LazyFrame | pl.DataFrame,
        suffix_pattern: str = r"_(\d+)$",
    ):
        """
        Rename any columns that do not already end with digits to have
        a suffix that is strictly larger than the largest seen so far.

        This avoids collisions or silent overwrites after unnesting.
        Note: always unnest a single column at a time to avoid collisions.
        """
        if self._verbose:
            print("Renaming with index:", data.schema_index.flat_index)
        return data.rename(data.schema_index.flat_index)

    def append_unnest_relabel(
        self,
        data: pl.LazyFrame | pl.DataFrame,
        *,
        column: str | pl.Expr,
        suffix_pattern: str = r"_(\d+)$",
    ):
        appended_and_unnested = data.permute.append(column).unnest(column)
        return self.rename_new_columns_with_gaps(appended_and_unnested)


@snoop()
def flatten_nested_data(
    df: pl.DataFrame | pl.LazyFrame,
    explode_lists: bool = True,
    max_passes: int = 1000,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Iteratively flatten all nested columns (structs) in `df`. If `explode_lists=True`,
    also explode list columns before unnesting them (which often helps flatten
    “list[struct]” columns).

    This uses a column-renaming strategy (via schema_index) to guarantee that
    newly unnested columns will never silently overwrite existing ones.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        The (possibly nested) Polars object to flatten.
    explode_lists : bool
        If True, list columns are exploded before attempting to unnest them.
        If False, list columns remain as-is, unless you unnest them manually.
    max_passes : int
        Safety limit to avoid accidental infinite loops if something is pathological.

    Returns
    -------
    A new DataFrame or LazyFrame with all struct columns flattened. Columns are renamed
    with suffixes to avoid collisions. The iteration continues until there are no struct
    columns left (or until `max_passes` is reached).
    """

    # # Ensure we have the schema_index plugin “attached”
    # # (We only need to do this once, so we can consistently rename as we go.)
    # if not hasattr(df, "schema_index"):
    #     df = df._clone()  # any safe copy will do, just to attach

    # We'll iterate up to `max_passes` times or until no nested columns remain.
    passes = 0
    while passes < max_passes:
        passes += 1

        # (1) rename columns that do not have numeric suffixes yet
        df = df.schema_index.rename_new_columns_with_gaps(df)

        # (2) find struct columns and/or list-of-struct columns
        schema = df.collect_schema()  # or df.schema if DF
        struct_cols = []
        list_struct_cols = []
        for c, dt in schema.items():
            if isinstance(dt, pl.datatypes.Struct):
                struct_cols.append(c)
            elif isinstance(dt, pl.datatypes.List):
                inner = dt.inner
                if isinstance(inner, pl.datatypes.Struct) and explode_lists:
                    struct_cols.append(c)
                    list_struct_cols.append(c)

        # If nothing to flatten, break early
        if not struct_cols and not list_struct_cols:
            break

        # (3) For list[struct] columns, optionally explode them (row-wise)
        #     so that they become a struct column, then flatten as normal.
        if explode_lists:
            for col in list_struct_cols:
                # move this column to the end (so unnest expansions won't collide)
                df = df.explode(col)

        # (4) Unnest each struct column
        #     push each to the end first, then unnest
        #     rename with gaps afterwards to keep consistent naming
        for col in struct_cols:
            df = df.schema_index.append_unnest_relabel(df, column=col)

    return df
