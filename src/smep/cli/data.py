"""Data fetching and processing CLI commands."""

from pathlib import Path
from typing import Optional
import typer
import logging

from smep.data import (
    get_registry,
    get_exporter_registry,
    get_builder_registry,
    KaggleDownloadError,
)
from smep.data.builders.default import DEFAULT_DROP_COLUMNS

logger = logging.getLogger(__name__)

app = typer.Typer(
    no_args_is_help=True,
    help="Manage data fetching and processing for SMEP datasets.",
)


# ------------------------------------------------------------------
# export  –  raw MIMIC → base table
# ------------------------------------------------------------------


@app.command(name="export")
def export_base_table(
    name: str = typer.Argument(
        ...,
        help="Exporter name (e.g. 'mimic3')",
    ),
    source: Path = typer.Argument(
        ...,
        help="Path to the raw data directory",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Output directory for the base table. "
            "Defaults to <source>_exported."
        ),
    ),
    min_age: int = typer.Option(
        18,
        "--min-age",
        help="Minimum patient age for cohort inclusion.",
    ),
    first_stay_only: bool = typer.Option(
        True,
        "--first-stay-only/--all-stays",
        help="Keep only the first ICU stay per subject.",
    ),
    time_window_hours: int = typer.Option(
        24,
        "--time-window-hours",
        help=(
            "Aggregation window (hours from ICU admission). "
            "Also used as the minimum ICU stay length."
        ),
    ),
    schema_version: str = typer.Option(
        "v1",
        "--schema-version",
        help="Schema version tag written into metadata.",
    ),
) -> None:
    """Export a base table from raw data.

    The base table is an extractedMimic-style wide table that can later
    be consumed by ``smep data build-dataset``.

    Example:
        smep data export mimic3 .data/mimic-iii-clinical-database-demo-1.4
    """
    try:
        source = source.resolve()
        if output is None:
            output = source.parent / f"{source.name}_exported"
        else:
            output = output.resolve()

        registry = get_exporter_registry()

        try:
            exporter = registry.get_exporter(
                name,
                min_age=min_age,
                first_stay_only=first_stay_only,
                time_window_hours=time_window_hours,
                schema_version=schema_version,
            )
        except (KeyError, ValueError) as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

        typer.echo(f"Exporting base table with '{name}' exporter…")
        typer.echo(f"Source: {source}")
        typer.echo(f"Output: {output}")
        typer.echo(f"Time window: {time_window_hours} h")
        typer.echo(f"Min age: {min_age}")
        typer.echo(f"First stay only: {first_stay_only}")

        exporter.export(source, output)
        typer.echo(f"✓ Base table exported to {output}")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logger.exception("Export failed")
        raise typer.Exit(code=1)


# ------------------------------------------------------------------
# build  –  base table → train/val/test dataset
# ------------------------------------------------------------------


@app.command(name="build")
def build_dataset(
    base_table: Path = typer.Argument(
        ...,
        help="Path to base_table.csv or directory containing it.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Output directory for the dataset. "
            "Defaults to <base_table_dir>_dataset."
        ),
    ),
    label: str = typer.Option(
        "hospital_expire_flag",
        "--label",
        help="Label column name.",
    ),
    split: str = typer.Option(
        "0.8,0.1,0.1",
        "--split",
        help="Train/val/test ratios, comma-separated (must sum to 1.0).",
    ),
    random_state: int = typer.Option(
        42,
        "--random-state",
        help="Random seed for reproducibility.",
    ),
    stratify: bool = typer.Option(
        True,
        "--stratify/--no-stratify",
        help="Stratify splits by label.",
    ),
    imputer: str = typer.Option(
        "median",
        "--imputer",
        help="Imputation strategy: median, mean, knn, none.",
    ),
    scaler: str = typer.Option(
        "standard",
        "--scaler",
        help="Scaling strategy: standard, minmax, robust, none.",
    ),
    min_coverage: float = typer.Option(
        0.0,
        "--min-coverage",
        help="Minimum non-missing rate (0-1) to keep a column.",
    ),
    drop_columns: Optional[str] = typer.Option(
        None,
        "--drop-columns",
        help="Comma-separated column names to exclude.",
    ),
    keep_columns: Optional[str] = typer.Option(
        None,
        "--keep-columns",
        help="Comma-separated column names to keep (mutually exclusive with --drop-columns).",
    ),
) -> None:
    """Build a training dataset from a base table.

    Reads a base_table.csv (produced by ``smep data export``), applies
    column selection, imputation, encoding, scaling, and stratified
    splitting to produce train/val/test CSV files.

    Example:
        smep data build ./exported -o ./dataset
    """
    try:
        base_table = base_table.resolve()
        if output is None:
            parent = base_table if base_table.is_dir() else base_table.parent
            output = parent.parent / f"{parent.name}_dataset"
        else:
            output = output.resolve()

        # Parse split ratios
        try:
            parts = [float(x.strip()) for x in split.split(",")]
        except ValueError:
            typer.echo(
                "Error: --split must be three comma-separated floats", err=True
            )
            raise typer.Exit(code=1)

        if len(parts) != 3:
            typer.echo(
                "Error: --split must have exactly 3 values (train,val,test)",
                err=True,
            )
            raise typer.Exit(code=1)

        if abs(sum(parts) - 1.0) > 0.001:
            typer.echo(
                f"Error: --split values must sum to 1.0 (got {sum(parts):.4f})",
                err=True,
            )
            raise typer.Exit(code=1)

        split_tuple = (parts[0], parts[1], parts[2])

        # Validate imputer/scaler choices
        valid_imputers = {"median", "mean", "knn", "none"}
        if imputer not in valid_imputers:
            typer.echo(
                f"Error: --imputer must be one of {valid_imputers}", err=True
            )
            raise typer.Exit(code=1)

        valid_scalers = {"standard", "minmax", "robust", "none"}
        if scaler not in valid_scalers:
            typer.echo(
                f"Error: --scaler must be one of {valid_scalers}", err=True
            )
            raise typer.Exit(code=1)

        # Mutually exclusive
        if drop_columns is not None and keep_columns is not None:
            typer.echo(
                "Error: --drop-columns and --keep-columns are mutually exclusive",
                err=True,
            )
            raise typer.Exit(code=1)

        drop_list = (
            [c.strip() for c in drop_columns.split(",")]
            if drop_columns
            else DEFAULT_DROP_COLUMNS
        )
        keep_list = (
            [c.strip() for c in keep_columns.split(",")]
            if keep_columns
            else None
        )

        registry = get_builder_registry()
        builder = registry.get_builder(
            "default",
            label=label,
            split=split_tuple,
            random_state=random_state,
            stratify=stratify,
            imputer=imputer,
            scaler=scaler,
            min_coverage=min_coverage,
            drop_columns=drop_list,
            keep_columns=keep_list,
        )

        typer.echo("Building dataset…")
        typer.echo(f"Source: {base_table}")
        typer.echo(f"Output: {output}")
        typer.echo(f"Label: {label}")
        typer.echo(f"Split: {split}")
        typer.echo(f"Imputer: {imputer}  Scaler: {scaler}")

        builder.build(base_table, output)
        typer.echo(f"✓ Dataset built at {output}")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logger.exception("Build failed")
        raise typer.Exit(code=1)


@app.command()
def fetch(
    name: str = typer.Argument(
        ..., help="Name of the data source to fetch (e.g., 'mimic3-demo')"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for downloaded data. Defaults to current working directory.",
    ),
) -> None:
    """Fetch data from a specified source.

    Example:
        smep data fetch mimic3-demo
        smep data fetch mimic3-demo --output /path/to/data
    """
    try:
        # Use current working directory if output not specified
        if output is None:
            output = Path.cwd()

        output = output.resolve()

        # Get the fetcher from registry
        registry = get_registry()

        try:
            fetcher = registry.get_fetcher(name)
        except KeyError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

        # Fetch the data
        typer.echo(f"Fetching {name} dataset...")
        typer.echo(f"Output directory: {output}")

        try:
            fetcher.fetch(output)
            typer.echo(f"✓ Successfully downloaded {name} dataset to {output}")
        except KaggleDownloadError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        except RuntimeError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        except Exception as e:
            typer.echo(f"Unexpected error: {e}", err=True)
            logger.exception("Unexpected error during fetch")
            raise typer.Exit(code=1)

    except typer.Exit:
        raise


@app.command()
def list_sources() -> None:
    """List all available data sources.

    Example:
        smep data list-sources
    """
    try:
        registry = get_registry()
        fetchers = registry.get_all_fetcher_info()

        if not fetchers:
            typer.echo("No data sources available.")
            return

        # Display fetchers
        typer.echo("\nAvailable Data Sources:")
        for fetcher_info in fetchers:
            typer.echo(
                f"  • {fetcher_info['name']:<20} - {fetcher_info['description']}"
            )

        typer.echo("\nUse 'smep data fetch <name>' to download a dataset.")
    except Exception as e:
        typer.echo(f"Error listing data sources: {e}", err=True)
        logger.exception("Error listing data sources")
        raise typer.Exit(code=1)


@app.command()
def info(
    name: str = typer.Argument(..., help="Name of the data source"),
) -> None:
    """Show detailed information about a data source.

    Example:
        smep data info mimic3-demo
    """
    try:
        registry = get_registry()

        try:
            source_info = registry.get_fetcher_info(name)
        except KeyError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

        # Display info
        typer.echo(f"\nData Source: {source_info['name']}")
        typer.echo(f"Description: {source_info['description']}")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logger.exception("Error getting data source info")
        raise typer.Exit(code=1)
