"""Data fetching and processing CLI commands."""

from pathlib import Path
from typing import List, Optional
import typer
import logging

from smep.data import get_registry, get_processor_registry, KaggleDownloadError

logger = logging.getLogger(__name__)

app = typer.Typer(
    no_args_is_help=True,
    help="Manage data fetching and processing for SMEP datasets.",
)


@app.command()
def fetch(
    name: str = typer.Argument(
        ..., help="Name of the data source to fetch (e.g., 'mimic3')"
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
        smep data fetch mimic3
        smep data fetch mimic3 --output /path/to/data
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
        smep data info mimic3
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


@app.command()
def process(
    name: str = typer.Argument(
        ..., help="Name of the processor to use (e.g., 'mimic3')"
    ),
    source: Path = typer.Argument(
        ..., help="Path to the source data directory"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for processed data. Defaults to source directory + '_processed'.",
    ),
    agg_stats: Optional[List[str]] = typer.Option(
        None,
        "--agg-stats",
        help=(
            "Aggregation statistics for temporal features. "
            "Valid values: mean, max, min, std. "
            "Can be specified multiple times (e.g. --agg-stats mean --agg-stats std). "
            "Defaults to 'mean'."
        ),
    ),
) -> None:
    """Process data using a specified processor.

    Example:
        smep data process mimic3 .data/mimic-iii-clinical-database-demo-1.4
        smep data process mimic3 .data/mimic-iii --output .data/processed
        smep data process mimic3 .data/mimic-iii --agg-stats mean --agg-stats std
    """
    try:
        # Resolve source path
        source = source.resolve()

        # Determine output directory
        if output is None:
            output = source.parent / f"{source.name}_processed"
        else:
            output = output.resolve()

        # Resolve agg_stats: default to ["mean"] if not specified
        resolved_agg_stats = agg_stats if agg_stats else ["mean"]

        # Get the processor from registry
        registry = get_processor_registry()

        try:
            processor = registry.get_processor(
                name, agg_stats=resolved_agg_stats
            )
        except (KeyError, ValueError) as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

        # Process the data
        typer.echo(f"Processing data with {name} processor...")
        typer.echo(f"Source directory: {source}")
        typer.echo(f"Output directory: {output}")
        typer.echo(f"Aggregation statistics: {', '.join(resolved_agg_stats)}")

        try:
            processor.process(source, output)
            typer.echo(f"✓ Successfully processed data to {output}")
        except FileNotFoundError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        except RuntimeError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        except Exception as e:
            typer.echo(f"Unexpected error: {e}", err=True)
            logger.exception("Unexpected error during processing")
            raise typer.Exit(code=1)

    except typer.Exit:
        raise


@app.command()
def list_processors() -> None:
    """List all available data processors.

    Example:
        smep data list-processors
    """
    try:
        registry = get_processor_registry()
        processors = registry.get_all_processor_info()

        if not processors:
            typer.echo("No data processors available.")
            return

        # Display processors
        typer.echo("\nAvailable Data Processors:")
        for proc_info in processors:
            typer.echo(
                f"  • {proc_info['name']:<20} - {proc_info['description']}"
            )

        typer.echo(
            "\nUse 'smep data process <name> <source>' to process a dataset."
        )
    except Exception as e:
        typer.echo(f"Error listing processors: {e}", err=True)
        logger.exception("Error listing processors")
        raise typer.Exit(code=1)


@app.command()
def processor_info(
    name: str = typer.Argument(..., help="Name of the processor"),
) -> None:
    """Show detailed information about a data processor.

    Example:
        smep data processor-info mimic3
    """
    try:
        registry = get_processor_registry()

        try:
            proc_info = registry.get_processor_info(name)
        except KeyError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

        # Display info
        typer.echo(f"\nData Processor: {proc_info['name']}")
        typer.echo(f"Description: {proc_info['description']}")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logger.exception("Error getting processor info")
        raise typer.Exit(code=1)
