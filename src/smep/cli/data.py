"""Data fetching CLI commands."""

from pathlib import Path
from typing import Optional
import typer
import logging

from smep.data import get_registry
from smep.data.kaggle import KaggleDownloadError

logger = logging.getLogger(__name__)

app = typer.Typer(
    no_args_is_help=True, help="Manage data fetching for SMEP datasets."
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

        # Add some helpful information based on the data source
        if name == "mimic3":
            typer.echo(
                "\nNote: MIMIC-III requires Kaggle credentials. "
                "Use KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json"
            )
            typer.echo(
                "See: https://github.com/Kaggle/kaggle-api#api-credentials"
            )

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logger.exception("Error getting data source info")
        raise typer.Exit(code=1)
