import logging
import sys
import typer

from . import data as data_cli

# Configure logging to output to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

app = typer.Typer(no_args_is_help=True)

# Add data sub-command group
app.add_typer(data_cli.app, name="data")
