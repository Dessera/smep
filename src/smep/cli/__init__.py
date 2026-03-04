import typer

from . import data as data_cli

app = typer.Typer(no_args_is_help=True)

# Add data sub-command group
app.add_typer(data_cli.app, name="data")
