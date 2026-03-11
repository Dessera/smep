"""Model training CLI commands."""

from pathlib import Path
from typing import Optional
import typer
import logging

from smep.models import get_registry

logger = logging.getLogger(__name__)

app = typer.Typer(
    no_args_is_help=True,
    help="Manage training and export of SMEP models.",
)


@app.command()
def train(
    model: str = typer.Argument(
        ..., help="Model name to look up in the registry (e.g. 'xgboost')"
    ),
    source: Path = typer.Argument(
        ..., help="Input data path (processed data directory)"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for exported weights, defaults to ./weights/<model>",
    ),
) -> None:
    """Train the specified model and export weights to the given path.

    Example:
        smep model train xgboost .data/mimic3_processed
        smep model train xgboost .data/mimic3_processed --output ./weights/xgboost_v1
    """
    try:
        # Resolve and validate source path
        source = source.resolve()
        if not source.exists():
            typer.echo(f"Error: data path does not exist: {source}", err=True)
            raise typer.Exit(code=1)

        # Determine output path
        if output is None:
            output = Path.cwd() / "weights" / model
        else:
            output = output.resolve()

        # Retrieve model instance from registry
        registry = get_registry()

        try:
            model_instance = registry.get_model(model)
        except KeyError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

        # Train model
        typer.echo(f"Training model '{model}'...")
        typer.echo(f"Source: {source}")
        typer.echo(f"Output: {output}")

        try:
            model_instance.train(source)
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
            typer.echo(f"Unexpected error during training: {e}", err=True)
            logger.exception("Unexpected error during training")
            raise typer.Exit(code=1)

        # Export model
        try:
            model_instance.export(output)
        except RuntimeError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        except Exception as e:
            typer.echo(f"Unexpected error during export: {e}", err=True)
            logger.exception("Unexpected error during export")
            raise typer.Exit(code=1)

        typer.echo(
            f"✓ Model '{model}' trained successfully. Weights exported to {output}"
        )

    except typer.Exit:
        raise


@app.command(name="list")
def list_models() -> None:
    """List all registered available models.

    Example:
        smep model list
    """
    try:
        registry = get_registry()
        models = registry.get_all_model_info()

        if not models:
            typer.echo("No models available.")
            return

        typer.echo("\nAvailable models:")
        for model_info in models:
            typer.echo(
                f"  • {model_info['name']:<20} - {model_info['description']}"
            )

        typer.echo(
            "\nUse 'smep model train <model> <source>' to train a model."
        )
    except Exception as e:
        typer.echo(f"Error listing models: {e}", err=True)
        logger.exception("Error listing models")
        raise typer.Exit(code=1)


@app.command()
def info(
    model: str = typer.Argument(..., help="Model name"),
) -> None:
    """Show detailed information for the specified model.

    Example:
        smep model info xgboost
    """
    try:
        registry = get_registry()

        try:
            model_info = registry.get_model_info(model)
        except KeyError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

        typer.echo(f"\nModel: {model_info['name']}")
        typer.echo(f"Description: {model_info['description']}")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logger.exception("Error retrieving model info")
        raise typer.Exit(code=1)


@app.command()
def infer(
    model: str = typer.Argument(
        ..., help="Model name to look up in the registry (e.g. 'xgboost')"
    ),
    weight_dir: Path = typer.Argument(
        ..., help="Directory containing exported weight files"
    ),
    source: Path = typer.Argument(
        ...,
        help="Processed data directory (X.csv, feature_names.txt, metadata.json)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output CSV path, defaults to ./predictions/<model>_predictions.csv",
    ),
) -> None:
    """Load a trained model and run inference on the given data.

    Example:
        smep model infer xgboost ./weights/xgboost .data/mimic3_processed
        smep model infer xgboost ./weights/xgboost .data/mimic3_processed -o ./results/preds.csv
    """
    try:
        # Resolve and validate weight_dir
        weight_dir = weight_dir.resolve()
        if not weight_dir.exists():
            typer.echo(
                f"Error: weight directory does not exist: {weight_dir}",
                err=True,
            )
            raise typer.Exit(code=1)

        # Resolve and validate source path
        source = source.resolve()
        if not source.exists():
            typer.echo(f"Error: data path does not exist: {source}", err=True)
            raise typer.Exit(code=1)

        # Determine output path
        if output is None:
            output = Path.cwd() / "predictions" / f"{model}_predictions.csv"
        else:
            output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        # Retrieve model instance from registry
        registry = get_registry()
        try:
            model_instance = registry.get_model(model)
        except KeyError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

        typer.echo(f"Loading model '{model}' from {weight_dir}...")

        # Load weights
        try:
            model_instance.load(weight_dir)
        except FileNotFoundError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        except RuntimeError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        except Exception as e:
            typer.echo(f"Unexpected error during model loading: {e}", err=True)
            logger.exception("Unexpected error during model loading")
            raise typer.Exit(code=1)

        typer.echo(f"Running inference on {source}...")

        # Run inference
        try:
            predictions = model_instance.infer(source)
        except FileNotFoundError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        except RuntimeError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        except Exception as e:
            typer.echo(f"Unexpected error during inference: {e}", err=True)
            logger.exception("Unexpected error during inference")
            raise typer.Exit(code=1)

        # Write predictions to CSV
        import pandas as pd

        df = pd.DataFrame(
            {"subject_id": range(len(predictions)), "prediction": predictions}
        )
        df.to_csv(output, index=False)

        typer.echo(
            f"✓ Inference complete. {len(predictions)} predictions written to {output}"
        )

    except typer.Exit:
        raise
