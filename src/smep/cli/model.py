"""Model training CLI commands."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional
import typer
import logging

from smep.models import get_registry
from smep.models.data_loader import load_training_data
from smep.models.evaluator import (
    evaluate,
    compute_curve_points,
    find_optimal_threshold,
    write_evaluation_outputs,
    _to_json_compatible,
)
from smep.models.explainer import write_explain_outputs, to_json_compatible
from smep.models.feature_selector import (
    evaluate_feature_importance,
    write_feature_importance_outputs,
)

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
        ..., help="Build artifact directory (containing X_train.csv, etc.)"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for exported weights, defaults to ./weights/<model>",
    ),
    tuning_strategy: str = typer.Option(
        "none",
        "--tuning-strategy",
        help="Hyperparameter tuning strategy: none, grid, or random.",
    ),
    cv: int = typer.Option(
        5,
        "--cv",
        help="Cross-validation folds for tuning.",
    ),
    scoring: str = typer.Option(
        "roc_auc",
        "--scoring",
        help="Scoring metric used by tuning search.",
    ),
    n_iter: int = typer.Option(
        100,
        "--n-iter",
        help="Number of sampled candidates when strategy is random.",
    ),
    param_space: Optional[Path] = typer.Option(
        None,
        "--param-space",
        help="JSON file containing param_grid or param_distributions.",
    ),
    threshold_strategy: str = typer.Option(
        "none",
        "--threshold-strategy",
        help="Threshold optimisation strategy on validation set: none (use 0.5), youden, or f1.",
    ),
) -> None:
    """Train the specified model on build artifacts and write outputs.

    Example:
        smep model train xgboost .data/build_output
        smep model train xgboost .data/build_output --output ./weights/xgboost_v1
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

        strategy = tuning_strategy.strip().lower()
        valid_strategies = {"none", "grid", "random"}
        if strategy not in valid_strategies:
            typer.echo(
                "Error: --tuning-strategy must be one of: none, grid, random",
                err=True,
            )
            raise typer.Exit(code=1)

        if cv < 2:
            typer.echo("Error: --cv must be >= 2", err=True)
            raise typer.Exit(code=1)

        if n_iter < 1:
            typer.echo("Error: --n-iter must be >= 1", err=True)
            raise typer.Exit(code=1)

        tuning: dict[str, Any] = {
            "strategy": strategy,
            "cv": cv,
            "scoring": scoring,
            "n_iter": n_iter,
            "n_jobs": -1,
            "random_state": 42,
        }

        if param_space is not None:
            param_space = param_space.resolve()
            if not param_space.exists():
                typer.echo(
                    f"Error: param-space file does not exist: {param_space}",
                    err=True,
                )
                raise typer.Exit(code=1)

            try:
                parsed_space = json.loads(
                    param_space.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError as e:
                typer.echo(
                    f"Error: invalid JSON in param-space file: {e}", err=True
                )
                raise typer.Exit(code=1)

            if not isinstance(parsed_space, dict):
                typer.echo(
                    "Error: param-space JSON must be an object/dict",
                    err=True,
                )
                raise typer.Exit(code=1)

            if strategy == "grid":
                tuning["param_grid"] = parsed_space
            elif strategy == "random":
                tuning["param_distributions"] = parsed_space

        typer.echo(f"Training model '{model}'...")
        typer.echo(f"Source: {source}")
        typer.echo(f"Output: {output}")
        typer.echo(f"Tuning strategy: {strategy}")

        # 1. Load data from build artifacts
        try:
            data = load_training_data(source)
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

        # 2. Train model
        try:
            model_instance.fit(
                data.X_train,
                data.y_train,
                data.X_val,
                data.y_val,
                tuning=tuning,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        except Exception as e:
            typer.echo(f"Unexpected error during training: {e}", err=True)
            logger.exception("Unexpected error during training")
            raise typer.Exit(code=1)

        # 3. Threshold tuning on validation set
        threshold_strategy_clean = threshold_strategy.strip().lower()
        valid_threshold_strategies = {"none", "youden", "f1"}
        if threshold_strategy_clean not in valid_threshold_strategies:
            typer.echo(
                "Error: --threshold-strategy must be one of: none, youden, f1",
                err=True,
            )
            raise typer.Exit(code=1)

        threshold = 0.5
        threshold_info: dict[str, Any] = {
            "strategy": threshold_strategy_clean,
            "threshold": 0.5,
            "metric_value": None,
        }
        if threshold_strategy_clean != "none":
            val_proba = model_instance.predict_proba(data.X_val)
            thr_result = find_optimal_threshold(
                data.y_val, val_proba, strategy=threshold_strategy_clean
            )
            threshold = thr_result.threshold
            threshold_info = {
                "strategy": thr_result.strategy,
                "threshold": thr_result.threshold,
                "metric_value": thr_result.metric_value,
            }
            typer.echo(
                f"Optimal threshold ({thr_result.strategy}): "
                f"{thr_result.threshold:.4f} "
                f"(metric={thr_result.metric_value:.4f})"
            )

        # 4. Evaluate on all three splits
        train_result = evaluate(
            data.y_train,
            model_instance.predict_proba(data.X_train),
            "train",
            threshold=threshold,
        )
        val_result = evaluate(
            data.y_val,
            model_instance.predict_proba(data.X_val),
            "val",
            threshold=threshold,
        )
        test_result = evaluate(
            data.y_test,
            model_instance.predict_proba(data.X_test),
            "test",
            threshold=threshold,
        )

        # 5. Save model weights
        try:
            model_meta = model_instance.save(output)
        except RuntimeError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        except Exception as e:
            typer.echo(f"Unexpected error during save: {e}", err=True)
            logger.exception("Unexpected error during save")
            raise typer.Exit(code=1)

        # 6. Write feature names
        feature_file = output / "feature_names.txt"
        feature_file.write_text("\n".join(data.feature_names), encoding="utf-8")

        # 7. Write evaluation outputs
        curve_points = compute_curve_points(
            data.y_test, model_instance.predict_proba(data.X_test)
        )
        write_evaluation_outputs(
            output, train_result, val_result, test_result, curve_points
        )

        # 8. Write metadata
        metadata: dict[str, Any] = {
            "model": model,
            "model_file": model_meta.get("model_file"),
            "n_features": len(data.feature_names),
            "feature_names_file": "feature_names.txt",
            "training_config": {
                "tuning_strategy": strategy,
                "cv": cv,
                "scoring": scoring,
            },
            "threshold": threshold_info,
            "tuning_summary": model_meta.get(
                "tuning_summary",
                {
                    "enabled": False,
                    "best_params": None,
                    "best_cv_score": None,
                },
            ),
            "dataset_manifest": data.manifest,
            "classifier_params": model_meta.get(
                "classifier_params", model_meta.get("training_config")
            ),
        }
        metadata = _to_json_compatible(metadata)
        meta_file = output / "metadata.json"
        meta_file.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

        typer.echo(
            f"✓ Model '{model}' trained successfully. Weights exported to {output}"
        )

        # Print summary
        typer.echo(
            f"Evaluation (test): "
            f"Accuracy={test_result.accuracy:.4f}, "
            f"Precision={test_result.precision:.4f}, "
            f"Recall={test_result.recall:.4f}, "
            f"F1={test_result.f1:.4f}, "
            f"ROC-AUC={test_result.roc_auc}, "
            f"PR-AUC={test_result.pr_auc}"
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
        help="CSV file or directory containing X_test.csv for inference",
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
        smep model infer xgboost ./weights/xgboost ./build_output/X_test.csv
        smep model infer xgboost ./weights/xgboost ./build_output -o ./results/preds.csv
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

        # Determine feature file
        if source.is_file():
            x_file = source
        else:
            # Prefer X_test.csv in directory
            x_file = source / "X_test.csv"
            if not x_file.exists():
                x_file = source / "X.csv"
            if not x_file.exists():
                typer.echo(
                    f"Error: no X_test.csv or X.csv found in {source}",
                    err=True,
                )
                raise typer.Exit(code=1)

        typer.echo(f"Running inference on {x_file}...")

        import pandas as pd
        import numpy as np

        try:
            X_df = pd.read_csv(x_file)
            if X_df.empty:
                typer.echo("Error: feature file is empty.", err=True)
                raise typer.Exit(code=1)
            X = X_df.values.astype(np.float32)
            predictions = model_instance.predict_proba(X)
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
        df = pd.DataFrame(
            {"subject_id": range(len(predictions)), "prediction": predictions}
        )
        df.to_csv(output, index=False)

        typer.echo(
            f"✓ Inference complete. {len(predictions)} predictions written to {output}"
        )

    except typer.Exit:
        raise


@app.command()
def explain(
    model: str = typer.Argument(
        ..., help="Model name to look up in the registry (e.g. 'xgboost')"
    ),
    weight_dir: Path = typer.Argument(
        ...,
        help="Directory containing exported model weights",
    ),
    data_source: Path = typer.Argument(
        ...,
        help="Build artifact directory or CSV file for explain data",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Output directory for explainability artifacts, "
            "defaults to ./explanations/<model>_<weight_dir_name>"
        ),
    ),
    max_samples: int = typer.Option(
        500,
        "--max-samples",
        help="Maximum number of samples used to compute SHAP explanations.",
    ),
) -> None:
    """Generate explainability artifacts for a trained model.

    Example:
        smep model explain xgboost ./weights/xgboost ./build_output
        smep model explain dnn ./weights/dnn ./build_output/X_test.csv -o ./explanations/run1
    """
    try:
        import pandas as pd

        # Resolve and validate weight_dir
        weight_dir = weight_dir.resolve()
        if not weight_dir.exists() or not weight_dir.is_dir():
            typer.echo(
                f"Error: weight directory does not exist: {weight_dir}",
                err=True,
            )
            raise typer.Exit(code=1)

        # Resolve and validate data_source
        data_source = data_source.resolve()
        if not data_source.exists():
            typer.echo(
                f"Error: data source does not exist: {data_source}",
                err=True,
            )
            raise typer.Exit(code=1)

        if max_samples < 1:
            typer.echo("Error: --max-samples must be >= 1", err=True)
            raise typer.Exit(code=1)

        # Determine output path
        if output is None:
            output = Path.cwd() / "explanations" / f"{model}_{weight_dir.name}"
        else:
            output = output.resolve()
        output.mkdir(parents=True, exist_ok=True)

        # Retrieve model instance from registry
        registry = get_registry()
        try:
            model_instance = registry.get_model(model)
        except KeyError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

        # Check compute_shap support
        compute_shap_method = getattr(model_instance, "compute_shap", None)
        if not callable(compute_shap_method):
            typer.echo(
                f"Error: model '{model}' does not support compute_shap.",
                err=True,
            )
            raise typer.Exit(code=1)

        # 1. Load model weights
        typer.echo(f"Loading model '{model}' from {weight_dir}...")
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

        # 2. Load feature data
        if data_source.is_file():
            x_file = data_source
        else:
            x_file = data_source / "X_test.csv"
            if not x_file.exists():
                x_file = data_source / "X.csv"
            if not x_file.exists():
                typer.echo(
                    f"Error: no X_test.csv or X.csv found in {data_source}",
                    err=True,
                )
                raise typer.Exit(code=1)

        try:
            X_df = pd.read_csv(x_file)
            if X_df.empty:
                typer.echo(f"Error: feature file is empty: {x_file}", err=True)
                raise typer.Exit(code=1)
        except Exception as e:
            typer.echo(f"Error reading feature data: {e}", err=True)
            raise typer.Exit(code=1)

        # 3. Load feature names (prefer weight_dir, fallback to data_source)
        feature_names: list[str] = []
        for candidate_dir in [weight_dir, data_source]:
            fn_file = candidate_dir / "feature_names.txt"
            if fn_file.is_file():
                feature_names = [
                    line.strip()
                    for line in fn_file.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                break

        if feature_names and len(feature_names) == X_df.shape[1]:
            X_df.columns = feature_names

        # 4. Sampling
        if len(X_df) > max_samples:
            X_sample = X_df.sample(n=max_samples, random_state=42).reset_index(
                drop=True
            )
        else:
            X_sample = X_df.reset_index(drop=True)

        typer.echo(f"Generating explainability artifacts for '{model}'...")
        typer.echo(f"Weight dir: {weight_dir}")
        typer.echo(f"Data source: {x_file}")
        typer.echo(f"Output: {output}")
        typer.echo(f"Samples: {len(X_sample)} (max {max_samples})")

        # 5. Compute SHAP values
        try:
            shap_result = model_instance.compute_shap(X_sample, max_samples)
        except RuntimeError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        except Exception as e:
            typer.echo(
                f"Unexpected error during SHAP computation: {e}", err=True
            )
            logger.exception("Unexpected error during SHAP computation")
            raise typer.Exit(code=1)

        # 6. Write SHAP output artifacts
        try:
            outputs = write_explain_outputs(output, shap_result, X_sample)
        except RuntimeError as e:
            typer.echo(f"Error writing explain outputs: {e}", err=True)
            raise typer.Exit(code=1)

        # 7. Write explain metadata
        metadata: dict[str, Any] = {
            "model": model,
            "method": "shap",
            "explainer": shap_result.explainer_type,
            "shap_version": shap_result.shap_version,
            "sample_size": int(len(X_sample)),
            "n_features": int(X_sample.shape[1]),
            "weight_dir": str(weight_dir),
            "data_source": str(x_file),
            "outputs": outputs,
            "status": "success",
        }
        metadata = to_json_compatible(metadata)
        meta_file = output / "explain_metadata.json"
        meta_file.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

        typer.echo(f"✓ Explainability artifacts generated at {output}")
        typer.echo(f"Metadata: {meta_file}")

        if isinstance(outputs.get("summary_bar"), str):
            typer.echo(f"SHAP bar plot: {output / outputs['summary_bar']}")
        if isinstance(outputs.get("summary_beeswarm"), str):
            typer.echo(
                f"SHAP beeswarm plot: {output / outputs['summary_beeswarm']}"
            )

    except typer.Exit:
        raise

    except typer.Exit:
        raise


@app.command(name="feature-importance")
def feature_importance(
    model: str = typer.Argument(
        ..., help="Model name to look up in the registry (e.g. 'xgboost')"
    ),
    weight_dir: Path = typer.Argument(
        ..., help="Directory containing exported model weights"
    ),
    data_source: Path = typer.Argument(
        ...,
        help="Build artifact directory containing X_test.csv / y_test.csv",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Output directory for importance artifacts, "
            "defaults to ./feature_importance/<model>"
        ),
    ),
    scoring: str = typer.Option(
        "roc_auc",
        "--scoring",
        help="Scoring metric for permutation importance.",
    ),
    n_repeats: int = typer.Option(
        10,
        "--n-repeats",
        help="Number of permutation repeats per feature.",
    ),
) -> None:
    """Evaluate feature importance using permutation importance.

    Identifies noise features (importance <= 0) and signal features,
    writes a JSON report and visualization.

    Example:
        smep model feature-importance xgboost ./weights/xgboost ./build_output
        smep model feature-importance dnn ./weights/dnn ./build_output --scoring f1
    """
    try:
        import pandas as pd
        import numpy as np

        # Resolve and validate weight_dir
        weight_dir = weight_dir.resolve()
        if not weight_dir.exists() or not weight_dir.is_dir():
            typer.echo(
                f"Error: weight directory does not exist: {weight_dir}",
                err=True,
            )
            raise typer.Exit(code=1)

        # Resolve and validate data_source
        data_source = data_source.resolve()
        if not data_source.exists():
            typer.echo(
                f"Error: data source does not exist: {data_source}",
                err=True,
            )
            raise typer.Exit(code=1)

        if n_repeats < 1:
            typer.echo("Error: --n-repeats must be >= 1", err=True)
            raise typer.Exit(code=1)

        # Determine output path
        if output is None:
            output = Path.cwd() / "feature_importance" / model
        else:
            output = output.resolve()
        output.mkdir(parents=True, exist_ok=True)

        # Retrieve model instance from registry
        registry = get_registry()
        try:
            model_instance = registry.get_model(model)
        except KeyError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

        # 1. Load model weights
        typer.echo(f"Loading model '{model}' from {weight_dir}...")
        try:
            model_instance.load(weight_dir)
        except (FileNotFoundError, RuntimeError) as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        except Exception as e:
            typer.echo(f"Unexpected error during model loading: {e}", err=True)
            logger.exception("Unexpected error during model loading")
            raise typer.Exit(code=1)

        # 2. Load test data
        x_file = data_source / "X_test.csv"
        y_file = data_source / "y_test.csv"
        if not x_file.exists() or not y_file.exists():
            typer.echo(
                f"Error: X_test.csv and y_test.csv required in {data_source}",
                err=True,
            )
            raise typer.Exit(code=1)

        try:
            X_df = pd.read_csv(x_file)
            y_df = pd.read_csv(y_file)
            if X_df.empty or y_df.empty:
                typer.echo("Error: test data files are empty.", err=True)
                raise typer.Exit(code=1)
            X = X_df.values.astype(np.float32)
            y = y_df.iloc[:, 0].values.astype(np.int32)
        except Exception as e:
            typer.echo(f"Error reading test data: {e}", err=True)
            raise typer.Exit(code=1)

        # 3. Load feature names
        feature_names: list[str] = []
        for candidate_dir in [weight_dir, data_source]:
            fn_file = candidate_dir / "feature_names.txt"
            if fn_file.is_file():
                feature_names = [
                    line.strip()
                    for line in fn_file.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                break

        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        typer.echo(f"Evaluating feature importance for '{model}'...")
        typer.echo(
            f"Data: {x_file} ({X.shape[0]} samples, {X.shape[1]} features)"
        )
        typer.echo(f"Scoring: {scoring}, Repeats: {n_repeats}")

        # 4. Compute permutation importance
        try:
            report = evaluate_feature_importance(
                model=model_instance,
                X=X,
                y=y,
                feature_names=feature_names,
                scoring=scoring,
                n_repeats=n_repeats,
            )
        except Exception as e:
            typer.echo(
                f"Error during feature importance computation: {e}", err=True
            )
            logger.exception("Error during feature importance computation")
            raise typer.Exit(code=1)

        # 5. Write outputs
        try:
            artifacts = write_feature_importance_outputs(output, report)
        except RuntimeError as e:
            typer.echo(f"Error writing outputs: {e}", err=True)
            raise typer.Exit(code=1)

        typer.echo(f"\n✓ Feature importance analysis complete.")
        typer.echo(f"Output: {output}")
        typer.echo(f"Report: {output / artifacts['report']}")
        typer.echo(f"Plot: {output / artifacts['plot']}")
        typer.echo(
            f"\nSignal features: {len(report.signal_features)} | "
            f"Noise features: {len(report.noise_features)}"
        )

        if report.noise_features:
            typer.echo(f"\nNoise features (importance ≤ 0):")
            for name in report.noise_features:
                typer.echo(f"  ✗ {name}")

        # Show top 10 signal features
        top_signal = [r for r in report.results if r.importance_mean > 0][:10]
        if top_signal:
            typer.echo(f"\nTop signal features:")
            for r in top_signal:
                typer.echo(
                    f"  ✓ {r.feature:<30} importance={r.importance_mean:.6f} "
                    f"(±{r.importance_std:.6f})"
                )

    except typer.Exit:
        raise
