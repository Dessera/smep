"""Kaggle-based data fetcher abstractions."""

from pathlib import Path
from typing import Any
import logging

from .fetcher import DataFetcher

logger = logging.getLogger(__name__)


class KaggleFetcherError(RuntimeError):
    """Base exception for Kaggle fetcher errors."""


class KaggleDownloadError(KaggleFetcherError):
    """Raised when Kaggle dataset download fails."""


class KaggleFetcher(DataFetcher):
    """Intermediate fetcher class for Kaggle-backed datasets."""

    def _get_kaggle_api(self) -> Any:
        """Create and authenticate a Kaggle API client."""
        from kaggle.api.kaggle_api_extended import (  # type: ignore
            KaggleApi,
        )

        api = KaggleApi()
        api.authenticate()
        return api

    def _download_kaggle_dataset(
        self, dataset: str, output_path: Path, unzip: bool = True
    ) -> None:
        """Download a dataset from Kaggle to the given directory."""
        try:
            api = self._get_kaggle_api()
            api.dataset_download_files(  # type: ignore
                dataset,
                path=str(output_path),
                unzip=unzip,
            )
        except Exception as error:
            logger.error(
                f"Failed to download Kaggle dataset '{dataset}': {error}"
            )
            raise KaggleDownloadError(
                f"Failed to download dataset '{dataset}' from Kaggle."
            ) from error
