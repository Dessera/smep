"""MIMIC-III Database fetcher."""

from pathlib import Path
import logging

from .kaggle import KaggleDownloadError, KaggleFetcher

logger = logging.getLogger(__name__)


class MIMIC3Fetcher(KaggleFetcher):
    """Fetcher for MIMIC-III Clinical Database from Kaggle.

    This fetcher uses the Kaggle API to download the MIMIC-III dataset.
    Requires Kaggle credentials to be configured in ~/.kaggle/kaggle.json
    """

    def fetch(self, output_path: Path) -> None:
        """Fetch MIMIC-III dataset from Kaggle.

        Args:
            output_path: Directory where the dataset will be saved.

        Raises:
            RuntimeError: If credentials are invalid or download fails.
            ValueError: If output path is not writable.
        """

        try:
            logger.info(f"Starting MIMIC-III dataset download to {output_path}")
            self._download_kaggle_dataset(
                "asjad99/mimiciii",
                output_path=output_path,
                unzip=True,
            )

            logger.info(
                f"MIMIC-III dataset successfully downloaded to {output_path}"
            )
        except KaggleDownloadError as e:
            logger.error(f"Failed to download MIMIC-III dataset: {e}")
            raise KaggleDownloadError(
                "Failed to download MIMIC-III dataset from Kaggle."
            ) from e
