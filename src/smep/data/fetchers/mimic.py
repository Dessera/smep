"""MIMIC-III Database fetcher."""

from pathlib import Path
import shutil
import logging

from .kaggle import KaggleDownloadError, KaggleFetcher

logger = logging.getLogger(__name__)


class MIMIC3DemoFetcher(KaggleFetcher):
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


class MIMIC310KFetcher(KaggleFetcher):
    """Fetcher for MIMIC-III 10K subset from Kaggle."""

    # Preferred source files from the 10k dataset package.
    _PREFERRED_SOURCE_FILES: dict[str, str] = {
        "D_ITEMS.csv": "D_ITEMS/D_ITEMS.csv",
        "D_LABITEMS.csv": "D_LABITEMS/D_LABITEMS.csv",
        "LABEVENTS.csv": "LABEVENTS/LABEVENTS_sorted.csv",
        "OUTPUTEVENTS.csv": "OUTPUTEVENTS/OUTPUTEVENTS_sorted.csv",
        "INPUTEVENTS_CV.csv": "INPUTEVENTS_CV/INPUTEVENTS_CV_sorted.csv",
        "PATIENTS.csv": "PATIENTS/PATIENTS_sorted.csv",
        "ADMISSIONS.csv": "ADMISSIONS/ADMISSIONS_sorted.csv",
        "ICUSTAYS.csv": "ICUSTAYS/ICUSTAYS_sorted.csv",
        "D_ICD_DIAGNOSES.csv": "D_ICD_DIAGNOSES/D_ICD_DIAGNOSES.csv",
        "DIAGNOSES_ICD.csv": "DIAGNOSES_ICD/DIAGNOSES_ICD_sorted.csv",
    }

    def fetch(self, output_path: Path) -> None:
        """Fetch MIMIC-III 10K dataset from Kaggle.

        Args:
            output_path: Directory where the dataset will be saved.

        Raises:
            RuntimeError: If credentials are invalid or download fails.
            ValueError: If output path is not writable.
        """

        try:
            logger.info(
                "Starting MIMIC-III 10K dataset download to %s",
                output_path,
            )
            self._download_kaggle_dataset(
                "bilal1907/mimic-iii-10k",
                output_path=output_path,
                unzip=True,
            )

            normalized_dir = self._normalize_to_mimic3_layout(output_path)

            logger.info(
                "MIMIC-III 10K dataset successfully downloaded to %s "
                "and normalized to %s",
                output_path,
                normalized_dir,
            )
        except KaggleDownloadError as e:
            logger.error("Failed to download MIMIC-III 10K dataset: %s", e)
            raise KaggleDownloadError(
                "Failed to download MIMIC-III 10K dataset from Kaggle."
            ) from e

    def _normalize_to_mimic3_layout(self, output_path: Path) -> Path:
        """Normalize nested 10K tables to flat MIMIC-III-style CSV layout."""
        source_root = self._find_10k_source_root(output_path)
        normalized_root = output_path / "mimic-iii-10k"
        normalized_root.mkdir(parents=True, exist_ok=True)

        copied_tables = 0
        copied_names: set[str] = set()

        for (
            target_name,
            source_rel_path,
        ) in self._PREFERRED_SOURCE_FILES.items():
            source_file = source_root / source_rel_path
            if not source_file.exists() or not source_file.is_file():
                logger.warning(
                    "Preferred source file not found for %s: %s",
                    target_name,
                    source_rel_path,
                )
                continue

            target_file = normalized_root / target_name
            shutil.copy2(source_file, target_file)
            copied_names.add(target_name)
            copied_tables += 1

            logger.debug(
                "Normalized table %s using mapped source %s",
                target_name,
                source_rel_path,
            )

        for table_dir in sorted(source_root.iterdir()):
            if not table_dir.is_dir():
                continue

            target_name = f"{table_dir.name}.csv"
            if target_name in copied_names:
                continue

            csv_files = sorted(
                file_path
                for file_path in table_dir.iterdir()
                if file_path.is_file() and file_path.suffix.lower() == ".csv"
            )
            if not csv_files:
                continue

            selected_file = self._select_preferred_csv(csv_files)
            target_file = normalized_root / target_name
            shutil.copy2(selected_file, target_file)
            copied_names.add(target_name)
            copied_tables += 1

            logger.debug(
                "Normalized table %s using %s",
                table_dir.name,
                selected_file.name,
            )

        if copied_tables == 0:
            raise RuntimeError(
                "No CSV tables were found to normalize in the downloaded "
                "MIMIC-III 10K dataset."
            )

        required_tables = [
            "ADMISSIONS.csv",
            "DIAGNOSES_ICD.csv",
            "PATIENTS.csv",
            "ICUSTAYS.csv",
            "LABEVENTS.csv",
            "CHARTEVENTS.csv",
        ]
        missing_required = [
            table
            for table in required_tables
            if not (normalized_root / table).exists()
        ]
        if missing_required:
            logger.warning(
                "Normalized dataset is missing processor-required tables: %s",
                ", ".join(missing_required),
            )

        return normalized_root

    def _find_10k_source_root(self, output_path: Path) -> Path:
        """Locate extracted root directory for MIMIC-III 10K dataset."""
        expected_root = output_path / "MIMIC -III (10000 patients)"
        if expected_root.exists() and expected_root.is_dir():
            return expected_root

        candidates: list[tuple[int, Path]] = []
        for child in output_path.iterdir():
            if not child.is_dir():
                continue

            subdirs = [sub for sub in child.iterdir() if sub.is_dir()]
            if not subdirs:
                continue

            has_csv_subdirs = any(
                any(
                    item.is_file() and item.suffix.lower() == ".csv"
                    for item in subdir.iterdir()
                )
                for subdir in subdirs
            )
            if not has_csv_subdirs:
                continue

            score = len(subdirs)
            if (child / "ADMISSIONS").is_dir():
                score += 10
            if (child / "DIAGNOSES_ICD").is_dir():
                score += 10
            candidates.append((score, child))

        if not candidates:
            raise RuntimeError(
                "Unable to locate extracted MIMIC-III 10K source directory "
                f"under {output_path}."
            )

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _select_preferred_csv(self, csv_files: list[Path]) -> Path:
        """Select sorted CSV when present; otherwise use the only/first CSV."""
        if len(csv_files) == 1:
            return csv_files[0]

        sorted_candidates = [
            file_path
            for file_path in csv_files
            if "sorted" in file_path.name.lower()
        ]
        if sorted_candidates:
            return sorted_candidates[0]

        return csv_files[0]
