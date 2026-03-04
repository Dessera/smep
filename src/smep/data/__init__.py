"""Data fetching and processing module."""

# Re-export fetcher components
from .fetchers import (
    DataFetcher,
    DataFetcherRegistry,
    KaggleDownloadError,
    KaggleFetcher,
    MIMIC3Fetcher,
    get_registry,
)

# Re-export processor components
from .processors import (
    DataProcessor,
    MIMIC3Processor,
    ProcessorRegistry,
    get_processor_registry,
)

__all__ = [
    # Fetchers
    "DataFetcher",
    "KaggleFetcher",
    "KaggleDownloadError",
    "MIMIC3Fetcher",
    "DataFetcherRegistry",
    "get_registry",
    # Processors
    "DataProcessor",
    "MIMIC3Processor",
    "ProcessorRegistry",
    "get_processor_registry",
]
