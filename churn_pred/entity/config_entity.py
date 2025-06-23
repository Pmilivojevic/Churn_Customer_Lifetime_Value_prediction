from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    local_data_file: Path
    STATUS_FILE: Path
    all_schema: list


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_file: Path
    train_file: Path
    test_file: Path
    transfrmation_params: dict
    dataset_val_status: bool
