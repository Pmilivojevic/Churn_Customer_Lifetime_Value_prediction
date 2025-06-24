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


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_file: Path
    model_1: Path
    model_1_scaler: Path
    model_2: Path
    model_params: dict


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_file: Path
    model_1: Path
    model_1_scaler: Path
    model_2: Path
    model_1_stats: Path
    model_2_stats: Path
