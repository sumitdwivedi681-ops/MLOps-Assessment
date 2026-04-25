import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from typing import Any, Dict, List

import yaml


REQUIRED_CONFIG_FIELDS = ["seed", "window", "version"]
REQUIRED_DATA_COLUMN = "close"


class ValidationError(Exception):
    """Custom validation error for clean failure messages."""


def setup_logging(log_file: str) -> None:
    """Configure logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate YAML configuration."""
    if not os.path.exists(config_path):
        raise ValidationError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValidationError(f"Invalid YAML config format: {exc}") from exc
    except OSError as exc:
        raise ValidationError(f"Unable to read config file: {exc}") from exc

    if not isinstance(config, dict):
        raise ValidationError("Invalid config structure: config must be a YAML object/dictionary")

    missing_fields = [field for field in REQUIRED_CONFIG_FIELDS if field not in config]
    if missing_fields:
        raise ValidationError(f"Missing required config fields: {missing_fields}")

    if not isinstance(config["seed"], int):
        raise ValidationError("Invalid config: seed must be an integer")

    if not isinstance(config["window"], int) or config["window"] <= 0:
        raise ValidationError("Invalid config: window must be a positive integer")

    if not isinstance(config["version"], str) or not config["version"].strip():
        raise ValidationError("Invalid config: version must be a non-empty string")

    return config


def load_close_values(input_path: str) -> List[float]:
    """Load and validate CSV dataset, returning numeric close values."""
    if not os.path.exists(input_path):
        raise ValidationError(f"Input file not found: {input_path}")

    try:
        with open(input_path, "r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                raise ValidationError("Empty file: input CSV has no header or data")

            if REQUIRED_DATA_COLUMN not in reader.fieldnames:
                raise ValidationError(f"Missing required column: {REQUIRED_DATA_COLUMN}")

            close_values: List[float] = []
            for row_number, row in enumerate(reader, start=2):
                raw_value = row.get(REQUIRED_DATA_COLUMN)
                if raw_value is None or raw_value == "":
                    raise ValidationError(f"Missing close value at CSV row {row_number}")
                try:
                    close_values.append(float(raw_value))
                except ValueError as exc:
                    raise ValidationError(f"Invalid numeric close value at CSV row {row_number}: {raw_value}") from exc

    except csv.Error as exc:
        raise ValidationError(f"Invalid CSV format: {exc}") from exc
    except OSError as exc:
        raise ValidationError(f"Unable to read input file: {exc}") from exc

    if not close_values:
        raise ValidationError("Input dataset is empty")

    return close_values


def compute_signals(close_values: List[float], window: int) -> List[int]:
    """Compute rolling mean and binary signals.

    Handling of first window-1 rows:
    They do not have a full rolling window, so they are excluded from signal computation.
    """
    if len(close_values) < window:
        return []

    signals: List[int] = []

    for index in range(window - 1, len(close_values)):
        window_values = close_values[index - window + 1 : index + 1]
        rolling_mean = sum(window_values) / window
        close = close_values[index]
        signal = 1 if close > rolling_mean else 0
        signals.append(signal)

    return signals


def write_metrics(metrics: Dict[str, Any], output_path: str) -> None:
    """Write metrics as machine-readable JSON."""
    try:
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
    except OSError as exc:
        raise ValidationError(f"Unable to write metrics file: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal MLOps batch signal pipeline")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output", required=True, help="Path to output metrics JSON file")
    parser.add_argument("--log-file", required=True, help="Path to log file")
    return parser.parse_args()


def main() -> int:
    start_time = time.perf_counter()
    args = parse_args()
    setup_logging(args.log_file)

    try:
        logging.info("Batch job started")
        logging.info("Input file: %s", args.input)
        logging.info("Config file: %s", args.config)
        logging.info("Output metrics file: %s", args.output)

        config = load_config(args.config)
        random.seed(config["seed"])
        logging.info(
            "Config loaded successfully: seed=%s, window=%s, version=%s",
            config["seed"],
            config["window"],
            config["version"],
        )

        close_values = load_close_values(args.input)
        logging.info("Dataset loaded successfully with %d rows", len(close_values))

        signals = compute_signals(close_values, config["window"])
        logging.info("Signals computed successfully")
        logging.info("Rows excluded due to rolling window: %d", len(close_values) - len(signals))

        rows_processed = len(signals)
        signal_rate = sum(signals) / rows_processed if rows_processed > 0 else 0.0
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

        metrics = {
            "rows_processed": rows_processed,
            "signal_rate": round(signal_rate, 6),
            "latency_ms": latency_ms,
            "version": config["version"],
        }

        write_metrics(metrics, args.output)
        logging.info("Metrics written successfully: %s", metrics)
        logging.info("Batch job completed successfully")
        return 0

    except ValidationError as exc:
        logging.error("Validation error: %s", exc)
        return 1
    except Exception as exc:
        logging.exception("Unexpected error occurred: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
