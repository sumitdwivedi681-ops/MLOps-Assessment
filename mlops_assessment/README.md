# T0 - ML/MLOps Engineering Internship Assessment

This project implements a minimal MLOps-style batch job in Python.

## Features

- Loads configuration from YAML
- Uses deterministic seed for reproducibility
- Reads OHLCV CSV data
- Validates input dataset and config
- Computes rolling mean on `close`
- Generates binary signal
- Writes machine-readable metrics JSON
- Writes detailed logs
- Runs locally and inside Docker

## Project Structure

```text
.
├── run.py
├── config.yaml
├── data.csv
├── requirements.txt
├── Dockerfile
└── README.md
```

## Config

`config.yaml`

```yaml
seed: 42
window: 5
version: "v1"
```

## Local Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the batch job:

```bash
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

## Docker Run

Build image:

```bash
docker build -t mlops-assessment .
```

Run container:

### Windows PowerShell

```bash
docker run --rm -v ${PWD}:/app mlops-assessment
```

### Linux / macOS

```bash
docker run --rm -v "$PWD:/app" mlops-assessment
```

## Output Files

After running, the program creates:

- `metrics.json`: structured metrics
- `run.log`: detailed execution logs

Example metrics:

```json
{
  "rows_processed": 9996,
  "signal_rate": 0.4983,
  "latency_ms": 45.32,
  "version": "v1"
}
```

## First Window Handling

Rolling mean uses the configured window size. The first `window - 1` rows produce missing rolling mean values and are excluded from signal computation.
