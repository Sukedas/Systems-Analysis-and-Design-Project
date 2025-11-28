# Workshop 4 Simulation

This project implements a simulation software package for Workshop 4, featuring:
1.  **Data-driven ML Simulation**: Random Forest and MLP models with drift detection.
2.  **Event-based CA Simulation**: Cellular Automata model for microenterprise density.
3.  **System Integration**: Mapping to an 8-layer architecture.

## Setup

### Local
1.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Docker
1.  Build the image:
    ```bash
    docker build -t workshop4-sim .
    ```
2.  Run the container:
    ```bash
    docker run --rm -v $(pwd)/reports:/app/reports workshop4-sim
    ```

## Usage

### Run ML Simulation
```bash
python run_ml_simulation.py --config config/ml_config.yaml
```
Outputs metrics and plots to `reports/experiments/<timestamp>/`.

### Run CA Simulation
```bash
python run_ca_simulation.py --config config/ca_config.yaml
```
Outputs snapshots and animation to `reports/figs/`.

### Run Tests
```bash
pytest
```

## Report
The final report is available in `reports/Workshop_4_Report.pdf` (compile from `.tex`).
