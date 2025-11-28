FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directory for reports if it doesn't exist (though it should be mounted)
RUN mkdir -p reports/experiments reports/figs

CMD ["python", "run_ml_simulation.py", "--config", "config/ml_config.yaml"]
