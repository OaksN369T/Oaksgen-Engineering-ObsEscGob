FROM python:3.10-slim

WORKDIR /app

COPY pipeline_fase3_oaksgen.py test_pipeline.py ./

RUN pip install --no-cache-dir fastapi httpx pytest mlflow xgboost scikit-learn pandas numpy matplotlib

EXPOSE 8000

CMD ["pytest", "-v", "test_pipeline.py"]
