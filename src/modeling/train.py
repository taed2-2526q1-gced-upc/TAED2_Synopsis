from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import mlflow
from sklearn.linear_model import LogisticRegression

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- MLflow dummy training ----
    with mlflow.start_run():
        logger.info("Training dummy model...")
        
        model = LogisticRegression(C=1.0, max_iter=100, random_state=42)

        mlflow.log_param("C", 1.0)
        mlflow.log_param("max_iter", 100)
        mlflow.log_param("random_state", 42)

        logger.success("Dummy model training complete with MLflow logging.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
