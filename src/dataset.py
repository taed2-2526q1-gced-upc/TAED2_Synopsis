from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import polars as pl

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

def remove_special_characters(dataframe: pl.DataFrame) -> pl.DataFrame:
    """
    Remove special emojis.
    """
    emojis = "[\U0001F600-\U0001F64F" \
        "\U0001F300-\U0001F5FF"  \
        "\U0001F680-\U0001F6FF" \
        "\U0001F1E0-\U0001F1FF"  \
        "\U00002700-\U000027BF"  \
        "\U000024C2-\U0001F251]+"
    

    dataframe = dataframe.with_columns(
        pl.col("article").str.replace_all(emojis, "", literal=False).alias("article"),
        pl.col("highlights").str.replace_all(emojis, "", literal=False).alias("highlights")
        )

    return dataframe

def remove_nulls_or_empty_or_duplicates(dataframe: pl.DataFrame) -> pl.DataFrame:
    """
    Remove duplicates or rows with nulls
    """

    dataframe = dataframe.filter(pl.col("article").is_not_null() & pl.col("highlights").is_not_null())
    dataframe = dataframe.filter((pl.col("article") != "") & (pl.col("highlights") != ""))
    dataframe = dataframe.unique("id")
    return dataframe.unique(subset=["article", "highlights"])


def remove_short_articles(dataframe:pl.DataFrame) -> pl.DataFrame:
    """
    Remove short articles
    """

    dataframe = dataframe.filter(pl.col("article").str.len_chars() > 50)
    return dataframe

@app.command()
def main():

    # ---- Preprocess data ----
    logger.info("Loading datasets...")

    df_train = pl.read_parquet(RAW_DATA_DIR / "train.parquet")

    df_val = pl.read_parquet(RAW_DATA_DIR / "validation.parquet")

    df_test = pl.read_parquet(RAW_DATA_DIR / "test.parquet")

    logger.success("Loading complete")

    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    splits = {
        "train": df_train,
        "val": df_val,
        "test": df_test
    }

    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)

    for name, df in tqdm(splits.items()):
        df_clean = remove_short_articles(remove_nulls_or_empty_or_duplicates(remove_special_characters((df))))
        df_clean.write_parquet(PROCESSED_DATA_DIR / f"clean_{name}.parquet")

    logger.success("Processing dataset complete.")
    # -----------------------------------------

if __name__ == "__main__":
    app()