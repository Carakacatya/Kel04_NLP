import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import os
import time

# =====================================================
# CONFIG
# =====================================================

DATASET_PATH = "dataset/jobs.csv"

EMBEDDINGS_OUT = "dataset/job_embeddings.npy"
DATAFRAME_OUT = "dataset/jobs_clean.pkl"

MODEL_NAME = "all-mpnet-base-v2"

# gunakan 50 ribu data dulu agar tidak berat
MAX_ROWS = 50000

BATCH_SIZE = 256


# =====================================================
# PREPROCESSING
# =====================================================

def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s,]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =====================================================
# MAIN
# =====================================================

def main():

    print("=" * 60)
    print("PRECOMPUTE EMBEDDINGS")
    print("=" * 60)

    # -------------------------------------------------
    # LOAD DATASET
    # -------------------------------------------------

    print("\n[1/4] Loading dataset...")

    t0 = time.time()

    needed_cols = [
        "Job Title",
        "Company",
        "location",
        "Job Description"
    ]

    df = pd.read_csv(
        DATASET_PATH,
        usecols=needed_cols,
        nrows=MAX_ROWS
    )

    print(
        f"✓ {len(df):,} baris dimuat "
        f"dalam {time.time()-t0:.2f}s"
    )

    # rename agar konsisten
    df.rename(
        columns={
            "Job Title": "job_title",
            "Company": "company",
            "Job Description": "description"
        },
        inplace=True
    )

    # -------------------------------------------------
    # PREPROCESS
    # -------------------------------------------------

    print("\n[2/4] Preprocessing...")

    t0 = time.time()

    df["clean_description"] = (
        df["description"]
        .fillna("")
        .apply(preprocess_text)
    )

    df = df[
        df["clean_description"].str.len() > 10
    ].reset_index(drop=True)

    print(
        f"✓ {len(df):,} data valid "
        f"dalam {time.time()-t0:.2f}s"
    )

    # -------------------------------------------------
    # LOAD MODEL
    # -------------------------------------------------

    print("\n[3/4] Loading model...")

    model = SentenceTransformer(MODEL_NAME)

    texts = df["clean_description"].tolist()

    print(
        f"Encoding {len(texts):,} job descriptions..."
    )

    t0 = time.time()

    job_embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print(
        f"✓ Encoding selesai "
        f"dalam {time.time()-t0:.2f}s"
    )

    print(
        f"✓ Shape embeddings: "
        f"{job_embeddings.shape}"
    )

    # -------------------------------------------------
    # SAVE
    # -------------------------------------------------

    print("\n[4/4] Menyimpan file...")

    os.makedirs("dataset", exist_ok=True)

    np.save(
        EMBEDDINGS_OUT,
        job_embeddings
    )

    df.to_pickle(
        DATAFRAME_OUT
    )

    print(
        f"✓ Saved: {EMBEDDINGS_OUT}"
    )

    print(
        f"✓ Saved: {DATAFRAME_OUT}"
    )

    print("\nSELESAI!")
    print("Sekarang jalankan:")
    print("python app.py")


if __name__ == "__main__":
    main()