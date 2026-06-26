import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

CSV_PATH = "dataset/jobs.csv"
OUTPUT_DF = "dataset/jobs_clean.pkl"
OUTPUT_EMB = "dataset/job_embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def build_job_text(row):
    parts = [
        row.get("Job Title", ""),
        row.get("Role", ""),
        row.get("skills", ""),
        row.get("Qualifications", ""),
        str(row.get("Job Description", ""))[:800],
        str(row.get("Responsibilities", ""))[:500],
        row.get("Benefits", ""),
        row.get("Work Type", ""),
    ]

    parts = [clean_text(x) for x in parts if pd.notna(x) and str(x).strip()]
    return " | ".join(parts)

    parts = [clean_text(x) for x in parts if pd.notna(x) and str(x).strip()]
    return " | ".join(parts)


def main():
    print("[precompute] Loading CSV...")

    # baca csv lebih toleran
    try:
        df = pd.read_csv(
            CSV_PATH,
            engine="python",
            encoding="utf-8",
            on_bad_lines="skip"
        )
    except Exception:
        df = pd.read_csv(
            CSV_PATH,
            engine="python",
            encoding="latin-1",
            on_bad_lines="skip"
        )

    print(f"[precompute] CSV loaded: {df.shape}")
    # Ambil 3.000 data saja agar ringan
    df = df.sample(n=3000, random_state=42)
    print(f"[precompute] Using sample: {df.shape}")
    print("[precompute] Kolom terdeteksi:")
    print(df.columns.tolist())

    # rapikan nama kolom
    df.columns = [col.strip() for col in df.columns]

    # isi NaN biar aman
    text_cols = [
        "Job Title", "Role", "Job Description", "skills", "Responsibilities",
        "Qualifications", "Benefits", "Company", "Company Profile",
        "Salary Range", "Experience", "location", "Country", "Work Type",
        "Preference"
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # buat kolom gabungan untuk embedding
    print("[precompute] Building combined text for embeddings...")
    df["combined_text"] = df.apply(build_job_text, axis=1)

    # buang baris kosong
    df = df[df["combined_text"].str.strip() != ""].copy()
    df = df.reset_index(drop=True)

    print(f"[precompute] Data after cleaning: {df.shape}")

    # load model
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    print("[precompute] Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # encode
    print("[precompute] Encoding job texts...")
    embeddings = model.encode(
        df["combined_text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # simpan
    os.makedirs("dataset", exist_ok=True)
    df.to_pickle(OUTPUT_DF)
    np.save(OUTPUT_EMB, embeddings)

    print(f"[precompute] Saved cleaned dataframe -> {OUTPUT_DF}")
    print(f"[precompute] Saved embeddings -> {OUTPUT_EMB}")
    print("[precompute] DONE.")


if __name__ == "__main__":
    main()