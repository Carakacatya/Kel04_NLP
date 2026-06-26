import os
import re
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ==========================================================
# MODEL YANG HARUS SAMA DENGAN precompute_embeddings.py
# ==========================================================
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("[model] Loading SentenceTransformer...")
model = SentenceTransformer(MODEL_NAME)
print("[model] Model loaded.")

jobs_df = None
job_embeddings = None


# ==========================================================
# TEXT PREPROCESS
# ==========================================================
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ==========================================================
# LOAD DATASET
# ==========================================================
def load_jobs_and_embeddings():

    global jobs_df
    global job_embeddings

    if jobs_df is not None:
        return jobs_df, job_embeddings

    df_path = "dataset/jobs_clean.pkl"
    emb_path = "dataset/job_embeddings.npy"

    if not os.path.exists(df_path):
        raise FileNotFoundError(
            "jobs_clean.pkl tidak ditemukan.\n"
            "Silakan jalankan precompute_embeddings.py terlebih dahulu."
        )

    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            "job_embeddings.npy tidak ditemukan.\n"
            "Silakan jalankan precompute_embeddings.py terlebih dahulu."
        )

    start = time.time()

    jobs_df = pd.read_pickle(df_path)
    job_embeddings = np.load(emb_path)

    print(f"[model] Dataset : {jobs_df.shape}")
    print(f"[model] Embeddings : {job_embeddings.shape}")
    print(f"[model] Loaded in {time.time()-start:.2f} sec")

    return jobs_df, job_embeddings


# ==========================================================
# RECOMMENDATION
# ==========================================================
def get_recommendations(user_skill, top_n=8):

    jobs_df, job_embeddings = load_jobs_and_embeddings()

    query = preprocess_text(user_skill)

    user_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarity = np.dot(job_embeddings, user_embedding.T).flatten()

    result = jobs_df.copy()

    # skor similarity
    result["similarity_score"] = similarity
    result["similarity_pct"] = (similarity * 100).round(2)

    # label rekomendasi
    result["label"] = result["similarity_score"].apply(
        lambda x: "Recommended" if x >= 0.40 else "Not Recommended"
    )

    # urutkan dari skor terbesar
    result = (
        result.sort_values(
            "similarity_score",
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return result