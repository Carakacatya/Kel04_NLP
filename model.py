"""
model.py — NLP Core: Embedding + Cosine Similarity
"""

import re
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────
# Load model sekali saja saat startup
# ─────────────────────────────────────────────────────────────

print("[model.py] Memuat model SentenceTransformer...")
_model = SentenceTransformer('all-mpnet-base-v2')
print("[model.py] Model siap.")

_jobs_df = None
_job_embeddings = None


# ─────────────────────────────────────────────────────────────
# Load cached dataframe + embeddings
# ─────────────────────────────────────────────────────────────
def load_jobs_and_embeddings(
    df_path='dataset/jobs_clean.pkl',
    emb_path='dataset/job_embeddings.npy'
):

    global _jobs_df, _job_embeddings

    if _jobs_df is not None:
        return _jobs_df, _job_embeddings

    import os

    if not os.path.exists(df_path):
        raise FileNotFoundError(
            f"File tidak ditemukan: {df_path}\n"
            "Jalankan dulu: python precompute_embeddings.py"
        )

    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"File tidak ditemukan: {emb_path}\n"
            "Jalankan dulu: python precompute_embeddings.py"
        )

    t0 = time.time()

    print("[model.py] Loading dataframe...")
    _jobs_df = pd.read_pickle(df_path)

    print("[model.py] Loading embeddings...")
    _job_embeddings = np.load(emb_path)

    print(
        f"[model.py] Data siap "
        f"({_jobs_df.shape[0]:,} lowongan) "
        f"dalam {time.time()-t0:.2f}s"
    )

    return _jobs_df, _job_embeddings


# ─────────────────────────────────────────────────────────────
# Text Preprocessing
# ─────────────────────────────────────────────────────────────
def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(
        r'[^a-zA-Z0-9\s,]',
        '',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text


# ─────────────────────────────────────────────────────────────
# Recommendation Engine
# ─────────────────────────────────────────────────────────────
def get_recommendations(
    user_skill_input,
    top_n=8
):

    df, job_embeddings = load_jobs_and_embeddings()

    # STEP 1
    user_text = preprocess_text(user_skill_input)

    # STEP 2
    user_vec = _model.encode(
        [user_text],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # STEP 3
    similarities = np.dot(
        job_embeddings,
        user_vec.T
    ).flatten()

    # STEP 4
    result_df = df.copy()

    result_df["similarity_score"] = similarities

    result_df["similarity_pct"] = (
        similarities * 100
    ).round(1)

    result_df["label"] = result_df[
        "similarity_score"
    ].apply(
        lambda x:
        "Recommended"
        if x >= 0.40
        else "Not Recommended"
    )

    result_df = (
        result_df
        .sort_values(
            "similarity_score",
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    print("Kolom dataframe:")
    print(result_df.columns.tolist())

    return result_df