from flask import Flask, render_template, request

import json
import ast

from model import load_jobs_and_embeddings, get_recommendations

app = Flask(__name__)

def clean_display(value):

    if value is None:
        return ""

    value = str(value).strip()

    if value == "":
        return ""

    # JSON
    try:
        obj = json.loads(value)

        if isinstance(obj, dict):

            text = ""

            for k, v in obj.items():
                text += f"{k}: {v}\n"

            return text.strip()

    except:
        pass

    # Python object
    try:

        obj = ast.literal_eval(value)

        if isinstance(obj, dict):

            text = ""

            for k, v in obj.items():
                text += f"{k}: {v}\n"

            return text.strip()

        elif isinstance(obj, (list, tuple, set)):

            return ", ".join(sorted(list(obj)))

    except:
        pass

    return value

# =====================================================
# PRELOAD DATA
# =====================================================

print("=" * 60)
print("NLP JOB RECOMMENDER")
print("=" * 60)

try:
    load_jobs_and_embeddings()
    print("[OK] Dataset & Embeddings berhasil dimuat.")
except Exception as e:
    print("[ERROR] Gagal preload data")
    print(e)

print("=" * 60)


# =====================================================
# HOME
# =====================================================

@app.route("/")
def index():
    return render_template("index.html")


# =====================================================
# SEARCH
# =====================================================

@app.route("/search", methods=["POST"])
def search():

    try:

        user_skill = request.form.get("skill_input", "").strip()

        if user_skill == "":
            return render_template(
                "index.html",
                error="Silakan masukkan skill terlebih dahulu."
            )

        if len(user_skill) < 3:
            return render_template(
                "index.html",
                error="Minimal 3 karakter."
            )

        results = get_recommendations(
            user_skill,
            top_n=8
        )

        result_list = results.to_dict(
            orient="records"
        )

        for job in result_list:

            job["Benefits"] = clean_display(
                job.get("Benefits", "")
            )

            job["Company Profile"] = clean_display(
                job.get("Company Profile", "")
            )

        return render_template(
            "result.html",
            skill_input=user_skill,
            results=result_list
        )

    except Exception as e:

        print("=" * 60)
        print("ERROR SEARCH")
        print(e)
        print("=" * 60)

        return render_template(
            "index.html",
            error=str(e)
        )


# =====================================================
# ERROR PAGE
# =====================================================

@app.errorhandler(404)
def page_not_found(e):
    return render_template(
        "index.html",
        error="Halaman tidak ditemukan."
    ), 404


@app.errorhandler(500)
def internal(e):
    return render_template(
        "index.html",
        error="Terjadi kesalahan pada server."
    ), 500


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )