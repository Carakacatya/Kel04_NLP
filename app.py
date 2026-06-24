"""
app.py — Flask web server

Optimasi vs versi lama:
  ✓ Embeddings & dataframe di-load sekali saat startup (bukan tiap request)
  ✓ Tiap request hanya encode 1 kalimat (input user), bukan seluruh dataset
  ✓ Cosine similarity pakai np.dot (lebih cepat dari sklearn karena sudah normalized)
"""

from flask import Flask, render_template, request, jsonify
from model import load_jobs_and_embeddings, get_recommendations

app = Flask(__name__)

# ── Pre-load data saat startup (bukan saat request pertama) ──────────────────
print("[app.py] Pre-loading dataset & embeddings...")
load_jobs_and_embeddings()   # panggil sekali agar di-cache di memori
print("[app.py] Siap menerima request.")


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    user_skill = request.form.get('skill_input', '').strip()

    if not user_skill:
        return render_template('index.html', error="Harap masukkan skill terlebih dahulu.")

    if len(user_skill) < 3:
        return render_template('index.html', error="Masukkan minimal 3 karakter.")

    results = get_recommendations(user_skill, top_n=8)
    results_list = results.to_dict(orient='records')

    return render_template(
        'result.html',
        skill_input=user_skill,
        results=results_list,
    )


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)