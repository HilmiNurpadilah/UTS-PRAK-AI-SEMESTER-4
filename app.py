import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

import logging

logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = 'app/data/uploads'
RESULTS_FOLDER = 'app/data/results'
ALLOWED_EXTENSIONS = {'csv'}

# Path file untuk komentar user
USER_COMMENTS_PATH = 'user_comments.csv'

app = Flask(__name__)
app.secret_key = 'secret-key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Komentar akan disimpan di session sementara
COMMENTS_KEY = 'comments'

# Pastikan folder upload ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model dan tokenizer jika ada
MODEL_PATH = 'sentiment_lstm.h5'
TOKENIZER_PATH = 'tokenizer.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

model = None
tokenizer = None
label_encoder = None
if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH) and os.path.exists(LABEL_ENCODER_PATH):
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    grafik_paths = {
        'sentiment': os.path.join(RESULTS_FOLDER, 'sentiment_distribution.png'),
        'activity': os.path.join(RESULTS_FOLDER, 'comment_activity.png'),
        'wordcloud_negatif': os.path.join(RESULTS_FOLDER, 'wordcloud_negatif.png'),
        'wordcloud_netral': os.path.join(RESULTS_FOLDER, 'wordcloud_netral.png'),
        'wordcloud_positif': os.path.join(RESULTS_FOLDER, 'wordcloud_positif.png'),
    }
    # Baca komentar user dari file user_comments.csv
    user_comments = []
    if os.path.exists(USER_COMMENTS_PATH):
        try:
            df = pd.read_csv(USER_COMMENTS_PATH)
            if 'komentar' in df.columns and 'sentimen' in df.columns:
                user_comments = [(row['komentar'], row['sentimen']) for _, row in df.iterrows()]
                komentar_neg = df[df['sentiment'].str.lower() == 'negatif'].sample(n=min(7, len(df[df['sentiment'].str.lower() == 'negatif'])), random_state=2)
                komentar_net = df[df['sentiment'].str.lower() == 'netral'].sample(n=min(6, len(df[df['sentiment'].str.lower() == 'netral'])), random_state=3)
                gabung = pd.concat([komentar_pos, komentar_neg, komentar_net]).sample(frac=1, random_state=42) # acak urutan
                csv_comments = [(row['komentar'], row['sentiment']) for _, row in gabung.iterrows()]
            else:
                # fallback: ambil 20 komentar pertama jika kolom tidak lengkap
                csv_comments = [(row[0], row[1] if len(row) > 1 else '-') for row in df.values[:20]]
        except Exception as e:
            csv_comments = [(f'Gagal membaca CSV: {e}', '-')]
    # Kirim user_comments ke template
    analysis_result = session.pop('analysis_result', None)
    return render_template('index.html', grafik_paths=grafik_paths, comments=user_comments, analysis_result=analysis_result)


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Proses dan tampilkan grafik
        df = pd.read_csv(filepath)
        # Cek kolom yang ada
        if 'sentiment' in df.columns:
            from utils import plot_sentiment_distribution, plot_comment_activity, generate_wordcloud
            plot_sentiment_distribution(df, sentiment_col='sentiment')
            plot_comment_activity(df, time_col='timestamp' if 'timestamp' in df.columns else df.columns[0])
            generate_wordcloud(df, text_col='stemmed' if 'stemmed' in df.columns else df.columns[0], sentiment_col='sentiment')
        flash('File uploaded and processed!')
        return redirect(url_for('index'))
    else:
        flash('Invalid file type!')
        return redirect(url_for('index'))

@app.route('/add_comment', methods=['POST'])
def add_comment():
    comment = request.form.get('comment')
    if comment:
        # Prediksi sentimen jika model tersedia
        sentimen = '-'
        if model and tokenizer and label_encoder:
            try:
                seq = tokenizer.texts_to_sequences([comment])
                padded = pad_sequences(seq, maxlen=100)
                pred = model.predict(padded)
                sentimen = label_encoder.inverse_transform([np.argmax(pred)])[0]
            except Exception as e:
                sentimen = f'error: {e}'
        # Simpan ke file user_comments.csv
        import csv
        user_comments_path = os.path.join('user_comments.csv')
        file_exists = os.path.isfile(user_comments_path)
        with open(user_comments_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(['komentar', 'sentimen'])
            writer.writerow([comment, sentimen])
    return redirect(url_for('index'))


@app.route('/analyze_lstm', methods=['POST'])
def analyze_lstm():
    text = request.form.get('analyze_text')
    if not text or not model or not tokenizer or not label_encoder:
        session['analysis_result'] = 'LSTM model not available or input kosong.'
        return redirect(url_for('index'))
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    pred = model.predict(padded)
    label = label_encoder.inverse_transform([np.argmax(pred)])[0]
    session['analysis_result'] = f"Prediksi sentimen: <b>{label}</b>"
    return redirect(url_for('index'))

@app.route('/results/<filename>')
def results_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))