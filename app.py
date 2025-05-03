import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

UPLOAD_FOLDER = 'app/data/uploads'
RESULTS_FOLDER = 'app/data/results'
ALLOWED_EXTENSIONS = {'csv'}

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
    # Ambil maksimal 15 komentar dari file dataset_tiktok_processed.csv, pastikan ada positif, negatif, netral jika tersedia
    csv_comments = []
    try:
        df = pd.read_csv('dataset_tiktok_processed.csv')
        if 'komentar' in df.columns and 'sentiment' in df.columns:
            # Ambil satu contoh positif, negatif, netral jika ada
            positif = df[df['sentiment'].str.lower() == 'positif'].head(1)
            negatif = df[df['sentiment'].str.lower() == 'negatif'].head(1)
            netral = df[df['sentiment'].str.lower() == 'netral'].head(1)
            sisa = df.drop(positif.index.union(negatif.index).union(netral.index)).head(12)
            sample_df = pd.concat([positif, negatif, netral, sisa])
            csv_comments = sample_df[['komentar', 'sentiment']].values.tolist()
        elif 'komentar' in df.columns:
            csv_comments = [[row, '-'] for row in df['komentar'].astype(str).head(15).tolist()]
    except Exception as e:
        pass
    # Ambil komentar baru dari session
    user_comments = session.get(COMMENTS_KEY, [])
    # Daftar utama hanya dari dataset_tiktok_processed.csv, komentar user hanya tampil setelah submit
    comments = csv_comments + user_comments
    analysis_result = session.pop('analysis_result', None)
    return render_template('index.html', grafik_paths=grafik_paths, comments=comments, analysis_result=analysis_result)

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
    comments = session.get(COMMENTS_KEY, [])
    if comment:
        # Cek sentimen dengan LSTM jika model tersedia
        sentiment = '-'
        if model and tokenizer and label_encoder:
            seq = tokenizer.texts_to_sequences([comment])
            padded = pad_sequences(seq, maxlen=100)
            pred = model.predict(padded)
            sentiment = label_encoder.inverse_transform([np.argmax(pred)])[0]
        comments.append([comment, sentiment])
        session[COMMENTS_KEY] = comments
    return redirect(url_for('index'))


@app.route('/results/<filename>')
def results_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
