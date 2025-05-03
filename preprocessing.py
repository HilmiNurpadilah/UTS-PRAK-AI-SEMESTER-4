import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import pipeline

# Preprocessing functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)     # remove symbols, emoji, angka
    text = re.sub(r'\s+', ' ', text).strip()    # remove extra spaces
    return text

def stemming(text, stemmer):
    return ' '.join([stemmer.stem(word) for word in text.split()])

# Load dataset
import csv
try:
    df = pd.read_csv('dataset tiktok.csv', engine='python', quoting=csv.QUOTE_ALL, on_bad_lines='skip')
except Exception as e:
    print('Gagal membaca dataset tiktok.csv:', e)
    exit(1)

# Tambahkan kolom timestamp dari createTimeISO
if 'createTimeISO' in df.columns:
    df['timestamp'] = df['createTimeISO']
else:
    df['timestamp'] = None

# Preprocessing
factory = StemmerFactory()
stemmer = factory.create_stemmer()
df['cleaned'] = df['komentar'].astype(str).apply(clean_text)
df['stemmed'] = df['cleaned'].apply(lambda x: stemming(x, stemmer))

# IndoBERT Sentiment Labeling
def indo_label(label):
    label = label.lower()
    if label == 'positive':
        return 'positif'
    elif label == 'negative':
        return 'negatif'
    elif label == 'neutral':
        return 'netral'
    else:
        return 'netral'

# Sentiment analysis dengan Roberta IndoBERT
sentiment_model = pipeline("sentiment-analysis", model="w11wo/indonesian-roberta-base-sentiment-classifier")
def get_sentiment_label(text):
    if not isinstance(text, str) or text.strip() == '':
        return 'netral'
    try:
        label = sentiment_model(text)[0]['label'].lower()
        return indo_label(label)
    except Exception as e:
        print(f'Error pada komentar: "{text}" => {e}')
        return 'netral'

df['sentiment'] = df['stemmed'].apply(get_sentiment_label)

# Pastikan hanya tiga label yang digunakan
allowed_labels = ['positif', 'negatif', 'netral']
df['sentiment'] = df['sentiment'].apply(lambda x: x if x in allowed_labels else 'netral')

# Save processed data
df.to_csv('dataset_tiktok_processed.csv', index=False)