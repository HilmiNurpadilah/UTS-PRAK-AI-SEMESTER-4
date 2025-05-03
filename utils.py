import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from datetime import datetime

RESULTS_DIR = 'app/data/results/'

def plot_sentiment_distribution(df, sentiment_col='sentiment'):
    plt.figure(figsize=(6,4))
    sns.countplot(x=sentiment_col, data=df, palette='Set2')
    plt.title('Distribusi Sentimen')
    plt.xlabel('Sentimen')
    plt.ylabel('Jumlah')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'sentiment_distribution.png'))
    plt.close()

def plot_comment_activity(df, time_col='timestamp'):
    # Pastikan ada kolom timestamp (format datetime)
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df['date'] = df[time_col].dt.date
        activity = df.groupby('date').size()
        plt.figure(figsize=(8,4))
        activity.plot(kind='bar', color='skyblue')
        plt.title('Aktivitas Komentar per Hari')
        plt.xlabel('Tanggal')
        plt.ylabel('Jumlah Komentar')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'comment_activity.png'))
        plt.close()
    else:
        print(f"Kolom '{time_col}' tidak ditemukan di dataframe.")

def generate_wordcloud(df, text_col='stemmed', sentiment_col='sentiment'):
    # Pastikan direktori hasil ada
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sentiments = df[sentiment_col].unique()
    for sent in sentiments:
        text = ' '.join(df[df[sentiment_col]==sent][text_col].astype(str))
        wc = WordCloud(width=600, height=400, background_color='white', colormap='Set2').generate(text)
        plt.figure(figsize=(6,4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Wordcloud - {sent}')
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, f'wordcloud_{sent}.png')
        plt.savefig(save_path)
        print(f"Wordcloud untuk sentimen '{sent}' disimpan di: {save_path}")
        print("Wordcloud berhasil disimpan.")
        plt.close()

def plot_comment_activity_hour(df, time_col='timestamp'):
    # Visualisasi aktivitas komentar per jam
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df['hour'] = df[time_col].dt.hour
        activity = df.groupby('hour').size()
        plt.figure(figsize=(8,4))
        activity.plot(kind='bar', color='coral')
        plt.title('Aktivitas Komentar per Jam')
        plt.xlabel('Jam (0-23)')
        plt.ylabel('Jumlah Komentar')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'comment_activity_hour.png'))
        plt.close()
    else:
        print(f"Kolom '{time_col}' tidak ditemukan di dataframe.")

# Contoh cara pakai:
if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_csv('dataset_tiktok_processed.csv')
    plot_sentiment_distribution(df, sentiment_col='sentiment')
    plot_comment_activity(df, time_col='timestamp')  # Aktivitas per hari
    plot_comment_activity_hour(df, time_col='timestamp')  # Aktivitas per jam
    generate_wordcloud(df, text_col='stemmed', sentiment_col='sentiment')

# CATATAN: Abaikan FutureWarning dari seaborn, tidak mempengaruhi hasil visualisasi.