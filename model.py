import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Load processed data
df = pd.read_csv('dataset_tiktok_processed.csv')

# Encode label
# Pastikan label hanya terdiri dari 'positif', 'negatif', 'netral' (konsisten dengan preprocessing.py)
label_set = ['negatif', 'netral', 'positif']
df['sentiment'] = df['sentiment'].astype(str).str.lower().apply(lambda x: x if x in label_set else 'netral')
le = LabelEncoder()
le.fit(label_set)
df['label'] = le.transform(df['sentiment'])

# Pastikan kolom 'stemmed' tidak ada NaN dan semuanya string
# Ini mencegah error AttributeError: 'float' object has no attribute 'lower'
df['stemmed'] = df['stemmed'].fillna('').astype(str)

# Tokenization
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['stemmed'])
X = tokenizer.texts_to_sequences(df['stemmed'])
X = pad_sequences(X, maxlen=100)
y = to_categorical(df['label'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=100),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Save model & tokenizer
model.save('sentiment_lstm.h5')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Evaluation
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Mapping label angka ke sentimen
print("\nMapping label angka ke sentimen:")
for idx, sentimen in enumerate(le.classes_):
    print(f"{idx} = {sentimen}")

# Pastikan semua kelas ditampilkan meskipun tidak muncul di data test
labels = list(range(len(le.classes_)))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_, labels=labels, zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=labels))