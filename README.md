# Analisis Sentimen Teks Berbahasa Indonesia dengan Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange.svg)](https://tensorflow.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8.1-green.svg)](https://www.nltk.org/)

## Deskripsi Proyek
Proyek ini merupakan implementasi dan komparasi berbagai arsitektur Deep Learning untuk tugas Analisis Sentimen pada ulasan berbahasa Indonesia. Proyek ini membandingkan tiga skema pemodelan yang berbeda (LSTM, BiLSTM, dan CNN-LSTM) dipadukan dengan berbagai teknik ekstraksi fitur (TF-IDF, Word2Vec, dan Embedding Layer) untuk menemukan arsitektur paling optimal.

Data ulasan diekstraksi secara otomatis menggunakan `google-play-scraper` dan diproses melalui tahapan *text preprocessing* bahasa Indonesia menggunakan `PySastrawi` dan `NLTK`.

## Dampak Proyek untuk Bisnis

Proyek ini tidak hanya sebatas eksperimen teknis pemodelan, tetapi juga dirancang untuk memberikan dampak nyata dalam ekosistem bisnis dan layanan digital. Beberapa nilai tambah dari implementasi proyek ini meliputi:

* **Efisiensi Umpan Balik Pelanggan:** Mengotomatisasi proses klasifikasi ribuan ulasan pengguna aplikasi yang sebelumnya memakan waktu berjam-jam jika dilakukan secara manual.
* **Pengambilan Keputusan Berbasis Data (Data-Driven):** Membantu tim pengembang produk dan customer relations untuk dengan cepat mengidentifikasi keluhan utama (seperti bug aplikasi atau masalah layanan) berdasarkan sentimen negatif yang terdeteksi.
* **Peningkatan Kualitas Produk:** Implementasi Artificial Intelligence dalam memahami sentimen pengguna merupakan langkah esensial dalam membangun ekonomi digital yang lebih pintar. Hal ini dapat menjadi pendorong bagi para pengusaha masa depan dan perusahaan teknologi untuk terus menyempurnakan produk mereka di pasar yang semakin kompetitif.

## Teknologi & Pustaka
Proyek ini dibangun menggunakan berbagai pustaka Data Science dan Machine Learning Python:
* **Pemodelan Deep Learning:** `tensorflow`, `scikit-learn`
* **NLP & Ekstraksi Fitur:** `nltk`, `PySastrawi`, `gensim`
* **Scraping Data:** `google-play-scraper`
* **Pengolahan Data & Visualisasi:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `wordcloud`

## Skema Arsitektur Model
Proyek ini mengeksplorasi tiga skema utama untuk mengklasifikasikan sentimen:

1. **Skema 1: LSTM + TF-IDF**
   - **Algoritma:** Long Short-Term Memory (LSTM)
   - **Ekstraksi Fitur:** Term Frequency-Inverse Document Frequency (TF-IDF)
   - **Rasio Split Data:** 80/20

2. **Skema 2: BiLSTM + Word2Vec**
   - **Algoritma:** Bidirectional LSTM (BiLSTM)
   - **Ekstraksi Fitur:** Word2Vec (Gensim)
   - **Rasio Split Data:** 80/20

3. **Skema 3: CNN-LSTM + Embedding Layer**
   - **Algoritma:** Gabungan Convolutional Neural Network (CNN) dan LSTM
   - **Ekstraksi Fitur:** Keras Embedding Layer
   - **Rasio Split Data:** 70/30

## Hasil dan Evaluasi
Berdasarkan metrik evaluasi yang dikumpulkan dari pengujian model, berikut adalah ringkasan performa masing-masing skema:

| Skema | Algoritma | Ekstraksi Fitur | Train Accuracy | Test Accuracy | F1-Macro |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Skema 1** | LSTM | TF-IDF | 97.50% | 86.00% | 85.85 |
| **Skema 2** | BiLSTM | Word2Vec | 92.47% | **88.37%** | **88.34** |
| **Skema 3** | CNN-LSTM | Embedding | 97.49% | 87.89% | 87.77 |

**Kesimpulan Evaluasi:**
Skema 2 (BiLSTM + Word2Vec) menunjukkan performa generalisasi terbaik pada data uji dengan akurasi pengujian mencapai 88.37% dan skor F1-Macro tertinggi sebesar 88.34. Hal ini menunjukkan bahwa pendekatan representasi semantik (Word2Vec) digabungkan dengan pemrosesan urutan kata dua arah (BiLSTM) sangat efektif menangkap konteks pada teks ulasan. Skema 1 dan Skema 3 cenderung mengalami *overfitting* yang lebih tinggi, terlihat dari akurasi pelatihan yang sangat tinggi (>97%) namun tidak selaras dengan performa pengujiannya.

## Cara Menjalankan Proyek

1. **Clone repositori ini:**
   ```bash
   git clone https://github.com/Bimzt/Indonesian-Review-Sentiment-Analysis
   cd Indonesian-Review-Sentiment-Analysis
   ```
2. **Instal pustaka yang dibutuhkan:**
Sangat disarankan menggunakan virtual environment.
   ```Bash
   pip install -r requirements.txt
   ```

Alur Eksekusi Notebook:
- Jalankan file Jupyter Notebook sesuai urutan berikut:
- Scraping.ipynb: Untuk mengambil raw data dari Google Play.
- Preprocessing.ipynb: Untuk membersihkan data, stopword removal, dan stemming.
- Skenario_1_LSTM_TFIDF.ipynb: Melatih dan mengevaluasi Skema 1.
- Skenario_2_BiLSTM_Word2Vec.ipynb: Melatih dan mengevaluasi Skema 2.
- Skenario_3_CNNLSTM_Embedding.ipynb: Melatih dan mengevaluasi Skema 3.
- Inference.ipynb: Menggunakan model terbaik yang telah disimpan (berformat .keras) untuk melakukan prediksi pada teks sentimen baru.

## Struktur Direktori
```Bash
├── data/                    # Berisi dataset mentah dan dataset yang telah diproses
├── models/                  # Direktori penyimpanan model (.keras), tokenizer, dan vectorizer
├── reports/                 # Hasil inferensi dan log evaluasi (.json, .csv)
├── Scraping.ipynb           # Kode pengambilan dataset
├── Preprocessing.ipynb      # Pembersihan teks & tokenisasi
├── Skenario_1...            # Skema 1
├── Skenario_2...            # Skema 2
├── Skenario_3...            # Skema 3
├── Inference.ipynb          # Implementasi model untuk prediksi
└── requirements.txt         # Daftar *dependencies*
```
