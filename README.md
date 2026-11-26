# Analisis Sentimen Ulasan Aplikasi E-Commerce Indonesia dengan Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8.1%2B-green)](https://www.nltk.org/)
[![Sastrawi](https://img.shields.io/badge/PySastrawi-Indonesian%20NLP-blue)](https://github.com/har07/PySastrawi)

> Proyek *Natural Language Processing (NLP)* & *Deep Learning* **end-to-end** untuk mengklasifikasikan sentimen ulasan pengguna berbahasa Indonesia (**Negatif**, **Netral**, **Positif**) dari Google Play Store (**Tokopedia**, **Shopee**, **Bukalapak**), membandingkan tiga arsitektur neural network (**LSTM + TF-IDF**, **BiLSTM + Word2Vec**, dan **Hybrid CNN-LSTM + Embedding Layer**).

---

## Latar Belakang & Permasalahan

Platform e-commerce di Indonesia menerima puluhan ribu ulasan pengguna setiap minggu di Google Play Store yang berisi pujian, saran perbaikan, hingga laporan bug kritis. Teks ulasan berbahasa Indonesia memiliki karakteristik khusus: banyak singkatan informal (`yg`, `gak`, `bgt`, `udh`, `klo`), imbuhan kompleks, serta ketimpangan kelas sentimen yang ekstrem (ulasan bintang 3/netral jauh lebih sedikit dibandingkan ulasan ekstrem bintang 1 dan bintang 5).

Membaca ribuan ulasan secara manual tidak efisien dan rentan bias. Proyek ini membangun pipeline otomatis mulai dari pengambilan data, *text preprocessing* bahasa Indonesia, penyeimbangan kelas, hingga komparasi tiga arsitektur *Deep Learning* untuk menemukan model dengan generalisasi terbaik.

---

## Tujuan & Dampak Bisnis

- **Otomatisasi Analisis Umpan Balik:** Mengklasifikasikan ribuan ulasan aplikasi secara otomatis dalam hitungan detik menjadi sentimen `Negatif`, `Netral`, dan `Positif`.
- **Deteksi Keluhan Kritis Lebih Cepat:** Memungkinkan tim produk dan *customer experience* mengidentifikasi keluhan utama (seperti crash aplikasi, masalah pengiriman, atau hilangnya metode pembayaran COD) secara berbasis data.
- **Benchmarking Arsitektur NLP Bahasa Indonesia:** Memberikan perbandingan empiris antara representasi kata berbasis statistik/sekuens dan *dense semantic embeddings* (`Word2Vec` vs. `Embedding Layer`).

---

## Pengumpulan & Preprocessing Data

### 1. Web Scraping (`Scraping.ipynb`)
- Menggunakan `google-play-scraper` (`lang='id'`, `country='id'`, `sort=Sort.NEWEST`) dengan target 4.000 ulasan per aplikasi pada:
  - **Tokopedia** (`com.tokopedia.tkpd`)
  - **Shopee** (`com.shopee.id`)
  - **Bukalapak** (`com.bukalapak.android`)
- Dari 12.000 ulasan yang diambil, setelah deduplikasi `review_id` dan filter panjang teks (`> 5` karakter), terkumpul **10.831 ulasan mentah** (`Bukalapak`: 3.708, `Tokopedia`: 3.605, `Shopee`: 3.518) dengan distribusi rating: ★1 (`4.545`), ★2 (`576`), ★3 (`520`), ★4 (`454`), ★5 (`4.736`).

### 2. Pelabelan & Text Preprocessing (`Preprocessing.ipynb`)
- **Pelabelan Berbasis Rating:** Skor `1–2` → `Negatif` (`5.121`), Skor `3` → `Netral` (`520`), Skor `4–5` → `Positif` (`5.190`).
- **Pembersihan Teks:** Lowercasing, penghapusan URL/email/mention/hashtag/emoji, pembersihan karakter non-alfabet, penghapusan *stopword* (`PySastrawi` + daftar kata gaul `EXTRA_STOPWORDS` dan filter token `len > 2`), serta *stemming* bahasa Indonesia dengan `Sastrawi`. Setelah membuang baris kosong (`len <= 3`), tersisa **10.769 sampel bersih** (`Positif`: 5.142, `Negatif`: 5.109, `Netral`: 518).
- **Penanganan Class Imbalance (`smart_balance`):** Penyeimbangan ke target **5.000 sampel per kelas** (undersampling `Positif` & `Negatif`, oversampling `Netral`) sehingga dataset final berjumlah **15.000 sampel seimbang** (`0`: Negatif, `1`: Netral, `2`: Positif).

---

## Arsitektur Tiga Skenario Pemodelan

Seluruh skenario dilatih menggunakan panjang sekuens `MAX_SEQ_LEN = 100`, optimizer `Adam(lr=0.001)`, loss `categorical_crossentropy`, serta callback `EarlyStopping`, `ModelCheckpoint`, dan `ReduceLROnPlateau`:

| Skenario | Notebook | Split Data | Fitur & Representasi | Arsitektur Utama |
| :--- | :--- | :---: | :--- | :--- |
| **Skema 1** | `Skenario_1_LSTM_TFIDF.ipynb` | **80/20** (12.000 train / 1.500 val / 1.500 test) | `TfidfVectorizer` (`ngram=(1,2)`, max 6.000) + Keras `Tokenizer` (vocab 7.115) & Embedding (128-d) | `SpatialDropout1D(0.4)` → `BiLSTM(64)` → `BiLSTM(32)` → `BatchNorm` → `Dense(64)` |
| **Skema 2** | `Skenario_2_BiLSTM_Word2Vec.ipynb` | **80/20** (12.000 train [15% val] / 3.000 test) | **Gensim Word2Vec** (200-d, window=5, min_count=2, vocab 3.811, **92.0% coverage**) | `Word2Vec Embedding(200-d, trainable=True)` → `SpatialDropout1D(0.2)` → `BiLSTM(128)` → `BiLSTM(64)` → `Dense(64)` |
| **Skema 3** | `Skenario_3_CNNLSTM_Embedding.ipynb` | **70/30** (10.500 train [15% val] / 4.500 test) | Keras Trainable **Embedding Layer** (128-d, vocab 6.674) | Arsitektur paralel bercabang: Branch 1 (`Conv1D(128, k=3)` → `LSTM(128)`) + Branch 2 (`Conv1D(64, k=5)` → `GlobalMaxPooling1D`) → `Concatenate` → `Dense(128, L2)` → `Dense(64)` |

---

## Hasil Evaluasi & Inferensi

### 1. Perbandingan Performa Model (`reports/scheme*_results.json`)

| Skema | Algoritma | Ekstraksi Fitur | Split | Train Accuracy | Test Accuracy | F1-Macro |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Skema 1** | LSTM (BiLSTM seq) | TF-IDF / Sequence | 80/20 | 97.50% | 86.00% | 85.85% |
| **Skema 2** | **BiLSTM** | **Word2Vec** | **80/20** | **92.47%** | **88.37%** | **88.34%** |
| **Skema 3** | CNN-LSTM Hybrid | Embedding Layer | 70/30 | 97.49% | 87.89% | 87.77% |

- **Detail Per Kelas Model Terbaik (Skema 2 — `3.000` data uji):**
  - `Negatif`: Precision `0.83`, Recall `0.88`, F1-Score `0.85`
  - `Netral`: Precision `0.94`, Recall `0.95`, F1-Score `0.94`
  - `Positif`: Precision `0.89`, Recall `0.82`, F1-Score `0.85`
- **Analisis Perbandingan:** **Skema 2 (BiLSTM + Word2Vec)** menghasilkan akurasi uji (**88.37%**) dan F1-Macro (**88.34%**) tertinggi sekaligus celah *train–test* terkecil (`4.10%`), membuktikan bahwa representasi semantik `Word2Vec` + pemrosesan dua arah `BiLSTM` paling mampu menangkap konteks ulasan tanpa overfitting berlebih (dibandingkan Skema 1 dan Skema 3 yang mencapai akurasi latih >97.4% namun gap generalisasi lebih lebar).

### 2. Hasil Pengujian Inferensi (`Inference.ipynb` & `reports/inference_results.csv`)
Pengujian batch pada 12 kalimat ulasan baru (positif, netral, dan negatif kustom) menghasilkan:
- Kalimat positif eksplisit dan keluhan negatif kritis (seperti *"Aplikasi sering crash..."*, *"Fitur COD nya hilang..."*) terklasifikasi dengan tepat oleh ketiga skema (confidence **97%–100%** pada Skema 3).
- Pada `reports/inference_results.csv` (menggunakan model Skema 3), dari 12 sampel uji terprediksi **7 Positif** dan **5 Negatif**. Kalimat ambigu/netral (misalnya *"Biasa aja, tidak ada yang istimewa..."*) cenderung lebih mudah meleset ke kelas Positif/Negatif pada Skema 3 dibandingkan Skema 1 & 2, dipengaruhi oleh tingginya rasio *oversampling* kelas Netral (`518` → `5.000`) saat pelatihan.

---

## Struktur Direktori

```text
Indonesian-Review-Sentiment-Analysis/
├── reports/
│   ├── scheme1_results.json              # Metrik evaluasi Skema 1 (86.00% test acc)
│   ├── scheme2_results.json              # Metrik evaluasi Skema 2 (88.37% test acc)
│   ├── scheme3_results.json              # Metrik evaluasi Skema 3 (87.89% test acc)
│   └── inference_results.csv             # Hasil batch inference pada 12 sampel kalimat baru
├── Scraping.ipynb                        # Pengambilan ulasan Google Play (Tokopedia, Shopee, Bukalapak)
├── Preprocessing.ipynb                   # Pelabelan, pembersihan Sastrawi, balancing & EDA WordCloud
├── Skenario_1_LSTM_TFIDF.ipynb           # Pelatihan & evaluasi Skema 1
├── Skenario_2_BiLSTM_Word2Vec.ipynb      # Pelatihan & evaluasi Skema 2 (Model terbaik)
├── Skenario_3_CNNLSTM_Embedding.ipynb    # Pelatihan & evaluasi Skema 3
├── Inference.ipynb                       # Demo inferensi single-text, batch & perbandingan 3 skema
├── requirements.txt                      # Daftar dependensi Python
└── README.md
```

> Catatan: Direktori `data/` dan `models/` dibuat secara dinamis saat notebook dijalankan (tidak disertakan dalam repositori karena ukuran file dataset dan bobot `.keras`).

---

## Teknologi & Pustaka

- **Deep Learning & ML:** `tensorflow>=2.12.0`, `scikit-learn>=1.2.0`
- **NLP & Scraping:** `google-play-scraper==1.2.7`, `nltk>=3.8.1`, `PySastrawi>=1.0.1`, `gensim>=4.3.0`
- **Data & Visualisasi:** `pandas>=1.5.0`, `numpy>=1.23.0`, `matplotlib>=3.7.0`, `seaborn>=0.12.0`, `wordcloud>=1.9.0`, `tqdm>=4.65.0`

---

## Cara Menjalankan Proyek

1. **Clone repositori & instal dependensi:**
   ```bash
   git clone https://github.com/Bimzt/Indonesian-Review-Sentiment-Analysis.git
   cd Indonesian-Review-Sentiment-Analysis
   pip install -r requirements.txt
   ```
2. **Jalankan notebook secara berurutan:**
   - `Scraping.ipynb` → menghasilkan `data/raw_reviews.csv`
   - `Preprocessing.ipynb` → menghasilkan dataset bersih hasil balancing *(catatan: sel simpan menulis ke `data/processed_data3.csv`, sedangkan notebook skenario membaca `data/processed_data.csv` — samakan nama file tersebut sebelum menjalankan pelatihan)*
   - `Skenario_1_LSTM_TFIDF.ipynb`, `Skenario_2_BiLSTM_Word2Vec.ipynb`, `Skenario_3_CNNLSTM_Embedding.ipynb` → melatih model, menyimpan `.keras` & tokenizer ke `models/scheme*/`, dan menghasilkan `reports/scheme*_results.json`
   - `Inference.ipynb` → memuat ketiga model tersimpan untuk pengujian interaktif dan menyimpan `reports/inference_results.csv`
