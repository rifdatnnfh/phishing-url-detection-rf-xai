# Pengembangan Model Random Forest Berbasis Explainable AI untuk Deteksi Ancaman Phishing pada Uniform Resource Locator

Paket ini berisi **dua bagian yang saling terhubung** dari satu penelitian yang sama, menggunakan model, fitur, dan data yang identik:

```
PENGEMBANGAN_MODEL_RF_XAI_PHISHING_URL/
├── notebook/                      <- Notebook penelitian (Jupyter/.ipynb)
│   ├── Deteksi_Phishing_RF_XAI.ipynb
│   ├── dataset_kaggle.csv
│   ├── dataset_phishtank.csv
│   ├── random_forest_phishing_model.joblib   (model hasil pelatihan notebook)
│   └── feature_columns.joblib
│
└── website_streamlit/              <- Aplikasi web pengecekan URL (Streamlit)
    ├── app.py
    ├── features.py
    ├── train_model.py
    ├── requirements.txt
    ├── README.md                   (panduan khusus aplikasi web)
    ├── dataset_kaggle.csv
    ├── dataset_phishtank.csv
    └── model/
        ├── random_forest_phishing_model.joblib
        └── feature_columns.joblib
```

## Tujuan Penelitian

Kedua bagian pada paket ini disusun untuk memenuhi tiga tujuan penelitian berikut:

1. **Ekstraksi & Pemodelan** \u2014 Menganalisis fitur-fitur leksikal URL dan membangun model klasifikasi Random Forest yang mampu membedakan URL *phishing* dan *legitimate* dengan akurasi tinggi.
   \u2192 Dipenuhi pada **notebook** (Bagian 1\u20135): ekstraksi 14 fitur URL, Grid Search hyperparameter tuning, evaluasi model (Accuracy, Precision, Recall, F1-score, ROC-AUC).

2. **Interpretabilitas via SHAP** \u2014 Menerapkan metode *Shapley Additive exPlanations* (SHAP) untuk memberikan penjelasan transparan, baik secara **global** (fitur apa yang paling berpengaruh secara umum) maupun **lokal** (kenapa satu URL spesifik diprediksi sebagai phishing), agar hasil prediksi dapat dipertanggungjawabkan dan dipahami oleh pengguna atau analis keamanan.
   \u2192 Dipenuhi pada **notebook** (Bagian 6: SHAP summary/bar/beeswarm plot untuk interpretasi global, waterfall plot untuk interpretasi lokal) **dan** pada **aplikasi web** (setiap URL yang diperiksa langsung ditampilkan alasan "kenapa phishing/legitimate" berbasis SHAP).

3. **Implementasi Sistem** \u2014 Menguji dan mengimplementasikan pipeline model beserta modul penjelas SHAP ke dalam sistem pengujian berbasis *black-box* agar dapat digunakan secara praktis.
   \u2192 Dipenuhi pada **notebook** (Bagian 7\u20138: pipeline `predict_url()` + pengujian black-box: prediksi tunggal/batch, ketahanan input tidak valid, konsistensi model) **dan** diwujudkan secara nyata sebagai **aplikasi web Streamlit** yang siap dipakai.

## Bagian 1 \u2014 Notebook (`notebook/`)

Notebook Jupyter berisi seluruh alur penelitian end-to-end: pengumpulan data, eksplorasi data, ekstraksi 14 fitur URL, studi kasus perbaikan bug kebocoran data (*data leakage*) pada fitur `NoHttps`, pemodelan Random Forest dengan Grid Search, evaluasi model, interpretasi SHAP (global & lokal), implementasi pipeline prediksi, dan pengujian sistem *black-box*.

**Cara menjalankan:**
```bash
cd notebook
pip install pandas numpy scikit-learn matplotlib seaborn shap joblib jupyter
jupyter notebook Deteksi_Phishing_RF_XAI.ipynb
```
Lalu jalankan seluruh sel secara berurutan (Run All).

## Bagian 2 \u2014 Aplikasi Web (`website_streamlit/`)

Aplikasi web lokal berbasis **Streamlit** (bukan Flask/HTML/CSS terpisah) untuk memeriksa sebuah URL secara interaktif. Model yang dipakai **sudah terlatih dan disertakan**, sehingga bisa langsung dijalankan tanpa perlu training ulang.

**Cara menjalankan:**
```bash
cd website_streamlit
pip install -r requirements.txt
streamlit run app.py
```
Buka `http://localhost:8501` di browser. Masukkan URL pada kolom yang tersedia, lalu klik **"Jalankan Analisis"**.

**Output yang ditampilkan untuk setiap URL:**
- Label **Phishing** atau **Legitimate**, beserta probabilitasnya.
- **Alasan utama** kenapa URL tersebut diprediksi demikian, dalam bahasa natural, contoh:
  - *"URL ini terindikasi PHISHING karena: URL tidak menggunakan protokol HTTPS yang aman; hostname URL cukup panjang (39 karakter); URL memiliki beberapa level subdomain, teknik umum untuk menyerupai domain resmi."*
  - *"URL ini terindikasi LEGITIMATE karena: URL sudah menggunakan protokol HTTPS; struktur subdomain URL sederhana; panjang hostname wajar."*
- Grafik kontribusi SHAP untuk seluruh 14 fitur (merah = mendukung phishing, hijau = mendukung legitimate).
- Tabel rincian nilai setiap fitur beserta kontribusi SHAP-nya.

Panduan lebih lengkap (struktur kode, cara melatih ulang model, dsb.) tersedia di `website_streamlit/README.md`.

## Konsistensi Notebook & Aplikasi Web

Modul ekstraksi fitur (`extract_features_14` pada notebook, identik dengan `features.py` pada aplikasi web) dan model Random Forest yang dihasilkan **telah diverifikasi identik** \u2014 URL yang sama akan menghasilkan prediksi dan probabilitas yang sama persis baik dijalankan lewat notebook maupun lewat aplikasi web, karena keduanya menggunakan `random_state` yang sama, dataset yang sama, dan pipeline yang sama.

## Catatan Penting: Perbaikan Kebocoran Data (Data Leakage)

Selama pengembangan, ditemukan bahwa dataset Kaggle yang digunakan hampir selalu menuliskan URL **tanpa skema protokol** (mis. `bbc.com/news`), sedangkan dataset PhishTank selalu menuliskan skema secara eksplisit (`https://...`). Jika tidak ditangani, fitur `NoHttps` akan membocorkan *asal sumber data* alih-alih benar-benar mengukur keamanan URL, menyebabkan situs sah seperti `wikipedia.org` salah diklasifikasikan sebagai phishing. Masalah ini **ditemukan lewat proses interpretasi model** dan telah diperbaiki (lihat Bagian 4 notebook & fungsi `infer_no_https()` pada `features.py`) \u2014 sebuah contoh nyata bagaimana Explainable AI berguna sebagai alat jaminan kualitas model, bukan sekadar penjelas di akhir.

## Keterbatasan

Model hanya menggunakan 14 fitur leksikal/struktural URL, tanpa reputasi domain (WHOIS), umur domain, atau analisis konten halaman. Pengujian pada sejumlah URL populer di luar dataset menunjukkan model masih dapat keliru pada kasus tertentu. Lihat bagian Kesimpulan pada notebook dan `website_streamlit/README.md` untuk saran pengembangan lanjutan.
