# URL Sentinel — Deteksi Phishing URL (Random Forest + XAI/SHAP)

Aplikasi web **lokal** berbasis **Streamlit** untuk memeriksa apakah sebuah URL
tergolong **phishing** atau **legitimate**, menggunakan model **Random Forest**
yang dilatih pada notebook `Deteksi_Phishing_RF_XAI_Bab3.ipynb`, dilengkapi
penjelasan **Explainable AI (SHAP)** atas setiap prediksi.

## Struktur Proyek

```
streamlit_app/
├── app.py                     # Aplikasi Streamlit utama
├── features.py                # Ekstraksi 14 fitur URL (Tabel 6, Bab III)
├── train_model.py             # Skrip pelatihan ulang model (opsional)
├── requirements.txt
├── dataset_kaggle.csv         # Dataset sumber (untuk train_model.py)
├── dataset_phishtank.csv      # Dataset sumber (untuk train_model.py)
└── model/
    ├── random_forest_phishing_model.joblib   # Model terlatih (siap pakai)
    └── feature_columns.joblib                # Urutan 14 fitur
```

Model yang sudah terlatih **sudah disertakan** di folder `model/`, sehingga
kamu bisa langsung menjalankan aplikasi tanpa perlu melatih ulang.

## Cara Menjalankan

1. **Instal dependensi** (disarankan pakai virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

2. **Jalankan aplikasi**:
   ```bash
   streamlit run app.py
   ```

3. Browser akan otomatis terbuka di `http://localhost:8501`.
   Jika tidak, buka tautan tersebut secara manual.

4. Masukkan sebuah URL pada kolom input, lalu klik **"Jalankan Analisis"**
   (atau coba salah satu tombol contoh URL yang tersedia).

## (Opsional) Melatih Ulang Model

Jika ingin melatih ulang model dari dataset (misalnya setelah mengganti
dataset), jalankan:
```bash
python train_model.py
```
Ini akan menghasilkan ulang `model/random_forest_phishing_model.joblib` dan
`model/feature_columns.joblib` mengikuti pipeline yang sama dengan notebook
(gabungan Kaggle + PhishTank → ekstraksi 14 fitur → split 70/15/15 →
Grid Search CV → simpan model terbaik).

## Cara Kerja Penjelasan XAI

Untuk setiap URL yang diperiksa:
1. **Random Forest** menghasilkan label (`Phishing`/`Legitimate`) beserta
   probabilitasnya.
2. **SHAP (TreeExplainer)** menghitung kontribusi tiap satu dari 14 fitur
   terhadap prediksi tersebut.
3. Fitur dengan kontribusi SHAP **positif** mendorong prediksi ke arah
   **Phishing**; kontribusi **negatif** mendorong ke arah **Legitimate**.
4. Aplikasi menampilkan:
   - Ringkasan alasan utama dalam bahasa natural (mis. *"URL tidak
     menggunakan protokol HTTPS yang aman"*).
   - Grafik batang divergen (merah = mendukung phishing, hijau = mendukung
     legitimate) untuk seluruh 14 fitur.
   - Tabel lengkap nilai fitur & kontribusi SHAP-nya.

## Catatan

- Seluruh proses (ekstraksi fitur, prediksi, SHAP) berjalan **sepenuhnya di
  komputer lokal** — tidak ada URL yang dikirim ke server pihak ketiga.
- Fitur `RandomString` menggunakan heuristik sederhana (rasio huruf vokal &
  rangkaian konsonan) sebagai indikasi domain hasil generate otomatis (DGA),
  bukan model klasifikasi string acak yang canggih.
- **Fitur `NoHttps` ditangani secara khusus.** Dataset Kaggle yang dipakai
  hampir selalu menuliskan URL tanpa skema (`http://`/`https://`), sedangkan
  dataset PhishTank selalu menuliskan skema secara eksplisit. Jika skema yang
  hilang begitu saja diasumsikan "http" (tidak aman), model akan salah
  belajar membedakan **asal sumber data**, bukan **keamanan URL** — situs sah
  seperti wikipedia.org yang eksplisit HTTPS bisa keliru diklasifikasikan
  sebagai phishing. Untuk menghindarinya, saat skema tidak dituliskan secara
  eksplisit, aplikasi mengasumsikan situs tersebut menggunakan HTTPS (sesuai
  kondisi web modern), bukan mengasumsikan tidak aman. Lihat `infer_no_https()`
  pada `features.py` untuk detail implementasinya.
- **Keterbatasan umum:** model hanya memakai 14 fitur leksikal/struktural
  (Tabel 6) tanpa reputasi domain, umur domain (WHOIS), atau analisis konten
  halaman, sehingga untuk URL dengan probabilitas mendekati 50% hasil
  sebaiknya tidak dijadikan satu-satunya acuan keputusan. Ini konsisten
  dengan saran pengembangan lanjutan pada notebook penelitian.
