# NLP Analisis Sentimen

Proyek ini melakukan analisis sentimen pada data teks menggunakan BERT (Bidirectional Encoder Representations from Transformers). 
Proyek ini mengkategorikan teks ke dalam tiga kelas: Positif, Negatif, dan Netral. Model dilatih dan dievaluasi menggunakan dataset yang berisi label sentimen.

## Prasyarat

Untuk menjalankan proyek ini, perlu mengatur CUDA, cuDNN, dan PyTorch, serta menginstal pustaka Python yang diperlukan.

### Langkah 1: Instal CUDA dan cuDNN

1. **Instal CUDA**   : Pastikan menginstal versi CUDA yang sesuai.
                       Ikuti panduan resmi untuk menginstal [CUDA](https://developer.nvidia.com/cuda-downloads) berdasarkan konfigurasi sistem yang tersedia.
   
3. **Instal cuDNN**  : Unduh dan instal [cuDNN](https://developer.nvidia.com/cudnn) yang kompatibel dengan versi CUDA yang siap digunakan.

### Langkah 2: Instal Python dan Pustaka

1. **Versi Python**  : Proyek ini membutuhkan Python versi 3.7 atau lebih baru. Dapat diunduh dari [python.org](https://www.python.org/downloads/).
2. **Pustaka**       : Pustaka yang digunakan mencakup sebagai berikut:
                       -torch
                       -transformers
                       -pandas
                       -scikit-learn
                       -numpy
                       -0matplotlib

### Langkah 3: Jalankan Program

1. **Lokasi PATH**   : Pastikan program yang akan dijalan berada path/alamat yang sudah sesuai dengan lokasi beserta datasetnya
2. **Running**       : Jalankan printah **"Python NLP.py"** pada terminal IDE. Ini akan melatih model analisis sentimen menggunakan dataset yang disediakan dan menampilkan
                       metrik evaluasi seperti Precision, Recall, dan F1-Score untuk setiap kelas. Program ini juga akan menyimpan model dan tokenizer di direktori yang ditentukan.

### Deskripsi Proyek

**Tujuan**           : Tujuan dari proyek ini adalah melatih sebuah model untuk melakukan analisis sentimen pada data teks, yang mengklasifikasikan teks ke dalam tiga kategori: Posit, Netral, Negatif

**File**             :

**NLP.py**: Skrip utama yang memuat data, memprosesnya, melatih model BERT, dan mengevaluasi model.
**sentiment_analysis.csv**: Dataset yang digunakan untuk pelatihan, berisi data teks dan label sentimen yang sesuai.
**precision_recall_f1.png**: Gambar yang menyimpan metrik Precision, Recall, dan F1-Score untuk setiap kelas.

**Model**            : 
BERT (Bidirectional Encoder Representations from Transformers) digunakan untuk melakukan klasifikasi urutan untuk analisis sentimen.

**Metrik Evaluasi**  : 
Kinerja model dievaluasi menggunakan metrik klasifikasi, termasuk Precision, Recall, dan F1-Score.

![precision_recall_f1](https://github.com/user-attachments/assets/6d8006b2-9bd2-4157-a087-e70a26636baa)

