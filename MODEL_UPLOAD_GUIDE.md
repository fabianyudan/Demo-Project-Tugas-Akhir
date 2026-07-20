# Panduan Upload Model ke Hugging Face Space

## 1. Jalankan cell penyimpanan model di notebook

Notebook kamu sudah memiliki cell akhir untuk menyimpan model:

- `indobert_bilstm_crf_aspect/model.pt`
- `deberta_v3_aspect_sentiment/`

Jalankan cell tersebut setelah training selesai.

## 2. Download folder model

Dari Colab/Kaggle/Jupyter, download dua folder berikut:

```text
indobert_bilstm_crf_aspect
/deberta_v3_aspect_sentiment
```

## 3. Buat Space baru

Di Hugging Face:

1. Klik **New Space**
2. Pilih SDK: **Streamlit**
3. Upload isi folder project ini
4. Upload dua folder model ke root Space

## 4. Pastikan struktur akhirnya seperti ini

```text
app.py
requirements.txt
README.md
indobert_bilstm_crf_aspect/model.pt
deberta_v3_aspect_sentiment/config.json
```

Catatan: nama folder sentimen yang benar di app adalah:

```text
deberta_v3_aspect_sentiment
```

## 5. Kalau model terlalu besar

Gunakan Git LFS atau upload model ke repo Hugging Face Model terpisah, lalu isi nama repo di sidebar aplikasi atau Space Variables:

```text
ASPECT_MODEL_DIR=username/indobert_bilstm_crf_aspect
SENTIMENT_MODEL_DIR=username/deberta_v3_aspect_sentiment
```
