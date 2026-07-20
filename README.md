---
title: Demo ABSA Tokopedia
emoji: 🛒
colorFrom: teal
colorTo: orange
sdk: streamlit
sdk_version: 1.36.0
app_file: app.py
pinned: false
---

# Demo ABSA Tokopedia

Aplikasi Streamlit untuk demo **Aspect-Based Sentiment Analysis (ABSA)** pada ulasan produk Tokopedia.

Pipeline yang digunakan:

1. **Ekstraksi Aspek**: IndoBERT-BiLSTM-CRF  
2. **Klasifikasi Sentimen Aspek**: mDeBERTa-v3  
3. **Fallback/Koreksi Lexicon**: digunakan untuk konteks lokal sentimen dan agar demo tetap berjalan saat model belum diunggah.

## Struktur folder yang disarankan

Upload file ini ke Hugging Face Space:

```text
.
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── indobert_bilstm_crf_aspect/
│   ├── model.pt
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── vocab.txt / tokenizer files
│   └── ...
└── deberta_v3_aspect_sentiment/
    ├── config.json
    ├── model.safetensors atau pytorch_model.bin
    ├── tokenizer_config.json
    ├── tokenizer.json / spm.model
    └── ...
```

## Cara export model dari notebook

Pastikan cell terakhir di notebook dijalankan:

```python
# Simpan model ekstraksi aspek
os.makedirs("./indobert_bilstm_crf_aspect", exist_ok=True)
torch.save({
    "model_state_dict": aspect_model.state_dict(),
    "label2id": label2id,
    "id2label": id2label,
    "model_name": INDOBERT_MODEL,
}, "./indobert_bilstm_crf_aspect/model.pt")
aspect_tokenizer.save_pretrained("./indobert_bilstm_crf_aspect")

# Simpan model klasifikasi sentimen aspek
trainer.save_model("./deberta_v3_aspect_sentiment")
sentiment_tokenizer.save_pretrained("./deberta_v3_aspect_sentiment")
```

Setelah itu, download dua folder model tersebut dan upload ke Hugging Face Space bersama `app.py` dan `requirements.txt`.

## Alternatif: model disimpan di repo Hugging Face terpisah

Di sidebar aplikasi, isi:

- `ASPECT_MODEL_DIR`: `username/indobert_bilstm_crf_aspect`
- `SENTIMENT_MODEL_DIR`: `username/deberta_v3_aspect_sentiment`

Atau set sebagai **Space Variables** dengan nama yang sama.

## Catatan untuk demo sidang

Jika model belum ter-upload, aplikasi tetap berjalan dengan fallback lexicon. Namun untuk hasil yang sesuai eksperimen Tugas Akhir, gunakan dua folder model hasil training dari notebook.
