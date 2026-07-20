import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover
    snapshot_download = None

try:
    from torchcrf import CRF
except Exception:  # pragma: no cover
    try:
        from TorchCRF import CRF
    except Exception:
        CRF = None

# ============================================================
# Konfigurasi dasar
# ============================================================
st.set_page_config(
    page_title="Demo ABSA Tokopedia",
    page_icon="🛒",
    layout="wide",
)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH_AE = int(os.getenv("MAX_LENGTH_AE", "64"))
MAX_LENGTH_ASC = int(os.getenv("MAX_LENGTH_ASC", "160"))

DEFAULT_ASPECT_DIR = os.getenv("ASPECT_MODEL_DIR", "indobert_bilstm_crf_aspect")
DEFAULT_SENTIMENT_DIR = os.getenv("SENTIMENT_MODEL_DIR", "deberta_v3_aspect_sentiment")

ASPECT_DISPLAY = {
    "quality": "Kualitas Produk",
    "price": "Harga",
    "delivery": "Pengiriman",
    "service": "Pelayanan Toko",
    "packaging": "Kemasan/Packing",
    "general": "Umum",
}

SENTIMENT_DISPLAY = {
    "positive": "Positif",
    "neutral": "Netral",
    "negative": "Negatif",
}

SENTIMENT_EMOJI = {
    "positive": "🟢",
    "neutral": "🟡",
    "negative": "🔴",
}

# ============================================================
# Lexicon yang sama dengan notebook
# ============================================================
ASPECT_LEXICON = {
    "quality": [
        "kualitas", "mutu", "produk", "barang", "bahan", "fungsi", "ukuran",
        "warna", "kondisi", "deskripsi", "original", "ori", "rusak", "cacat",
        "palsu", "awet", "sesuai", "bagus", "jelek",
    ],
    "price": [
        "harga", "harganya", "biaya", "ongkir", "diskon", "promo",
        "murah", "mahal", "kemahalan", "terjangkau", "worth", "overprice",
    ],
    "delivery": [
        "pengiriman", "kirim", "dikirim", "kurir", "paket", "resi",
        "sampai", "datang", "cepat", "lama", "telat", "lambat", "terlambat",
    ],
    "service": [
        "seller", "penjual", "toko", "admin", "pelayanan", "layanan",
        "respon", "respons", "chat", "ramah", "responsif", "slow respon", "fast respon",
    ],
    "packaging": [
        "packing", "packaging", "kemasan", "bungkus", "bubble", "bubblewrap",
        "dus", "aman", "rapi", "penyok", "sobek", "hancur",
    ],
}

ASPECT_PRIORITY = ["delivery", "quality", "price", "service", "packaging"]
ASPECT_ORDER = ["delivery", "quality", "price", "service", "packaging", "general"]

POSITIVE_LEXICON = set("""
bagus baik mantap original ori sesuai awet berfungsi normal murah terjangkau worth hemat promo diskon cepat aman tepat satset kilat ramah responsif rapi tebal kuat puas recommended rekomendasi suka oke ok
""".split())

NEGATIVE_LEXICON = set("""
jelek buruk rusak cacat palsu mengecewakan pecah mahal kemahalan overprice lama telat lambat terlambat gagal buruk jutek slow penyok sobek hancur berantakan kurang
""".split())

NEGATION_WORDS = {"tidak", "nggak", "ngga", "ga", "gak", "bukan", "belum", "kurang"}

ASPECT_CANONICAL_TERMS = {
    "quality": ["barang", "produk", "kualitas", "mutu", "bahan", "fungsi", "ukuran", "warna", "kondisi", "deskripsi", "original", "ori"],
    "price": ["harga", "biaya", "ongkir", "diskon", "promo"],
    "delivery": ["pengiriman", "kirim", "dikirim", "kurir", "paket", "resi", "sampai", "datang"],
    "service": ["seller", "penjual", "toko", "admin", "pelayanan", "layanan", "respon", "respons", "chat"],
    "packaging": ["packing", "packaging", "kemasan", "bungkus", "bubble", "bubblewrap", "dus"],
}

ASPECT_IMPLICIT_TERMS = {
    "quality": ["bagus", "jelek", "rusak", "cacat", "palsu", "awet", "sesuai", "original", "ori"],
    "price": ["murah", "mahal", "kemahalan", "worth", "terjangkau", "overprice"],
    "delivery": ["cepat", "lama", "telat", "lambat", "terlambat"],
    "service": ["ramah", "responsif", "slow", "fast", "jutek"],
    "packaging": ["rapi", "aman", "penyok", "sobek", "hancur", "tebal"],
}

LABEL_LIST = ["O", "B-ASP", "I-ASP"]
DEFAULT_LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
DEFAULT_ID2LABEL = {i: label for label, i in DEFAULT_LABEL2ID.items()}
DEFAULT_SENTIMENT_LABELS = ["negative", "neutral", "positive"]
DEFAULT_ID2SENTIMENT = {i: label for i, label in enumerate(DEFAULT_SENTIMENT_LABELS)}


# ============================================================
# Utility teks dan lexicon
# ============================================================
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", str(text).lower(), flags=re.UNICODE)


def normalize_token(token: str) -> str:
    token = str(token).lower().strip()
    token = re.sub(r"[^a-zA-Z0-9]", "", token)

    if token == "harganya":
        return "harga"

    for suffix in ["nya", "ku", "mu"]:
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            token = token[:-len(suffix)]
            break
    return token


def normalized_tokens(tokens: List[str]) -> List[str]:
    return [normalize_token(tok) for tok in tokens]


def make_keyword_phrases(aspect_lexicon: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    keyword_phrases = []
    for aspect, keywords in aspect_lexicon.items():
        for kw in keywords:
            kw_tokens = simple_tokenize(kw)
            kw_tokens_norm = [normalize_token(t) for t in kw_tokens if normalize_token(t)]
            if kw_tokens_norm:
                keyword_phrases.append({
                    "aspect": aspect,
                    "keyword": kw,
                    "tokens": kw_tokens_norm,
                    "length": len(kw_tokens_norm),
                    "priority": ASPECT_PRIORITY.index(aspect) if aspect in ASPECT_PRIORITY else 999,
                })
    return keyword_phrases


def find_one_aspect_match(tokens: List[str], aspect_lexicon: Dict[str, List[str]]) -> Optional[Dict[str, Any]]:
    keyword_phrases = make_keyword_phrases(aspect_lexicon)
    toks_norm = normalized_tokens(tokens)
    matches = []

    for i in range(len(toks_norm)):
        for item in keyword_phrases:
            kw_tokens = item["tokens"]
            n = item["length"]
            if i + n <= len(toks_norm) and toks_norm[i:i + n] == kw_tokens:
                matches.append({
                    "aspect_category": item["aspect"],
                    "aspect_text": " ".join(tokens[i:i + n]),
                    "start": i,
                    "end": i + n,
                    "keyword": item["keyword"],
                    "length": n,
                    "priority": item["priority"],
                })

    if not matches:
        return None

    matches = sorted(matches, key=lambda x: (x["start"], -x["length"], x["priority"]))
    return matches[0]


def score_lexicon_sentiment(tokens: List[str], selected_aspect: Optional[Dict[str, Any]], window: int = 4) -> str:
    toks_norm = [t for t in normalized_tokens(tokens) if t]
    if not toks_norm:
        return "neutral"

    if selected_aspect is None:
        context = toks_norm
    else:
        start = selected_aspect["start"]
        end = selected_aspect["end"]
        left = max(0, start - window)
        right = min(len(toks_norm), end + window)
        context = toks_norm[left:right]

    pos_score = 0
    neg_score = 0
    for i, tok in enumerate(context):
        has_negation = any(w in NEGATION_WORDS for w in context[max(0, i - 2):i])

        if tok in POSITIVE_LEXICON:
            neg_score += 1 if has_negation else 0
            pos_score += 0 if has_negation else 1

        if tok in NEGATIVE_LEXICON:
            pos_score += 1 if has_negation else 0
            neg_score += 0 if has_negation else 1

    if pos_score > neg_score:
        return "positive"
    if neg_score > pos_score:
        return "negative"
    return "neutral"


def tokenize_words(text: str) -> List[str]:
    raw_tokens = re.findall(r"\b\w+\b", str(text).lower(), flags=re.UNICODE)
    return [normalize_token(tok) for tok in raw_tokens if normalize_token(tok)]


def merge_aspect_spans(tokens: List[str], labels: List[str]) -> List[str]:
    spans = []
    current = []
    for tok, lab in zip(tokens, labels):
        if lab == "B-ASP":
            if current:
                spans.append(" ".join(current))
            current = [tok]
        elif lab == "I-ASP" and current:
            current.append(tok)
        else:
            if current:
                spans.append(" ".join(current))
                current = []
    if current:
        spans.append(" ".join(current))
    return spans


def map_raw_aspect_to_category(review_text: str, raw_aspects: Optional[List[str]] = None) -> str:
    if raw_aspects:
        for raw_asp in raw_aspects[:1]:
            raw_tokens = tokenize_words(raw_asp)
            raw_joined = " ".join(raw_tokens)
            for category in ASPECT_ORDER:
                if category == "general":
                    continue
                terms = ASPECT_CANONICAL_TERMS.get(category, []) + ASPECT_IMPLICIT_TERMS.get(category, [])
                terms_norm = [normalize_token(t) for t in terms]
                if any(t in raw_tokens for t in terms_norm) or raw_joined in terms_norm:
                    return category

    selected = find_one_aspect_match(simple_tokenize(review_text), ASPECT_LEXICON)
    if selected is not None:
        return selected["aspect_category"]
    return "general"


def local_context_sentiment(review_text: str, aspect_category: str, window: int = 4) -> str:
    tokens = simple_tokenize(review_text)
    selected = None
    if aspect_category != "general":
        selected = find_one_aspect_match(tokens, {aspect_category: ASPECT_LEXICON.get(aspect_category, [])})
    return score_lexicon_sentiment(tokens, selected, window=window)


# ============================================================
# Model custom IndoBERT-BiLSTM-CRF
# ============================================================
class IndoBERTBiLSTMCRF(nn.Module):
    def __init__(self, model_name: str, num_labels: int, lstm_hidden: int = 64, lstm_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        if CRF is None:
            raise ImportError("Library torchcrf/TorchCRF belum terpasang. Cek requirements.txt.")

        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(lstm_hidden * 2, num_labels)

        try:
            self.crf = CRF(num_labels, batch_first=True)
            self._crf_mode = "torchcrf"
        except TypeError:
            self.crf = CRF(num_labels)
            self._crf_mode = "TorchCRF"

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        lstm_output, _ = self.bilstm(sequence_output)
        emissions = self.classifier(self.dropout(lstm_output))
        mask = attention_mask.bool()

        if labels is not None:
            if hasattr(self.crf, "forward"):
                loss = -self.crf(emissions, labels, mask=mask, reduction="mean")
            else:
                loss = -self.crf(emissions, labels, mask)
            return loss, emissions

        if hasattr(self.crf, "decode"):
            return self.crf.decode(emissions, mask=mask)
        if hasattr(self.crf, "viterbi_decode"):
            return self.crf.viterbi_decode(emissions, mask)
        raise RuntimeError("CRF tidak memiliki method decode/viterbi_decode.")


def _safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def resolve_path_or_repo(path_or_repo: str, required_file: Optional[str] = None) -> Path:
    """Mendukung folder lokal di Space atau model repo Hugging Face via env var."""
    local = Path(path_or_repo)
    if local.exists():
        return local

    looks_like_repo = "/" in path_or_repo and not path_or_repo.startswith("/") and snapshot_download is not None
    if looks_like_repo:
        downloaded = snapshot_download(repo_id=path_or_repo)
        return Path(downloaded)

    return local


@st.cache_resource(show_spinner="Memuat model ekstraksi aspek...")
def load_aspect_resources(path_or_repo: str):
    model_dir = resolve_path_or_repo(path_or_repo, required_file="model.pt")
    ckpt_path = model_dir / "model.pt"
    if not ckpt_path.exists():
        return {"available": False, "error": f"File {ckpt_path} belum ditemukan."}

    try:
        checkpoint = _safe_torch_load(ckpt_path)
        label2id = checkpoint.get("label2id", DEFAULT_LABEL2ID)
        id2label = checkpoint.get("id2label", DEFAULT_ID2LABEL)
        id2label = {int(k): v for k, v in id2label.items()}
        model_name = checkpoint.get("model_name", "indobenchmark/indobert-base-p1")

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = IndoBERTBiLSTMCRF(
            model_name=model_name,
            num_labels=len(label2id),
            lstm_hidden=64,
            dropout=0.3,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(DEVICE)
        model.eval()

        return {
            "available": True,
            "model": model,
            "tokenizer": tokenizer,
            "label2id": label2id,
            "id2label": id2label,
            "model_dir": str(model_dir),
            "error": None,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


@st.cache_resource(show_spinner="Memuat model klasifikasi sentimen...")
def load_sentiment_resources(path_or_repo: str):
    model_dir = resolve_path_or_repo(path_or_repo)
    if not model_dir.exists():
        return {"available": False, "error": f"Folder {model_dir} belum ditemukan."}

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(DEVICE)
        model.eval()

        raw_id2label = getattr(model.config, "id2label", None) or DEFAULT_ID2SENTIMENT
        id2sentiment = {int(k): str(v).lower() for k, v in raw_id2label.items()}

        # Jika config masih LABEL_0/LABEL_1/LABEL_2, pakai mapping dari notebook.
        if all(v.startswith("label_") or v.startswith("LABEL_") for v in id2sentiment.values()):
            id2sentiment = DEFAULT_ID2SENTIMENT

        return {
            "available": True,
            "model": model,
            "tokenizer": tokenizer,
            "id2sentiment": id2sentiment,
            "model_dir": str(model_dir),
            "error": None,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def predict_aspects_model(text: str, aspect_bundle: Dict[str, Any], top_k: int = 1) -> Tuple[List[str], List[Tuple[str, str]]]:
    tokens = simple_tokenize(text)
    if not aspect_bundle.get("available"):
        selected = find_one_aspect_match(tokens, ASPECT_LEXICON)
        labels = ["O"] * len(tokens)
        if selected:
            labels[selected["start"]] = "B-ASP"
            for pos in range(selected["start"] + 1, selected["end"]):
                labels[pos] = "I-ASP"
            return [selected["aspect_text"]][:top_k], list(zip(tokens, labels))
        return [], list(zip(tokens, labels))

    model = aspect_bundle["model"]
    tokenizer = aspect_bundle["tokenizer"]
    id2label = aspect_bundle["id2label"]

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH_AE,
        return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0)
    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

    with torch.no_grad():
        pred_ids = model(**encoding)[0]

    word_label_map = {}
    for token_pos, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx not in word_label_map and token_pos < len(pred_ids):
            word_label_map[word_idx] = id2label.get(int(pred_ids[token_pos]), "O")

    word_labels = [word_label_map.get(i, "O") for i in range(len(tokens))]
    aspects = merge_aspect_spans(tokens, word_labels)
    return aspects[:top_k], list(zip(tokens, word_labels))


def predict_sentiment_model(review_text: str, aspect: str, sentiment_bundle: Dict[str, Any]) -> Tuple[str, float]:
    if not sentiment_bundle.get("available"):
        selected = None
        if aspect != "general":
            selected = find_one_aspect_match(simple_tokenize(review_text), {aspect: ASPECT_LEXICON.get(aspect, [])})
        sentiment = score_lexicon_sentiment(simple_tokenize(review_text), selected)
        return sentiment, 1.0 if sentiment != "neutral" else 0.65

    model = sentiment_bundle["model"]
    tokenizer = sentiment_bundle["tokenizer"]
    id2sentiment = sentiment_bundle["id2sentiment"]

    input_text = f"aspek: {aspect} ulasan: {review_text}"
    enc = tokenizer(
        [input_text],
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH_ASC,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        pred = int(np.argmax(probs))

    return id2sentiment.get(pred, DEFAULT_ID2SENTIMENT.get(pred, "neutral")), float(probs[pred])


def predict_absa(review_text: str, aspect_bundle: Dict[str, Any], sentiment_bundle: Dict[str, Any]) -> Dict[str, Any]:
    review_text = clean_text(review_text)
    raw_aspects, token_labels = predict_aspects_model(review_text, aspect_bundle, top_k=1)
    aspect = map_raw_aspect_to_category(review_text, raw_aspects)

    model_sentiment, confidence = predict_sentiment_model(review_text, aspect, sentiment_bundle)
    rule_sentiment = local_context_sentiment(review_text, aspect)

    if rule_sentiment is not None and rule_sentiment != "neutral":
        final_sentiment = rule_sentiment
        note = "dikoreksi oleh lexicon konteks lokal"
    else:
        final_sentiment = model_sentiment
        note = "prediksi model" if sentiment_bundle.get("available") else "fallback lexicon"

    return {
        "review": review_text,
        "aspect": aspect,
        "aspect_label": ASPECT_DISPLAY.get(aspect, aspect),
        "sentiment": final_sentiment,
        "sentiment_label": SENTIMENT_DISPLAY.get(final_sentiment, final_sentiment),
        "emoji": SENTIMENT_EMOJI.get(final_sentiment, "⚪"),
        "confidence": confidence,
        "sentiment_model": model_sentiment,
        "raw_aspects_from_AE": raw_aspects,
        "token_labels": token_labels,
        "note": note,
    }


# ============================================================
# Tampilan Streamlit
# ============================================================
st.title("🛒 Demo ABSA Ulasan Produk Tokopedia")
st.caption("Aspect-Based Sentiment Analysis: IndoBERT-BiLSTM-CRF untuk ekstraksi aspek dan mDeBERTa-v3 untuk klasifikasi sentimen aspek.")

with st.sidebar:
    st.header("Pengaturan Demo")
    aspect_dir = st.text_input("Folder / repo model ekstraksi aspek", value=DEFAULT_ASPECT_DIR)
    sentiment_dir = st.text_input("Folder / repo model sentimen", value=DEFAULT_SENTIMENT_DIR)
    st.caption("Bisa berupa folder lokal di Space, atau repo model Hugging Face seperti username/nama-model.")
    show_token_labels = st.toggle("Tampilkan token BIO", value=True)
    show_batch = st.toggle("Mode batch beberapa review", value=False)

aspect_bundle = load_aspect_resources(aspect_dir)
sentiment_bundle = load_sentiment_resources(sentiment_dir)

status_cols = st.columns(3)
with status_cols[0]:
    st.metric("Device", str(DEVICE).upper())
with status_cols[1]:
    st.metric("Model Ekstraksi Aspek", "Aktif" if aspect_bundle.get("available") else "Fallback")
with status_cols[2]:
    st.metric("Model Sentimen", "Aktif" if sentiment_bundle.get("available") else "Fallback")

if not aspect_bundle.get("available") or not sentiment_bundle.get("available"):
    with st.expander("Catatan model belum lengkap", expanded=False):
        if not aspect_bundle.get("available"):
            st.warning(f"Model ekstraksi aspek belum aktif: {aspect_bundle.get('error')}")
        if not sentiment_bundle.get("available"):
            st.warning(f"Model sentimen belum aktif: {sentiment_bundle.get('error')}")
        st.write("Aplikasi tetap bisa dipakai untuk demo awal karena memakai fallback lexicon, tetapi hasil final sidang sebaiknya memakai folder model hasil training dari notebook.")

examples = [
    "Barangnya bagus dan original, harga murah, tapi pengiriman sangat lama.",
    "Packing rapi dan aman, tetapi barangnya rusak saat sampai.",
    "Seller ramah dan respon cepat, tapi harga agak mahal.",
    "Pengiriman cepat, barang sesuai deskripsi, saya puas belanja di toko ini.",
]

if not show_batch:
    selected_example = st.selectbox("Contoh ulasan", options=["Tulis sendiri"] + examples)
    default_text = examples[0] if selected_example == "Tulis sendiri" else selected_example
    review_text = st.text_area("Masukkan ulasan produk", value=default_text, height=140)

    if st.button("Analisis Sentimen Aspek", type="primary", use_container_width=True):
        if not review_text.strip():
            st.error("Masukkan ulasan terlebih dahulu.")
        else:
            result = predict_absa(review_text, aspect_bundle, sentiment_bundle)

            st.subheader("Hasil Prediksi")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Aspek Utama", result["aspect_label"])
            with c2:
                st.metric("Sentimen", f"{result['emoji']} {result['sentiment_label']}")
            with c3:
                st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")

            st.info(f"Catatan: {result['note']}.")

            result_df = pd.DataFrame([{
                "review": result["review"],
                "aspect": result["aspect"],
                "sentiment": result["sentiment"],
                "confidence": round(result["confidence"], 4),
                "sentiment_model": result["sentiment_model"],
                "raw_aspects_from_AE": ", ".join(result["raw_aspects_from_AE"]) if result["raw_aspects_from_AE"] else "-",
                "note": result["note"],
            }])
            st.dataframe(result_df, use_container_width=True)

            if show_token_labels:
                token_df = pd.DataFrame(result["token_labels"], columns=["token", "BIO_label"])
                st.subheader("Token Label Ekstraksi Aspek")
                st.dataframe(token_df, use_container_width=True)
else:
    default_batch = "\n".join(examples)
    batch_text = st.text_area("Masukkan beberapa ulasan, satu ulasan per baris", value=default_batch, height=220)

    if st.button("Analisis Batch", type="primary", use_container_width=True):
        rows = [line.strip() for line in batch_text.splitlines() if line.strip()]
        if not rows:
            st.error("Masukkan minimal satu ulasan.")
        else:
            with st.spinner("Menganalisis batch..."):
                results = [predict_absa(row, aspect_bundle, sentiment_bundle) for row in rows]
            batch_df = pd.DataFrame([{
                "review": r["review"],
                "aspect": r["aspect"],
                "aspect_label": r["aspect_label"],
                "sentiment": r["sentiment"],
                "sentiment_label": r["sentiment_label"],
                "confidence": round(r["confidence"], 4),
                "note": r["note"],
            } for r in results])
            st.subheader("Hasil Prediksi Batch")
            st.dataframe(batch_df, use_container_width=True)
            csv = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download hasil CSV",
                data=csv,
                file_name="hasil_demo_absa_tokopedia.csv",
                mime="text/csv",
                use_container_width=True,
            )

st.divider()
st.markdown(
    """
**Alur sistem:** ulasan dibersihkan → aspek diekstraksi sebagai token BIO → aspek dipetakan ke kategori utama → teks `aspek: ... ulasan: ...` diklasifikasikan ke sentimen negative, neutral, atau positive.

**Kategori aspek:** quality, price, delivery, service, packaging, dan general.
"""
)
