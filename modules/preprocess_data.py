import re
import html
import string
import gc
import os
import pickle
import sys
import unidecode
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

sys.setrecursionlimit(5000)

# Tự động tải dữ liệu NLTK nếu máy chưa có
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Cấu hình stopwords và stemmer
english_stop_words = stopwords.words('english')
if 'not' in english_stop_words:
    english_stop_words.remove('not')
stemmer = PorterStemmer()


def normalize_text(text):
    """Tiền xử lý cho Traditional Pipeline (TF-IDF)"""
    text = text.lower()
    text = unidecode.unidecode(text)
    text = contractions.fix(text)
    text = html.unescape(text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = text.split()
    tokens = [w for w in tokens if w not in english_stop_words]
    tokens = [stemmer.stem(w) for w in tokens]

    text = ' '.join(tokens)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_text_dl(text: str, max_len: int = None) -> str:
    """Tiền xử lý cho Deep Learning Pipeline (DistilRoBERTa)"""
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = unidecode.unidecode(text)
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,!?\'-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_len] if max_len is not None else text


# ── Script entry point ────────────────────────────────────────────────────────
# Run this file directly to preprocess the full dataset and save to disk:
#
#   python modules/preprocess_data.py
#
# Called automatically by draft.ipynb Cell 5 via subprocess.
# Supports checkpointing: safe to re-run if interrupted.

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    # Paths are relative to the project root (cwd when called from notebook)
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR     = os.path.join(_project_root, "data")
    FEATURES_DIR = os.path.join(_project_root, "features")
    OUT_FILE     = os.path.join(FEATURES_DIR, "normalized_texts.pkl")
    CKPT_DIR     = os.path.join(FEATURES_DIR, "preprocess_ckpt")

    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR,     exist_ok=True)

    # Early exit if already done
    if os.path.exists(OUT_FILE):
        print(f"Output already exists at '{OUT_FILE}'. Nothing to do.")
        sys.exit(0)

    BATCH_SIZE = 5_000  # smaller batch → less peak memory per chunk

    # Load raw data
    print("Loading raw data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    X_train = train_df["comment_text"].astype(str).tolist()
    X_test  = test_df["comment_text"].astype(str).tolist()
    print(f"  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    def _process_split(texts, split_name, preprocess_fn):
        """Process one split in checkpointed batches to avoid OOM."""
        n_batches = (len(texts) - 1) // BATCH_SIZE + 1
        results = []
        for batch_idx in range(n_batches):
            ckpt_file = os.path.join(CKPT_DIR, f"{split_name}_batch_{batch_idx}.pkl")
            if os.path.exists(ckpt_file):
                with open(ckpt_file, "rb") as f:
                    batch_result = pickle.load(f)
                print(f"  [{split_name}] Batch {batch_idx + 1}/{n_batches} — resumed from checkpoint.")
            else:
                start = batch_idx * BATCH_SIZE
                batch = texts[start : start + BATCH_SIZE]
                batch_result = []
                for text in tqdm(batch, desc=f"[{split_name}] Batch {batch_idx + 1}/{n_batches}"):
                    try:
                        batch_result.append(preprocess_fn(text))
                    except Exception:
                        batch_result.append("")
                with open(ckpt_file, "wb") as f:
                    pickle.dump(batch_result, f)
                del batch
                gc.collect()
            results.extend(batch_result)
            del batch_result
            gc.collect()
        return results

    print("\nProcessing Train split (Traditional Pipeline)...")
    X_normalize = _process_split(X_train, "train", normalize_text)

    print("\nProcessing Test split (Traditional Pipeline)...")
    X_test_normalize = _process_split(X_test, "test", normalize_text)

    print(f"\nSaving to {OUT_FILE} ...")
    with open(OUT_FILE, "wb") as f:
        pickle.dump({"X_normalize": X_normalize, "X_test_normalize": X_test_normalize}, f)

    print("All done! You can now run draft.ipynb — it will load from disk.")