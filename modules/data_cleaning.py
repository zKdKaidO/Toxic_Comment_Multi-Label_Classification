# File: modules/data_cleaning.py
import re
import html
import unidecode
import contractions
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Tải data cho nltk (nếu máy chưa có thì nó tự tải)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

english_stop_words = stopwords.words('english')
if 'not' in english_stop_words:
    english_stop_words.remove('not')
stemmer = PorterStemmer()

def normalize_text(text):
    """Hàm này dùng cho các mô hình truyền thống (BoW, TF-IDF)"""
    text = text.lower()
    text = unidecode.unidecode(text)
    text = contractions.fix(text)
    text = html.unescape(text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = text.translate(str.maketrans('','',string.punctuation))
    tokens = text.split()
    tokens = [w for w in tokens if w not in english_stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    text = ' '.join(tokens)
    return re.sub(r'\s+', ' ', text).strip()

def clean_text(text):
    """Hàm này dùng cho Deep Learning (Transformer). Đã bỏ .lower() để giữ viết hoa"""
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = unidecode.unidecode(text)
    text = contractions.fix(text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!\'-]', '', text) # Cho phép cả chữ hoa
    return re.sub(r'\s+', ' ', text).strip()

def clean_text_dl(text):
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = unidecode.unidecode(text)
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,!?\'-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:2000]