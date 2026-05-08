import re
import html
import string
import unidecode
import contractions
import sys
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
    text = str(text)
    text = text.lower()
    text = unidecode.unidecode(text)
    text = contractions.fix(text)
    text = html.unescape(text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', text)
    text = text.translate(str.maketrans('','',string.punctuation))
    
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