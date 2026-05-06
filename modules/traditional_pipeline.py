import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import joblib
from pathlib import Path

def train_traditional_model(
    train_df: pd.DataFrame, 
    text_column: str, 
    label_columns: list[str],
    model_save_path: str
):
    """
    Huấn luyện mô hình TF-IDF + Logistic Regression và lưu lại.
    """
    print("Bắt đầu huấn luyện pipeline truyền thống...")
    
    # 1. Vectorization
    print("Tạo TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = vectorizer.fit_transform(train_df[text_column])
    y_train = train_df[label_columns].values
    
    # 2. Model Training
    print("Huấn luyện OneVsRestClassifier với LogisticRegression...")
    logreg = LogisticRegression(solver='liblinear', random_state=42)
    classifier = OneVsRestClassifier(logreg)
    classifier.fit(X_train, y_train)
    
    # 3. Save model and vectorizer
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'vectorizer': vectorizer, 'classifier': classifier}, model_save_path)
    print(f"Đã lưu model và vectorizer tại: {model_save_path}")
    
    return classifier, vectorizer