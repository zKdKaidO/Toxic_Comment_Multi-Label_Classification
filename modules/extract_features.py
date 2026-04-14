import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

def extract_roberta_features(data_path, output_npy_path, text_column='clean_text_dl', batch_size=128, max_length=128):
    # 1. Khởi tạo GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang chạy trên thiết bị: {device}")

    # 2. Load RoBERTa Model và Tokenizer
    model_name = 'roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Đẩy model lên VRAM và bật chế độ suy luận (không train)
    model = model.to(device)
    model.eval()

    # 3. Đọc dữ liệu đã clean từ bước 1.2
    print("Đang tải dữ liệu...")
    df = pd.read_csv(data_path)
    # Đảm bảo text là string và xử lý null
    texts = df[text_column].fillna("").astype(str).tolist() 

    all_embeddings = []

    # 4. Trích xuất theo từng Batch
    print(f"Bắt đầu trích xuất đặc trưng với batch_size = {batch_size}...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Tiến độ"):
        batch_texts = texts[i : i + batch_size]
        
        # Tokenization: Chuyển chữ thành số
        inputs = tokenizer(batch_texts, 
                           return_tensors='pt', 
                           padding=True, 
                           truncation=True, 
                           max_length=max_length)
        
        # Chuyển batch hiện tại lên VRAM 16GB
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass (Tắt tính toán gradient để siêu tiết kiệm VRAM)
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Lấy vector của token <s> (tương đương [CLS] trong BERT) ở vị trí đầu tiên
        # Kích thước: (batch_size, hidden_size) -> (128, 768)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Đưa về lại RAM CPU và chuyển sang Numpy
        all_embeddings.append(cls_embeddings.cpu().numpy())

    # 5. Lưu kết quả ra file .npy
    print("Gộp các batch và lưu trữ...")
    final_embeddings = np.vstack(all_embeddings)
    
    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
    np.save(output_npy_path, final_embeddings)
    
    print(f"Thành công! File lưu tại: {output_npy_path}")
    print(f"Kích thước ma trận đặc trưng: {final_embeddings.shape}")

# Kích hoạt chạy
if __name__ == "__main__":
    extract_roberta_features(
        data_path='../data/interim/train_cleaned.csv',
        output_npy_path='../features/deep_learning/roberta_train_features.npy',
        batch_size=128 # 16GB VRAM cân tốt mức này
    )