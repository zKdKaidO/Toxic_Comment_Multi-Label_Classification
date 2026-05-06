# Project: Toxic Comment Classification

## 📌 1. General Information
- **Course:** Machine Learning
- **Course Code:** CO3117
- **Semester:** HK252
- **Class:** CC01
- **Instructor:** Truong Vinh Lan
- **Lecturer in Charge:** Dr. Le Thanh Sach

### 👥 Team Members
| No. | Full Name | Student ID | Email | Contribution |
|:---:|:---|:---:|:---|:---:|
| 1 | Pham Nhat Nam | 2352785 | nam.phamnhat1301@hcmut.edu.vn | 100% |
| 2 | Tran Thuy Thuy Truc | 2353259 | truc.tran2105@hcmut.edu.vn | 100% |
| 3 | Nguyen Vu Long | 2352696 | long.nguyenty@hcmut.edu.vn | 100% |
| 4 | Le Phuoc Vu | 2353341 | vu.lephuocvu13@hcmut.edu.vn | 100% |
| 5 | Nguyen Hoang Quoc | 2353027 | quoc.nguyenhoang2305@hcmut.edu.vn | 100% |

*(Please refer to the PDF Report for detailed task assignments).*

---

## 🎯 2. Project Objectives
This project aims to build a Machine Learning / Deep Learning system capable of automatically analyzing and classifying English comments on cyberspace into 6 toxic behavior categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.

The system is designed to solve a Multi-label Classification problem, handle Imbalanced Data, and compare the performance between a Traditional Feature Extraction method and a Large Language Model (LLM) approach.

---

## 📂 3. Project Structure
```text
Project_Toxic_Comment/
│
├── notebooks/
│   └── CommentClassification.ipynb   # Main Google Colab notebook containing the entire workflow
│
├── modules/
│   └── text_preprocessing.py             # Python source code with text preprocessing functions
│
├── features/                             # (Do not push to Github if files are too large)
│   ├── tfidf_train_embeddings.npz        # Extracted features using TF-IDF
│   └── X_train_embeddings.npy            # Extracted features using DistilRoBERTa
│
├── models/                               # (Do not push to Github if files are too large)
│   ├── logreg_ovr.joblib                 # Trained weights of the Traditional ML model
│   └── pytorch_logreg_smoothed.pth       # Trained weights of the Deep Learning model
│
├── reports/
│   └── Group_Report.pdf                  # Scientific report in PDF format
│
├── requirements.txt                      # List of required Python dependencies
└── README.md                             # Project documentation
```

---

## 🚀 4. How to Run

### a. Prerequisites
- Python 3.9+
- Git

### b. Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd Project_Toxic_Comment
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook:**
   Open and run the `notebooks/01_toxic_comment_pipeline.ipynb` file using Jupyter Notebook or Google Colab. The notebook will automatically download the dataset from Hugging Face.

---

## 🔗 5. Links
- **PDF Report:** View Report
- **Google Colab Notebook:** Open in Colab