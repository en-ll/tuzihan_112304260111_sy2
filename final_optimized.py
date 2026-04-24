import pandas as pd
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from bs4 import BeautifulSoup

# 检查文件是否存在
import os
print("Checking files...")
print(f"labeledTrainData.tsv exists: {os.path.exists('labeledTrainData.tsv')}")
print(f"testData.tsv exists: {os.path.exists('testData.tsv')}")

# 文本预处理函数
def preprocess_text(text):
    try:
        text = BeautifulSoup(text, "lxml").get_text()
    except:
        pass
    text = re.sub(r"[^a-zA-Z0-9\s!?.]", ' ', text)
    text = text.lower()
    text = re.sub(r"n't", ' not', text)
    text = re.sub(r"can't", 'cannot', text)
    text = re.sub(r"won't", 'will not', text)
    text = re.sub(r"don't", 'do not', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

# 读取数据
print("Reading data...")
train_df = pd.read_csv('labeledTrainData.tsv', sep='\t', low_memory=True)
test_df = pd.read_csv('testData.tsv', sep='\t', low_memory=True)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# 预处理文本
print("Preprocessing text...")
train_text = train_df['review'].apply(preprocess_text).tolist()
test_text = test_df['review'].apply(preprocess_text).tolist()
y = train_df['sentiment'].values

# 合并文本
all_text = train_text + test_text

# 提取特征
print("Extracting features...")
word_vec = TfidfVectorizer(
    min_df=3,
    max_df=0.9,
    max_features=40000,
    ngram_range=(1, 2),
    stop_words='english',
    sublinear_tf=True
)

char_vec = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 5),
    min_df=3,
    max_features=40000,
    sublinear_tf=True
)

X_word = word_vec.fit_transform(all_text)
X_char = char_vec.fit_transform(all_text)
X_all = hstack([X_word, X_char]).tocsr()

# 分割数据
X_train = X_all[:len(train_text)]
X_test = X_all[len(train_text):]

# NB-SVM log-count ratio
def nbsvm_ratio(X, y, alpha=1.0):
    y = np.asarray(y)
    pos = X[y == 1].sum(axis=0) + alpha
    neg = X[y == 0].sum(axis=0) + alpha
    pos = pos / pos.sum()
    neg = neg / neg.sum()
    r = np.log(pos / neg)
    return np.asarray(r).ravel()

# 训练模型
print("Training models...")

# 模型1: 标准LR
lr1 = LogisticRegression(C=4.0, max_iter=800, solver="liblinear")
lr1.fit(X_train, y)

# 模型2: NB-SVM风格LR
r = nbsvm_ratio(X_train, y, alpha=1.0)
X_train_nb = X_train.multiply(r)
lr2 = LogisticRegression(C=4.0, max_iter=800, solver="liblinear")
lr2.fit(X_train_nb, y)

# 预测
print("Predicting...")
test_p1 = lr1.predict_proba(X_test)[:, 1]
X_test_nb = X_test.multiply(r)
test_p2 = lr2.predict_proba(X_test_nb)[:, 1]
test_pred = 0.5 * test_p1 + 0.5 * test_p2

# 生成提交文件
print("Generating submission...")
submission = pd.DataFrame({
    'id': test_df['id'],
    'sentiment': test_pred
})

submission.to_csv('submission.csv', index=False)
print(f"Submission saved with {len(submission)} rows")
print("First 5 rows:")
print(submission.head())
print("Submission file created successfully!")