import pandas as pd
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from bs4 import BeautifulSoup

# 文本预处理函数
def preprocess_text(text):
    # 移除HTML标签
    text = BeautifulSoup(text, "lxml").get_text()
    # 移除非字母字符，但保留数字和一些标点
    text = re.sub(r"[^a-zA-Z0-9\s!?.]", ' ', text)
    # 转换为小写
    text = text.lower()
    # 处理否定词
    text = re.sub(r"n't", ' not', text)
    text = re.sub(r"can't", 'cannot', text)
    text = re.sub(r"won't", 'will not', text)
    text = re.sub(r"don't", 'do not', text)
    # 处理重复字符（如"aaaa"变为"aa"）
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # 移除多余的空格
    text = re.sub(r"\s+", ' ', text).strip()
    return text

# 词级TF-IDF向量器初始化
def get_word_vectorizer():
    return TfidfVectorizer(
        min_df=2,
        max_df=0.85,
        max_features=50000,
        ngram_range=(1, 3),
        stop_words='english',
        sublinear_tf=True
    )

# 字符级TF-IDF向量器初始化
def get_char_vectorizer():
    return TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 6),
        min_df=2,
        max_features=50000,
        sublinear_tf=True
    )

# NB-SVM log-count ratio
def nbsvm_ratio(X, y, alpha=1.0):
    y = np.asarray(y)
    pos = X[y == 1].sum(axis=0) + alpha
    neg = X[y == 0].sum(axis=0) + alpha
    pos = pos / pos.sum()
    neg = neg / neg.sum()
    r = np.log(pos / neg)
    return np.asarray(r).ravel()  # 1D

# 读取训练数据
print("Reading training data...")
train_df = pd.read_csv('labeledTrainData.tsv', sep='\t', low_memory=True)

# 读取测试数据
print("Reading test data...")
test_df = pd.read_csv('testData.tsv', sep='\t', low_memory=True)

print(f"Train/Test shapes: {train_df.shape}, {test_df.shape}")

# 预处理文本
print("Preprocessing text...")
train_text = train_df['review'].apply(preprocess_text).tolist()
test_text = test_df['review'].apply(preprocess_text).tolist()
y = train_df['sentiment'].values

# 合并所有文本以训练TF-IDF向量器
all_text = train_text + test_text

# 提取词级和字符级TF-IDF特征
print("Building TF-IDF features (word + char)...")
word_vec = get_word_vectorizer()
char_vec = get_char_vectorizer()

X_word_all = word_vec.fit_transform(all_text)
X_char_all = char_vec.fit_transform(all_text)

# 合并特征
X_all = hstack([X_word_all, X_char_all]).tocsr()

# 分割训练和测试特征
X = X_all[:len(train_text)]
X_test = X_all[len(train_text):]

print(f"X shape: {X.shape}, X_test shape: {X_test.shape}")

# 7折交叉验证
N_FOLDS = 7
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=1993)

oof = np.zeros(len(y))
fold_scores = []

print("\nStarting cross-validation...")
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n========== FOLD {fold} / {N_FOLDS} ==========")
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    # ----- Model 1: plain LR -----
    lr1 = LogisticRegression(
        C=4.0,
        max_iter=1000,
        solver="liblinear"
    )
    lr1.fit(X_tr, y_tr)
    p1 = lr1.predict_proba(X_va)[:, 1]

    # ----- Model 2: NB-SVM style LR -----
    r = nbsvm_ratio(X_tr, y_tr, alpha=1.0)
    X_tr_nb = X_tr.multiply(r)
    X_va_nb = X_va.multiply(r)

    lr2 = LogisticRegression(
        C=4.0,
        max_iter=1000,
        solver="liblinear"
    )
    lr2.fit(X_tr_nb, y_tr)
    p2 = lr2.predict_proba(X_va_nb)[:, 1]

    # ----- Blend -----
    p = 0.5 * p1 + 0.5 * p2
    oof[va_idx] = p

    fold_auc = roc_auc_score(y_va, p)
    fold_scores.append(fold_auc)
    print(f"Fold {fold} AUC: {fold_auc:.5f}")

cv_auc = roc_auc_score(y, oof)
print("\n==============================")
print(f"OOF CV AUC: {cv_auc:.5f}")
print(f"Fold AUCs: {[round(s, 5) for s in fold_scores]}")
print("==============================")

# 训练最终模型
print("\nTraining final models on FULL data...")
final_lr1 = LogisticRegression(
    C=4.0,
    max_iter=1000,
    solver="liblinear"
)
final_lr1.fit(X, y)

r_full = nbsvm_ratio(X, y, alpha=1.0)
X_nb_full = X.multiply(r_full)
X_test_nb = X_test.multiply(r_full)

final_lr2 = LogisticRegression(
    C=4.0,
    max_iter=1000,
    solver="liblinear"
)
final_lr2.fit(X_nb_full, y)

# 预测测试数据
print("Predicting test set...")
test_p1 = final_lr1.predict_proba(X_test)[:, 1]
test_p2 = final_lr2.predict_proba(X_test_nb)[:, 1]
test_pred = 0.5 * test_p1 + 0.5 * test_p2

# 生成提交文件
print("Generating submission file...")
submission = pd.DataFrame({
    'id': test_df['id'],
    'sentiment': test_pred
})

submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
print(f"Submission file has {len(submission)} rows")