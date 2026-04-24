import pandas as pd
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack, csr_matrix
from bs4 import BeautifulSoup
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 下载必要的NLTK数据
nltk.download('vader_lexicon', quiet=True)

# 扩展的停用词列表（不包含否定词）
def get_stop_words():
    return set([
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'should', 'could', 'may', 'might', 'must', 'shall', 'can', 'cannot',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'what', 'which', 'who', 'whom',
        'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
        'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 's', 't', 'can', 'will', 'don', 'should', 'now'
    ])

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
    text = re.sub(r"isn't", 'is not', text)
    text = re.sub(r"aren't", 'are not', text)
    text = re.sub(r"wasn't", 'was not', text)
    text = re.sub(r"weren't", 'were not', text)
    text = re.sub(r"hasn't", 'has not', text)
    text = re.sub(r"haven't", 'have not', text)
    text = re.sub(r"hadn't", 'had not', text)
    text = re.sub(r"doesn't", 'does not', text)
    text = re.sub(r"didn't", 'did not', text)
    text = re.sub(r"shouldn't", 'should not', text)
    text = re.sub(r"wouldn't", 'would not', text)
    text = re.sub(r"couldn't", 'could not', text)
    text = re.sub(r"mightn't", 'might not', text)
    text = re.sub(r"mustn't", 'must not', text)
    # 处理重复字符（如"aaaa"变为"aa"）
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # 移除多余的空格
    text = re.sub(r"\s+", ' ', text).strip()
    # 分词
    words = text.split()
    # 移除停用词
    stop_words = get_stop_words()
    words = [word for word in words if word not in stop_words]
    # 移除长度小于2的单词
    words = [word for word in words if len(word) > 1]
    # 返回字符串
    return ' '.join(words)

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

# 词级CountVectorizer初始化
def get_word_count_vectorizer():
    return CountVectorizer(
        min_df=2,
        max_df=0.85,
        max_features=30000,
        ngram_range=(1, 2),
        stop_words='english'
    )

# 情感词典特征
def get_sentiment_features(texts):
    # 简单的情感词典
    positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome', 'love', 'like', 'best'])
    negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'dislike', 'hate', 'worst', 'poor', 'boring', 'dreadful'])
    
    features = []
    for text in texts:
        words = text.split()
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        features.append([pos_count, neg_count, pos_count - neg_count])
    
    return np.array(features)

# 文本长度特征
def get_text_length_features(texts):
    features = []
    for text in texts:
        words = text.split()
        features.append([len(text), len(words), len(set(words))])
    
    return np.array(features)

# VADER情感分析特征
def get_vader_features(texts):
    sia = SentimentIntensityAnalyzer()
    features = []
    for text in texts:
        sentiment = sia.polarity_scores(text)
        features.append([sentiment['compound'], sentiment['pos'], sentiment['neg'], sentiment['neu']])
    
    return np.array(features)

# 文本复杂度特征
def get_complexity_features(texts):
    features = []
    for text in texts:
        words = text.split()
        # 词汇多样性
        lexical_diversity = len(set(words)) / len(words) if words else 0
        # 平均句子长度
        sentences = re.split(r'[.!?]', text)
        sentences = [s for s in sentences if s.strip()]
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        # 长词比例（长度大于6）
        long_word_ratio = sum(1 for word in words if len(word) > 6) / len(words) if words else 0
        features.append([lexical_diversity, avg_sentence_length, long_word_ratio])
    
    return np.array(features)

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
count_vec = get_word_count_vectorizer()

X_word_all = word_vec.fit_transform(all_text)
X_char_all = char_vec.fit_transform(all_text)
X_count_all = count_vec.fit_transform(all_text)

# 提取情感特征
print("Extracting sentiment features...")
sentiment_features = get_sentiment_features(all_text)

# 提取文本长度特征
print("Extracting text length features...")
length_features = get_text_length_features(all_text)

# 提取额外特征
print("Extracting additional features...")
additional_features = []
for text in all_text:
    words = text.split()
    # 句子数量
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    # 平均词长
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    # 感叹号和问号数量
    exclamation_count = text.count('!')
    question_count = text.count('?')
    additional_features.append([sentence_count, avg_word_length, exclamation_count, question_count])
additional_features = np.array(additional_features)

# 提取VADER情感特征
print("Extracting VADER sentiment features...")
vader_features = get_vader_features(all_text)

# 提取文本复杂度特征
print("Extracting text complexity features...")
complexity_features = get_complexity_features(all_text)

# 标准化数值特征
scaler = StandardScaler()
sentiment_features = scaler.fit_transform(sentiment_features)
length_features = scaler.fit_transform(length_features)
additional_features = scaler.fit_transform(additional_features)
vader_features = scaler.fit_transform(vader_features)
complexity_features = scaler.fit_transform(complexity_features)

# 转换为稀疏矩阵
sentiment_features_sparse = csr_matrix(sentiment_features)
length_features_sparse = csr_matrix(length_features)
additional_features_sparse = csr_matrix(additional_features)
vader_features_sparse = csr_matrix(vader_features)
complexity_features_sparse = csr_matrix(complexity_features)

# 合并所有特征
X_all = hstack([X_word_all, X_char_all, X_count_all, sentiment_features_sparse, length_features_sparse, additional_features_sparse, vader_features_sparse, complexity_features_sparse]).tocsr()

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

    # ----- Model 3: XGBoost -----
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=1993,
        n_jobs=-1
    )
    xgb.fit(X_tr, y_tr)
    p3 = xgb.predict_proba(X_va)[:, 1]

    # ----- Model 4: LightGBM -----
    lgb = LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=1993,
        n_jobs=-1
    )
    lgb.fit(X_tr, y_tr)
    p4 = lgb.predict_proba(X_va)[:, 1]

    # ----- Model 5: CatBoost -----
    cat = CatBoostClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bylevel=0.8,
        random_state=1993,
        verbose=0
    )
    cat.fit(X_tr, y_tr)
    p5 = cat.predict_proba(X_va)[:, 1]

    # ----- Ensemble: Weighted Blend -----
    # 基于交叉验证结果调整权重
    p = 0.2 * p1 + 0.2 * p2 + 0.2 * p3 + 0.2 * p4 + 0.2 * p5
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

# 训练 XGBoost 模型
final_xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=1993,
    n_jobs=-1
)
final_xgb.fit(X, y)

# 训练 LightGBM 模型
final_lgb = LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=1993,
    n_jobs=-1
)
final_lgb.fit(X, y)

# 训练 CatBoost 模型
final_cat = CatBoostClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bylevel=0.8,
    random_state=1993,
    verbose=0
)
final_cat.fit(X, y)

# 预测测试数据
print("Predicting test set...")
test_p1 = final_lr1.predict_proba(X_test)[:, 1]
test_p2 = final_lr2.predict_proba(X_test_nb)[:, 1]
test_p3 = final_xgb.predict_proba(X_test)[:, 1]
test_p4 = final_lgb.predict_proba(X_test)[:, 1]
test_p5 = final_cat.predict_proba(X_test)[:, 1]

# 使用加权融合
test_pred = 0.2 * test_p1 + 0.2 * test_p2 + 0.2 * test_p3 + 0.2 * test_p4 + 0.2 * test_p5

# 生成提交文件
print("Generating submission file...")
submission = pd.DataFrame({
    'id': test_df['id'],
    'sentiment': test_pred
})

submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
print(f"Submission file has {len(submission)} rows")
