import pandas as pd
import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# 扩展的停用词列表
def get_stop_words():
    return set([
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'should', 'could', 'may', 'might', 'must', 'shall', 'can', 'cannot',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'what', 'which', 'who', 'whom',
        'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 's', 't', 'can', 'will', 'don', 'should', 'now'
    ])

# 文本预处理函数
def preprocess_text(text):
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 转换为小写
    text = text.lower()
    # 处理否定词
    text = re.sub(r"n't", ' not', text)
    text = re.sub(r"can't", 'cannot', text)
    # 移除标点符号和数字
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 分词
    words = text.split()
    # 移除停用词
    stop_words = get_stop_words()
    words = [word for word in words if word not in stop_words]
    # 移除长度小于2的单词
    words = [word for word in words if len(word) > 1]
    return words

# 计算文档的均值嵌入
def get_mean_embedding(model, words):
    embeddings = []
    for word in words:
        if word in model.wv:
            embeddings.append(model.wv[word])
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)

# 计算文档的加权均值嵌入（使用词频）
def get_weighted_embedding(model, words):
    embeddings = []
    weights = []
    word_freq = {}
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    
    total_freq = sum(word_freq.values())
    for word in words:
        if word in model.wv:
            embeddings.append(model.wv[word])
            weights.append(word_freq[word] / total_freq)
    
    if embeddings:
        return np.average(embeddings, axis=0, weights=weights)
    else:
        return np.zeros(model.vector_size)

# 提取额外特征
def extract_additional_features(text):
    # 文本长度
    text_length = len(text.split())
    # 情感词计数
    positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'superb', 'awesome', 'perfect', 'best'])
    negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'poor', 'worst', 'disappointing', 'horrendous', 'dreadful', 'pathetic'])
    words = text.lower().split()
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    # 否定词计数
    negation_words = set(['not', 'no', 'never', 'none', 'nobody', 'nothing', 'nowhere', 'hardly', 'scarcely', 'barely', 'seldom'])
    negation_count = sum(1 for word in words if word in negation_words)
    return [text_length, positive_count, negative_count, negation_count]

# 读取训练数据
print("Reading training data...")
train_df = pd.read_csv('labeledTrainData.tsv', sep='\t', low_memory=True, nrows=15000)  # 增加训练数据量

# 预处理训练数据
print("Preprocessing training data...")
train_df['processed_review'] = train_df['review'].apply(preprocess_text)
train_df['additional_features'] = train_df['review'].apply(extract_additional_features)

# 训练Word2Vec模型（优化参数）
print("Training Word2Vec model...")
model = Word2Vec(
    train_df['processed_review'], 
    vector_size=150,  # 增加向量大小
    window=7,  # 增加窗口大小
    min_count=2,  # 增加最小词频
    workers=4,
    sg=1,  # 使用skip-gram模型
    epochs=10  # 增加训练轮数
)

# 计算训练数据的嵌入
print("Calculating embeddings for training data...")
train_mean_embeddings = np.array([get_mean_embedding(model, words) for words in train_df['processed_review']])
train_weighted_embeddings = np.array([get_weighted_embedding(model, words) for words in train_df['processed_review']])

# 合并特征
print("Merging features...")
train_additional_features = np.array(train_df['additional_features'].tolist())
train_features = np.hstack([train_mean_embeddings, train_weighted_embeddings, train_additional_features])

# 标准化特征
print("Standardizing features...")
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
train_labels = train_df['sentiment']

# 网格搜索优化逻辑回归参数
print("Optimizing logistic regression parameters...")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
grid_search = GridSearchCV(
    LogisticRegression(max_iter=200),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(train_features, train_labels)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")

# 训练最佳模型
print("Training best model...")
best_classifier = grid_search.best_estimator_

# 读取测试数据
print("Reading test data...")
test_df = pd.read_csv('testData.tsv', sep='\t', low_memory=True)

# 预处理测试数据
print("Preprocessing test data...")
test_df['processed_review'] = test_df['review'].apply(preprocess_text)
test_df['additional_features'] = test_df['review'].apply(extract_additional_features)

# 计算测试数据的嵌入
print("Calculating embeddings for test data...")
test_mean_embeddings = np.array([get_mean_embedding(model, words) for words in test_df['processed_review']])
test_weighted_embeddings = np.array([get_weighted_embedding(model, words) for words in test_df['processed_review']])

# 合并特征
print("Merging features...")
test_additional_features = np.array(test_df['additional_features'].tolist())
test_features = np.hstack([test_mean_embeddings, test_weighted_embeddings, test_additional_features])

# 标准化测试特征
print("Standardizing test features...")
test_features = scaler.transform(test_features)

# 预测
print("Predicting sentiment...")
predictions = best_classifier.predict(test_features)

# 生成提交文件
print("Generating submission file...")
submission = pd.DataFrame({'id': test_df['id'], 'sentiment': predictions})
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
print(f"Submission file has {len(submission)} rows")
