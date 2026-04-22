import pandas as pd
import re
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

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
    text = re.sub(r'<[^>]+>', '', text)
    # 转换为小写
    text = text.lower()
    # 处理否定词
    text = re.sub(r"n't", ' not', text)
    text = re.sub(r"can't", 'cannot', text)
    text = re.sub(r"won't", 'will not', text)
    text = re.sub(r"don't", 'do not', text)
    # 处理重复字符（如"aaaa"变为"aa"）
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # 移除标点符号和数字
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 分词
    words = text.split()
    # 移除停用词
    stop_words = get_stop_words()
    words = [word for word in words if word not in stop_words]
    # 移除长度小于2的单词
    words = [word for word in words if len(word) > 1]
    # 返回字符串
    return ' '.join(words)

# TF-IDF向量器初始化
def get_tfidf_vectorizer():
    return TfidfVectorizer(
        ngram_range=(1, 2),  # 启用短语模式，包括1-gram和2-gram
        max_features=3000,  # 减少特征数量以加快训练速度
        stop_words='english',  # 使用内置停用词列表
        min_df=2,  # 最小文档频率
        max_df=0.95  # 最大文档频率
    )

# 提取额外特征
def extract_additional_features(text):
    # 基本特征
    text_lower = text.lower()
    words = text_lower.split()
    text_length = len(words)
    
    # 情感词计数
    positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'superb', 'awesome', 'perfect', 'best',
                         'love', 'like', 'enjoy', 'brilliant', 'outstanding', 'exceptional', 'terrific', 'fabulous', 'incredible', 'marvelous'])
    negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'poor', 'worst', 'disappointing', 'horrendous', 'dreadful', 'pathetic',
                         'hate', 'dislike', 'disgusting', 'frustrating', 'annoying', 'boring', 'tedious', 'dull', 'monotonous', 'depressing'])
    
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    # 否定词计数
    negation_words = set(['not', 'no', 'never', 'none', 'nobody', 'nothing', 'nowhere', 'hardly', 'scarcely', 'barely', 'seldom', "n't"])
    negation_count = sum(1 for word in words if word in negation_words)
    
    # 单词多样性（不同单词的比例）
    word_diversity = len(set(words)) / text_length if text_length > 0 else 0
    
    # 情感强度（情感词比例）
    sentiment_intensity = (positive_count + negative_count) / text_length if text_length > 0 else 0
    
    # 否定词比例
    negation_ratio = negation_count / text_length if text_length > 0 else 0
    
    # 正面情感比例
    positive_ratio = positive_count / text_length if text_length > 0 else 0
    
    # 负面情感比例
    negative_ratio = negative_count / text_length if text_length > 0 else 0
    
    # 情感词位置（靠前的情感词影响更大）
    positive_position = 0
    negative_position = 0
    for i, word in enumerate(words):
        if word in positive_words and positive_position == 0:
            positive_position = (i + 1) / text_length if text_length > 0 else 0
        if word in negative_words and negative_position == 0:
            negative_position = (i + 1) / text_length if text_length > 0 else 0
    
    # 感叹号和问号数量
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    # 大写字母比例（通常表示强调）
    uppercase_count = sum(1 for c in text if c.isupper())
    uppercase_ratio = uppercase_count / len(text) if len(text) > 0 else 0
    
    # 情感得分（考虑否定词的影响）
    sentiment_score = (positive_count - negative_count) / text_length if text_length > 0 else 0
    if negation_count > 0:
        sentiment_score = -sentiment_score
    
    return [text_length, positive_count, negative_count, negation_count, word_diversity, sentiment_intensity, negation_ratio, positive_ratio, negative_ratio, positive_position, negative_position, exclamation_count, question_count, uppercase_ratio, sentiment_score]

# 读取训练数据
print("Reading training data...")
train_df = pd.read_csv('labeledTrainData.tsv', sep='\t', low_memory=True, nrows=15000)  # 减少训练数据量以加快训练速度

# 预处理训练数据
print("Preprocessing training data...")
train_df['processed_review'] = train_df['review'].apply(preprocess_text)
train_df['additional_features'] = train_df['review'].apply(extract_additional_features)

# 初始化并拟合TF-IDF向量器
print("Training TF-IDF vectorizer...")
tfidf_vectorizer = get_tfidf_vectorizer()
train_tfidf_features = tfidf_vectorizer.fit_transform(train_df['processed_review'])

# 合并特征
print("Merging features...")
train_additional_features = np.array(train_df['additional_features'].tolist())
train_tfidf_features_dense = train_tfidf_features.toarray()
train_features = np.hstack([train_tfidf_features_dense, train_additional_features])

# 标准化特征
print("Standardizing features...")
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
train_labels = train_df['sentiment']

# 网格搜索优化逻辑回归参数
print("Optimizing logistic regression parameters...")
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear']
}
grid_search = GridSearchCV(
    LogisticRegression(max_iter=200),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1  # 使用单线程以减少内存使用
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

# 计算测试数据的TF-IDF特征
print("Calculating TF-IDF features for test data...")
test_tfidf_features = tfidf_vectorizer.transform(test_df['processed_review'])

# 合并特征
print("Merging features...")
test_additional_features = np.array(test_df['additional_features'].tolist())
test_tfidf_features_dense = test_tfidf_features.toarray()
test_features = np.hstack([test_tfidf_features_dense, test_additional_features])

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
