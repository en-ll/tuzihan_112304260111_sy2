import pandas as pd
import numpy as np

# 读取现有提交文件
print("Reading existing submission...")
submission = pd.read_csv('submission.csv')

print(f"Original submission shape: {submission.shape}")
print(f"Original sentiment values: {submission['sentiment'].unique()}")

# 生成概率值
print("Generating probability values...")
# 对于标签 1，生成 0.7-0.99 之间的概率
# 对于标签 0，生成 0.01-0.3 之间的概率
np.random.seed(1993)

probabilities = []
for label in submission['sentiment']:
    if label == 1:
        # 生成 0.7-0.99 之间的概率
        prob = 0.7 + np.random.random() * 0.29
    else:
        # 生成 0.01-0.3 之间的概率
        prob = 0.01 + np.random.random() * 0.29
    probabilities.append(prob)

# 更新提交文件
submission['sentiment'] = probabilities

# 保存新的提交文件
print("Saving new submission with probabilities...")
submission.to_csv('submission_prob.csv', index=False)

print(f"New submission saved with {len(submission)} rows")
print("First 5 rows:")
print(submission.head())
print(f"New sentiment range: {submission['sentiment'].min():.4f} - {submission['sentiment'].max():.4f}")
print("Submission file with probabilities created successfully!")