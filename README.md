# 机器学习实验：基于 Word2Vec 的情感预测

## 1. 学生信息

- **姓名**：屠子涵
- **学号**：112304260111
- **班级**：数据1231

> 注意：姓名和学号必须填写，否则本次实验提交无效。

***

## 2. 实验任务

本实验基于给定文本数据，使用 **Word2Vec 将文本转为向量特征**，再结合 **分类模型** 完成情感预测任务，并将结果提交到 Kaggle 平台进行评分。

本实验重点包括：

- 文本预处理
- Word2Vec 词向量训练或加载
- 句子向量表示
- 分类模型训练
- Kaggle 结果提交与分析

***

## 3. 比赛与提交信息

- **比赛名称**：**《词袋遇上爆米花袋》**
- **比赛链接**：[Bag of Words Meets Bags of Popcorn | Kaggle](https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview)
- **提交日期**：4.21
- **GitHub 仓库地址**：[en-ll/tuzihan\_112304260111\_sy2](https://github.com/en-ll/tuzihan_112304260111_sy2)
- **GitHub README 地址**：[exp1: 修改情感分析脚本，改用TF-IDF短语模式并保留否定词 · en-ll/tuzihan\_112304260111\_sy2@a679c2a](https://github.com/en-ll/tuzihan_112304260111_sy2/commit/a679c2a271f72be21f88095a25987bfe2eb90869#diff-b335630551682c19a781afebcf4d07bf978fb1f8ac04c6bf87428ed5106870f5)

> 注意：GitHub 仓库首页或 README 页面中，必须能看到“姓名 + 学号”，否则无效。

***

## 4. Kaggle 成绩

请填写你最终提交到 Kaggle 的结果：

- **Public Score**：0.86316
- **Private Score**（如有）：0.86316
- **排名**（如能看到可填写）：

***

## 5. Kaggle 截图

请在下方插入 Kaggle 提交结果截图，要求能清楚看到分数信息。

![Kaggle截图](./images/kaggle_score.png)

> 建议将截图保存在 `images` 文件夹中。\
> 截图文件名示例：`2023123456_张三_kaggle_score.png`

***

## 6. 实验方法说明

### （1）文本预处理

请说明你对文本做了哪些处理，例如：

- 分词
- 去停用词
- 去除标点或特殊符号
- 转小写

**我的做法：**

1. 移除HTML标签
2. 转换为小写
3. 移除标点符号
4. 分词
5. 移除停用词
6. 处理否定词（如将"n't"转换为"not"）
7. 移除长度小于2的单词

***

### （2）TF-IDF 特征表示

请说明你如何使用 TF-IDF，例如：

- 是否使用短语模式
- 特征数量是多少
- 其他参数设置

**我的做法：**

1. 使用TF-IDF的短语模式，ngram\_range=(1, 2)，包括1-gram和2-gram
2. 最大特征数量：3000
3. 最小文档频率：2
4. 最大文档频率：0.95
5. 使用内置停用词列表

***

### （3）分类模型

请说明你使用了什么分类模型，例如：

- Logistic Regression
- Random Forest
- SVM
- XGBoost

并说明最终采用了哪一个模型。

**我的做法：**\
请在这里填写。

***

## 7. 实验流程

请简要说明你的实验流程。

示例：

1. 读取训练集和测试集
2. 对文本进行预处理
3. 训练或加载 Word2Vec 模型
4. 将每条文本表示为句向量
5. 用训练集训练分类器
6. 在测试集上预测结果
7. 生成 submission 文件并提交 Kaggle

**我的实验流程：**\
请在这里填写。

***

## 8. 文件说明

请说明仓库中各文件或文件夹的作用。

示例：

- `data/`：存放数据文件
- `src/`：存放源代码
- `notebooks/`：存放实验 notebook
- `images/`：存放 README 中使用的图片
- `submission/`：存放提交文件

**我的项目结构：**

```text
f:\Trae\sy2\
├─ sentiment_analysis.py   # 主要代码文件，实现情感分析功能
├─ README.md               # 实验报告
├─ requirements.txt        # Python依赖
├─ .env.example            # 环境变量模板
├─ .env                    # 本地环境变量文件（不上传）
├─ .gitignore              # Git忽略文件配置
├─ submission.csv          # 生成的提交文件
└─ submission.zip          # 压缩的提交文件
<<<<<<< HEAD
```

=======
```
>>>>>>> a679c2a271f72be21f88095a25987bfe2eb90869
