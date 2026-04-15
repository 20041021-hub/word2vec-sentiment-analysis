# 机器学习实验：基于 Word2Vec 的情感预测

## 1. 学生信息
- **姓名**：[学生姓名]
- **学号**：[学生学号]
- **班级**：[班级]

> 注意：姓名和学号必须填写，否则本次实验提交无效。

---

## 2. 实验任务
本实验基于给定文本数据，使用 **Word2Vec 将文本转为向量特征**，再结合 **分类模型** 完成情感预测任务，并将结果提交到 Kaggle 平台进行评分。

本实验重点包括：
- 文本预处理
- Word2Vec 词向量训练或加载
- 句子向量表示
- 分类模型训练
- Kaggle 结果提交与分析

---

## 3. 比赛与提交信息
- **比赛名称**：Bag of Words Meets Bags of Popcorn
- **比赛链接**：https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview
- **提交日期**：2026-04-15

- **GitHub 仓库地址**：https://github.com/20041021-hub/word2vec-sentiment-analysis
- **GitHub README 地址**：https://github.com/20041021-hub/word2vec-sentiment-analysis/blob/main/README.md

> 注意：GitHub 仓库首页或 README 页面中，必须能看到“姓名 + 学号”，否则无效。

---

## 4. Kaggle 成绩
请填写你最终提交到 Kaggle 的结果：

- **Public Score**：[待填写]
- **Private Score**（如有）：[待填写]
- **排名**（如能看到可填写）：[待填写]

---

## 5. Kaggle 截图
请在下方插入 Kaggle 提交结果截图，要求能清楚看到分数信息。

![Kaggle截图](./images/kaggle_score.png)

> 建议将截图保存在 `images` 文件夹中。  
> 截图文件名示例：`2023123456_张三_kaggle_score.png`

---

## 6. 实验方法说明

### （1）文本预处理
请说明你对文本做了哪些处理，例如：
- 分词
- 去停用词
- 去除标点或特殊符号
- 转小写

**我的做法：**  
1. 去 HTML 标签：使用正则表达式去除所有 HTML 标签，如 `<br /><br />`
2. 小写化：将所有文本转换为小写
3. 标点处理：保留情感相关的标点（如感叹号和问号），移除其他标点
4. 停用词处理：使用 NLTK 提供的英文停用词表，保留否定词（如 not, no, never, nor）

---

### （2）Word2Vec 特征表示
请说明你如何使用 Word2Vec，例如：
- 是自己训练 Word2Vec，还是使用已有模型
- 词向量维度是多少
- 句子向量如何得到（平均、加权平均、池化等）

**我的做法：**  
1. 自己训练 Word2Vec 模型，使用 gensim 库
2. 词向量维度：100
3. 训练参数：window=5, min_count=5, workers=4, epochs=10
4. 句子向量：使用单词向量的平均值

---

### （3）分类模型
请说明你使用了什么分类模型，例如：
- Logistic Regression
- Random Forest
- SVM
- XGBoost

并说明最终采用了哪一个模型。

**我的做法：**  
尝试了三种分类模型：
1. Logistic Regression
2. Random Forest
3. SVM

最终选择 SVM 模型，因为其 AUC 指标最高（0.9438）。

---

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

**我的实验流程：**  
1. 读取训练集和测试集数据
2. 对文本进行预处理（去 HTML 标签、小写化、标点处理、停用词处理）
3. 训练 Word2Vec 模型
4. 将预处理后的文本转换为句向量（单词向量的平均值）
5. 划分训练集和验证集，训练多种分类模型并评估 AUC 指标
6. 选择最佳模型（SVM），对测试集进行预测
7. 生成 submission.csv 文件

---

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
word2vec-sentiment-analysis/
├─ labeledTrainData.tsv/  # 训练集数据
├─ testData.tsv/          # 测试集数据
├─ unlabeledTrainData.tsv/ # 无标签训练集数据
├─ preprocess.py           # 文本预处理脚本
├─ process_data.py         # 数据处理脚本
├─ train_word2vec.py       # Word2Vec模型训练脚本
├─ vectorize.py            # 文本向量化脚本
├─ train_model.py          # 分类模型训练脚本
├─ predict_test.py         # 测试集预测脚本
├─ submission.csv          # 提交文件
├─ README.md               # 实验报告
└─ requirements.txt        # 依赖包列表
```
