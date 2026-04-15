import re
import nltk
from nltk.corpus import stopwords

# 下载停用词
nltk.download('stopwords')

# 加载英文停用词
stop_words = set(stopwords.words('english'))

# 否定词列表
negation_words = {'not', 'no', 'never', 'nor'}

# 移除HTML标签
def remove_html_tags(text):
    return re.sub(r'<[^>]+>', '', text)

# 小写化
def to_lowercase(text):
    return text.lower()

# 标点处理
def handle_punctuation(text):
    # 保留情感相关的标点，如感叹号和问号
    text = re.sub(r'[.,;:]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 停用词处理（保留否定词）
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words or word in negation_words]
    return ' '.join(filtered_words)

# 完整的预处理流程
def preprocess_text(text):
    text = remove_html_tags(text)
    text = to_lowercase(text)
    text = handle_punctuation(text)
    text = remove_stopwords(text)
    return text

# 测试预处理函数
if __name__ == '__main__':
    test_text = "<br /><br />This movie is not bad at all! I really enjoyed it."
    print("原始文本:", test_text)
    print("预处理后:", preprocess_text(test_text))
