import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score

# 加载向量和标签
train_vectors = np.load('train_vectors.npy')
train_labels = np.load('train_labels.npy')

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    train_vectors, train_labels, test_size=0.2, random_state=42
)

# 训练并评估多种模型
print("训练基础模型...")
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

best_model = None
best_auc = 0

for model_name, model in models.items():
    print(f"训练 {model_name}...")
    model.fit(X_train, y_train)
    
    # 预测概率
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # 计算AUC
    auc = roc_auc_score(y_val, y_pred_proba)
    
    # 计算准确率
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    
    print(f"{model_name} - AUC: {auc:.4f}, Accuracy: {acc:.4f}")
    
    # 更新最佳模型
    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_model_name = model_name

# 调优Logistic Regression模型（因为它已经表现很好）
print("\n调优Logistic Regression模型...")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs']
}

grid_search = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=1000),
    param_grid=param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"最佳Logistic Regression参数: {grid_search.best_params_}")
print(f"最佳交叉验证AUC: {grid_search.best_score_:.4f}")

# 评估最佳Logistic Regression模型
best_lr = grid_search.best_estimator_
y_pred_proba = best_lr.predict_proba(X_val)[:, 1]
test_auc = roc_auc_score(y_val, y_pred_proba)
print(f"测试集AUC: {test_auc:.4f}")

# 更新最佳模型
if test_auc > best_auc:
    best_model = best_lr
    best_auc = test_auc
    best_model_name = f"Logistic Regression (tuned)"

print(f"\n最佳模型: {best_model_name}, AUC: {best_auc:.4f}")

# 保存最佳模型
import joblib
joblib.dump(best_model, 'best_model.joblib')
print("最佳模型已保存为 best_model.joblib")
