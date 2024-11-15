import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

def generate_classification_data(n_samples=3000, n_features=5, n_classes=4, n_informative=4, n_redundant=0):
    # 生成分类数据集
    X, y = make_classification(n_samples=n_samples,    # 样本数量
                               n_features=n_features,   # 特征数量
                               n_classes=n_classes,    # 类别数量
                               n_informative=n_informative,  # 有用特征数量
                               n_redundant=n_redundant,  # 冗余特征数量
                               n_clusters_per_class=1,
                               random_state=42)

    # 创建 DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y

    # 将目标变量转换为类型（可选）
    df['target'] = df['target'].astype('category')

    # 保存为 CSV 文件
    df.to_csv('classification_data.csv', index=False)

    return df

# 生成数据
df = generate_classification_data(n_samples=5000, n_features=4, n_classes=5, n_informative=4, n_redundant=0)

# 分离特征和目标变量
X = df.drop('target', axis=1)
y = df['target']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练模型
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 生成混淆矩阵和分类报告
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# 打印混淆矩阵和分类报告
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

 
