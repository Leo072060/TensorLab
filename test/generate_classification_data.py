import pandas as pd
from sklearn.datasets import load_wine

# 加载葡萄酒数据集
wine = load_wine()

# 创建 DataFrame
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
df['target'] = wine.target

# 将目标变量转换为类型（可选）
df['target'] = df['target'].astype('category')
df.to_csv('classification_data.csv', index=False)

