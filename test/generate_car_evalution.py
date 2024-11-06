import pandas as pd

# 载入 Car Evaluation 数据集
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv(url, names=column_names)

# 将类标签转换为数字
# df['class'] = df['class'].map({'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3})

# 打印数据集前几行
print(df.head())

# 保存到本地 CSV 文件
df.to_csv('car_evaluation.csv', index=False)
