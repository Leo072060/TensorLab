import pandas as pd
import numpy as np

# 设置随机种子以便结果可重复
np.random.seed(42)

# 生成数据
num_samples = 10000
feature1 = np.random.rand(num_samples) * 100  # 特征1
feature2 = np.random.rand(num_samples) * 50   # 特征2
feature3 = np.random.rand(num_samples) * 25   # 特征3
feature4 = np.random.rand(num_samples) * 10   # 特征4
feature5 = np.random.rand(num_samples) * 20   # 特征5
# feature6 = np.random.rand(num_samples) * 15   # 特征6
# feature7 = np.random.rand(num_samples) * 30   # 特征7
noise = np.random.randn(num_samples) * 0.001      # 随机噪声

# 生成目标变量（假设目标是所有特征的线性组合加上噪声）
target = 7 * feature1 + 3 * feature2 +noise

# 归一化目标变量到 0 到 1 之间
target_min = target.min()
target_max = target.max()
target  = (target - target_min) / (target_max - target_min)

# 创建DataFrame
data = pd.DataFrame({
    'feature1': feature1,
    'feature2': feature2,
    'feature3': feature3,
    'feature4': feature4,
    'feature5': feature5,
    'target': target
})

# 保存为CSV文件
data.to_csv('regression_data.csv', index=False)

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 读取数据
data = pd.read_csv('regression_data.csv')

# 分割特征和目标变量
X = data[['feature1', 'feature2', 'feature3', 'feature4', 'feature5']]
y = data['target']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建神经网络回归模型
model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 在测试集上预测
y_pred = model.predict(X_test)

# 计算均方误差和 R² 得分
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R^2 Score:", r2)
