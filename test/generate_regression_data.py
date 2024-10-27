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
noise = np.random.randn(num_samples) * 10      # 随机噪声

# 生成目标变量（假设目标是所有特征的线性组合加上噪声）
target = 7 * feature1 + 3 * feature2 + 5 * feature3 + 2 * feature4 + 4 * feature5

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

