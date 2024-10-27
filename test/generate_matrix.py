import numpy as np
import pandas as pd
from scipy.linalg import lu

# 生成随机矩阵
rows = 7
cols = 7
random_matrix = np.random.rand(rows, cols)

# 转换为DataFrame并保存
df = pd.DataFrame(random_matrix)
df.to_csv('matrix.csv', index=False)

# 将DataFrame转换为NumPy数组，以便进行矩阵运算
np_matrix = df.to_numpy()

# 计算行列式
determinant = np.linalg.det(np_matrix)
print("行列式为:", determinant)

# 判断矩阵是否可逆
if np.linalg.det(np_matrix) != 0:
    # 计算逆矩阵
    inverse_matrix = np.linalg.inv(np_matrix)
    print("逆矩阵为:\n", inverse_matrix)
else:
    print("该矩阵是奇异矩阵，不可逆")
    
 # 进行LUP分解
P, L, U = lu(np_matrix)

print("P:\n", P)
print("L:\n", L)
print("U:\n", U)   