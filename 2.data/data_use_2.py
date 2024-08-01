import numpy as np

# 创建一个包含8个元素的数组
arr = np.arange(8)

# # 尝试使用 reshape 将其形状改为 3x3
# new_arr = arr.reshape(3, 3)
# print(new_arr)

arr1 = arr
print(id(arr1) == id(arr))