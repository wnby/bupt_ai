import pandas as pd

# 读取 CSV 文件
df = pd.read_csv(r'C:\Users\49452\Desktop\神经网络 作业\刘逸-2022212054-A7\TR.csv', low_memory=False)

# 过滤条件：
# 1. last name 为空或包含非字母字符
# 2. first name 为空或包含非字母字符
import re
def is_english_name(name):
    return bool(re.match(r'^[A-Za-z]+$', str(name)))  

# 过滤掉 last name 为空或者不是英文
df_filtered = df[df.iloc[:, 1].notna() & df.iloc[:, 1].apply(is_english_name)]

# 过滤掉 first name 为空或者不是英文
df_filtered = df_filtered[df_filtered.iloc[:, 0].notna() & df_filtered.iloc[:, 0].apply(is_english_name)]

# 选择前 8000 行
df_top_8000 = df_filtered.head(8000)

# 输出前 8000 行数据（可选择打印或保存到新的 CSV 文件）
# print(df_top_8000)

# 如果需要保存为新的 CSV 文件
df_top_8000.to_csv(r'C:\Users\49452\Desktop\神经网络 作业\刘逸-2022212054-A7\TR_8000_M&F.csv', index=True)
