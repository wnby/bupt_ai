import random
import torch
import string
import matplotlib.pyplot as plt
import numpy as np
from EasyRnn_model import NameGenerationRNNModel2
import seaborn as sns
import pandas as pd

# 创建字符映射字典
special_end_symbol = '<'
char_to_index = {char: idx for idx, char in enumerate(string.ascii_lowercase + string.ascii_uppercase + ' ' + special_end_symbol)}
index_to_char = {idx: char for char, idx in char_to_index.items()}
input_size = len(char_to_index)  # 26小写 + 26大写 + 1空格 + 1结束符 = 54
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_name(model, start_char, max_length=20):
    model.eval()  # 设置为评估模式
    hidden = model.initHidden(1)
    # 将起始字符转换为 one-hot 编码
    input_tensor = torch.zeros((1, 1, len(char_to_index)), dtype=torch.float32)
    if start_char in char_to_index:
        input_tensor[0, 0, char_to_index[start_char]] = 1
    else:
        input_tensor[0, 0, char_to_index[' ']] = 1  # 如果起始字符不在字典中，使用空格
    
    # 移动到设备
    input_tensor = input_tensor.to(device)
    
    generated_name = start_char  # 初始化生成的名字
    count = 0
    prediction_history = []  # 用于存储每个时间步的前5个预测
    
    with torch.no_grad():
        for _ in range(max_length - 1):

            output, hidden = model(input_tensor, hidden)  # 模型输出 (1, 1, output_size)
            output = output.view(-1, len(char_to_index))  # 调整维度为 (1, output_size)
            probabilities = torch.exp(output)  # 使用 exp 将 log probabilities 转换为 probabilities
            
            # 获取前5个最可能的字符及其概率
            top_k = 5
            top_k_probabilities, top_k_indices = torch.topk(probabilities, top_k)
            top_k_probabilities = top_k_probabilities.cpu().numpy().flatten()
            top_k_indices = top_k_indices.cpu().numpy().flatten()
            top_k_chars = [index_to_char[idx] for idx in top_k_indices]
            
            # 存储当前时间步的预测
            prediction_history.append((top_k_chars, top_k_probabilities))
            
            # 选择最有可能的字符
            chosen_index = top_k_indices[0].item()
            chosen_char = index_to_char[chosen_index]
            
            # 将选中的字符添加到生成的名字中
            generated_name += chosen_char
            
            # 更新输入字符的one-hot编码
            input_tensor = torch.zeros((1, 1, len(char_to_index)), dtype=torch.float32)
            input_tensor[0, 0, chosen_index] = 1
            input_tensor = input_tensor.to(device)
            
            # 如果生成结束符 '<'，停止生成
            if chosen_char == special_end_symbol:
                break
    
    return generated_name, prediction_history

def visualize_predictions(prediction_history, generated_name, save_path=r'C:\Users\49452\Desktop\神经网络 作业\刘逸-2022212054-A7\Heatmap_of_easy_Rnn_TR.png'):
    time_steps = len(prediction_history)
    top_k = len(prediction_history[0][0]) if time_steps > 0 else 0

    # 准备数据
    chars = [prediction_history[t][0] for t in range(time_steps)]
    probs = [prediction_history[t][1] for t in range(time_steps)]

    # 创建一个 DataFrame 适合 Seaborn 绘图
    data = []
    for t in range(time_steps):
        for k in range(top_k):
            data.append({
                'Time Step': t+1,
                'Rank': k+1,
                'Character': chars[t][k],
                'Probability': probs[t][k]
            })

    df = pd.DataFrame(data)

    # 创建注释列，将字符和概率合并
    df['Char_Prob'] = df.apply(lambda row: f"{row['Character']}: {row['Probability']:.2f}", axis=1)

    # 创建一个透视表，行是时间步，列是Rank，值是Char_Prob
    pivot_table_chars = df.pivot(index='Time Step', columns='Rank', values='Char_Prob')
    pivot_table_probs = df.pivot(index='Time Step', columns='Rank', values='Probability')

    # 绘制热力图
    plt.figure(figsize=(12, max(8, time_steps * 0.3)))  # 动态调整高度以适应时间步
    sns.heatmap(pivot_table_probs, annot=pivot_table_chars, fmt="", cmap='viridis', cbar_kws={'label': 'Probability'})
    plt.title(f"Top {top_k} Candidate Characters Probabilities Over Time")
    plt.xlabel("Top K Rank")
    plt.ylabel("Time Step")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    # 输出生成的名字
    print(f"Generated name: {generated_name}")

# 创建模型实例
model = NameGenerationRNNModel2(input_size=input_size, hidden_size=128, output_size=input_size)

# 加载模型权重       
model.load_state_dict(torch.load(r'C:\Users\49452\Desktop\神经网络 作业\刘逸-2022212054-A7\name_generation_rnn2_TR_doublenamekongge.pth', map_location=device))
model.eval()  # 设置模型为评估模式

# 移动到适当的设备（如果有 GPU）
model.to(device)

# 输入的起始字符
start_char = 'F'

# 生成名字并获取预测历史
generated_name, prediction_history = generate_name(model, start_char, max_length=60)

# 可视化预测
visualize_predictions(prediction_history, generated_name)