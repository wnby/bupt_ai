from EasyRnn_model import NameGenerationRNNModel2
import string
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from load_data import load,load_JP,load_CN,load_TR
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np
import time
# 设置随机种子，确保结果可复现
random.seed(421)
np.random.seed(421)
torch.manual_seed(421)
# 创建一个字符映射字典，包括小写字母、大写字母、空格和结束符 '<'
special_end_symbol = '<'
char_to_index = {char: idx for idx, char in enumerate(string.ascii_lowercase + string.ascii_uppercase + ' ' + special_end_symbol)}
index_to_char = {idx: char for char, idx in char_to_index.items()}
input_size = len(char_to_index)  # 26小写 + 26大写 + 1空格 + 1结束符 = 54

# 2. 自定义 Dataset 类
class NameDataset(Dataset):
    def __init__(self):
        # 假设 load() 返回字典形式的数据，每个字典包含 'number' 和 'name' 键
        self.data = load_TR()  

    def __len__(self):
        # 返回数据的长度
        return len(self.data)

    def __getitem__(self, idx):
        # 获取每一行数据
        sample = self.data[idx]
        
        # 获取 "name" 字符串，并转成小写
        name = sample['name'].lower()
        
        # 过滤掉不在 char_to_index 的字符，并确保不包含特殊结束符
        filtered_name = ''.join([c for c in name if c in char_to_index and c != special_end_symbol])
        
        # 将字符串转换为数字（字符到索引），并添加结束符 '<'
        filtered_name += special_end_symbol
        name_tensor = torch.tensor([char_to_index[c] for c in filtered_name], dtype=torch.long)
        
        # 确保所有字符都在字典中
        for c in filtered_name:
            assert c in char_to_index, f"Character '{c}' not in char_to_index"
        
        return {'number': sample['number'], 'name': name_tensor}
def collate_fn(batch):
    # 获取名字张量（可能包含不同长度的名字）
    name_tensors = [item['name'] for item in batch]
    
    # 使用 pad_sequence 来对名字进行填充，使它们都具有相同的长度
    padded_names = pad_sequence(name_tensors, batch_first=True, padding_value=char_to_index[special_end_symbol])
    
    # 创建输入和目标张量
    # 输入: 所有字符 except the last one
    # Target: 所有 characters except the first one
    input_tensor = padded_names[:, :-1]  # (batch_size, seq_len -1)
    target_tensor = padded_names[:, 1:]  # (batch_size, seq_len -1)
    # def tensor_to_string(tensor):
    #     # 使用 .tolist() 将张量转换为列表，然后映射每个索引为对应字符
    #     return ''.join([index_to_char[idx] for idx in tensor.tolist()])
    # print("Input Tensor as Characters:")
    # for seq in input_tensor:
    #     print(tensor_to_string(seq))

    # print("Target Tensor as Characters:")
    # for seq in target_tensor:
    #     print(tensor_to_string(seq))

    # time.sleep(1)
    # Convert input_tensor to one-hot encoding
    batch_size, seq_len = input_tensor.size()
    one_hot_input = torch.zeros(batch_size, seq_len, input_size, dtype=torch.float32)
    one_hot_input.scatter_(2, input_tensor.unsqueeze(2), 1)  # scatter on the third dimension
    
    # Move tensors to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    one_hot_input = one_hot_input.to(device)
    target_tensor = target_tensor.to(device)
    return {'input': one_hot_input, 'target': target_tensor}

# 4. 创建 Dataset 对象
dataset = NameDataset()

# 5. 使用 DataLoader 加载数据，结合 tqdm 显示进度条
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
def train():
    # 定义模型参数
    hidden_size = 128
    output_size = 54

    # 实例化模型
    model = NameGenerationRNNModel2(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 初始化损失和准确率记录列表
    loss_history = []
    accuracy_history = []

    # 假设 dataloader 已经定义好，例如：
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # 训练循环
    num_epochs = 50  

    for epoch in range(num_epochs):
        model.train()  # 设置为训练模式
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch in tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}"):
            # 获取批次中的输入数据和目标
            input_batch = batch['input'].to(device)  # (batch_size, seq_len -1, input_size)
            target_batch = batch['target'].to(device)  # (batch_size, seq_len -1)

            batch_size = input_batch.size(0)

            # 初始化隐藏状态
            hidden = model.initHidden(batch_size)

            # 预测输出
            output, hidden = model(input_batch, hidden)  # (batch_size, seq_len -1, output_size)

            # 计算损失
            output = output.view(-1, output_size)  # (batch_size * (seq_len -1), output_size)
            target = target_batch.view(-1)  # (batch_size * (seq_len -1))
            loss = criterion(output, target)

            # 更新梯度并优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失值
            running_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == target).sum().item()
            total_predictions += target.size(0)

        # 记录每个epoch的损失和准确率
        avg_loss = running_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    torch.save(model.state_dict(), r'C:\Users\49452\Desktop\神经网络 作业\刘逸-2022212054-A7\name_generation_rnn2_TR_doublenamekongge.pth')
    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history, label='Training Accuracy', color='green')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig(r'C:\Users\49452\Desktop\神经网络 作业\刘逸-2022212054-A7\name_generation_rnn2_TR_doublenamekongge.png')
    plt.show()
train()