import torch.nn as nn
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 6. 定义 RNN 模型
class NameGenerationRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NameGenerationRNNModel, self).__init__()
        self.input_size = input_size  # 输入大小（字符集的大小）
        self.hidden_size = hidden_size  # 隐藏层大小
        self.output_size = output_size  # 输出大小（字符集的大小）
        
        # 定义 RNN 层
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        
        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Note: Removed Softmax because CrossEntropyLoss applies it internally

    def forward(self, x):
        out, _ = self.rnn(x)  # out: (batch_size, seq_len, hidden_size)
        out = self.fc(out)    # (batch_size, seq_len, output_size)
        return out            
class NameGenerationRNNModel2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NameGenerationRNNModel2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 定义 RNN 层
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

        # 定义全连接层，用于生成输出
        self.fc = nn.Linear(hidden_size, output_size)

        # Dropout 层用于防止过拟合
        self.dropout = nn.Dropout(0.1)

        # softmax 层用于输出概率分布
        self.softmax = nn.LogSoftmax(dim=2)  # 应用于 output_size 维度

    def forward(self, input, hidden):
        # RNN 层前向传播
        rnn_out, hidden = self.rnn(input, hidden)

        # 全连接层生成输出
        output = self.fc(rnn_out)

        # 应用 dropout 防止过拟合
        output = self.dropout(output)

        # 使用 softmax 得到每个字符的概率分布
        output = self.softmax(output)

        return output, hidden

    def initHidden(self, batch_size):
        # 初始化隐藏状态
        return torch.zeros(1, batch_size, self.hidden_size).to(device)