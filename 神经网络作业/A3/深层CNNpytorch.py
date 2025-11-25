import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.optim import lr_scheduler
import random
def set_seed(seed):
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

set_seed(42)
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_datasets(data_dir):
    train_data = []
    train_labels = []

    for i in range(1, 6):  
        with open(f'{data_dir}/data_batch_{i}', 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
        train_data.append(batch[b'data'])
        train_labels.append(batch[b'labels'])

    x_train = np.vstack(train_data)
    y_train = np.concatenate(train_labels)

    test_batch = unpickle(f'{data_dir}/test_batch')
    x_test = test_batch[b'data']
    y_test = np.array(test_batch[b'labels'])

    return (x_train, y_train), (x_test, y_test)

train_set, test_set = load_cifar10_datasets('cifar-10-batches-py/')
train_x, train_y = train_set
test_x, test_y = test_set

train_x = train_x / 255.0 
test_x = test_x / 255.0

train_x = torch.tensor(train_x, dtype=torch.float32).view(-1, 3, 32, 32)  
train_y = torch.tensor(train_y, dtype=torch.long)

test_x = torch.tensor(test_x, dtype=torch.float32).view(-1, 3, 32, 32)
test_y = torch.tensor(test_y, dtype=torch.long)

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)
import torch
from torch.utils.data import DataLoader

def test(model, test_dataset, epoch,batch_size=1000, device='cuda'):
    """
    测试函数，计算准确率以及每个类的分类正确率和预测次数

    参数:
    - model: 已训练的模型
    - test_dataset: 测试数据集 (TensorDataset)
    - batch_size: 批大小
    - device: 设备 ('cuda' 或 'cpu')

    输出:
    - 打印整体准确率和每个类的分类正确率以及每个类的预测次数
    """
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化统计变量
    correct = 0
    total = 0
    class_correct = [0] * 10  # 假设有10个类别
    class_total = [0] * 10
    class_predictions = [0] * 10  # 用于存储每个类别的预测次数

    # 不计算梯度，减少内存消耗
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 通过模型进行前向传播
            outputs = model(inputs)

            # 获取最大概率的类别
            _, predicted = torch.max(outputs, 1)

            # 统计整体准确率
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 统计每个类别的准确率和预测次数
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

                # 增加预测次数
                class_predictions[predicted[i]] += 1

    # 计算整体准确率
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    # 输出每个类别的准确率和预测次数
    for i in range(10):
        class_accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"Class {i} Accuracy: {class_accuracy:.2f}%")
        print(f"Predicted Class {i} Count: {class_predictions[i]}")
    return accuracy

class CNNModel(nn.Module):
    def __init__(self, use_dropout, normalization_method, use_l2_reg):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  
        self.pool = nn.MaxPool2d(2, 2) 
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        
        self.fc1 = nn.Linear(64 * 8 * 8, 128) 
        self.fc2 = nn.Linear(128, 10)  

        self.normalization_method = normalization_method
        if self.normalization_method == 'batchnorm':
            self.bn1 = nn.BatchNorm2d(32) 
            self.bn2 = nn.BatchNorm2d(64)  
        elif self.normalization_method == 'layernorm':
            self.ln1 = nn.LayerNorm([32, 32, 32]) 
            self.ln2 = nn.LayerNorm([64, 16, 16]) 
        else:
            pass
        
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(0.8)  
        
        self.use_l2_reg = use_l2_reg

    def forward(self, x):
        x = self.conv1(x)
        
        if self.normalization_method == 'batchnorm':
            x = self.bn1(x)
        elif self.normalization_method == 'layernorm':
            x = self.ln1(x)
        
        x = torch.relu(x)  
        x = self.pool(x)  
        
        x = self.conv2(x)
        
        if self.normalization_method == 'batchnorm':
            x = self.bn2(x)
        elif self.normalization_method == 'layernorm':
            x = self.ln2(x)
        
        x = torch.relu(x)  
        x = self.pool(x)  
        
        x = x.view(-1, 64 * 8 * 8) 
        
        x = self.fc1(x)
        
        if self.use_dropout:
            x = self.dropout(x)  
        
        x = self.fc2(x)
        
        return x

def train_model(model, train_loader, criterion, optimizer, device, num_epochs, test_dataset):
    model.to(device)
    model.train()
    
    train_loss = []
    train_accuracy = []
    true_accuracy = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        class_correct = [0] * 10
        class_total = [0] * 10
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i]
                class_total[label] += 1
                class_correct[label] += (predicted[i] == label).item()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        train_loss.append(avg_loss)
        train_accuracy.append(accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        if epoch+1 == 15:
            torch.save(model.state_dict(), 'basemodel_paramsYnorYdropYregYcroYlay.pth')
        if epoch+1 == 16:
            torch.save(model.state_dict(), '2model_paramsYnorYdropYregYcroYlay.pth')
        if epoch+1 == 17:
            torch.save(model.state_dict(), '3model_paramsYnorYdropYregYcroYlay.pth')    
        true_accuracy.append(test(model, test_dataset, epoch,batch_size=64, device=device))

    return train_loss, train_accuracy, true_accuracy

def plot_metrics(train_loss, train_accuracy, true_accuracy,num_epochs):
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracy, label='Train Accuracy', color='green')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, true_accuracy, label='True Accuracy', color='red')
    plt.title('true_accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Train Loss', color='blue')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

def cross_validate(model_class, num_folds=5, **kwargs):
    kf = KFold(n_splits=num_folds)
    accuracies = []  # 用来存储每一折的验证集准确率

    for train_idx, val_idx in kf.split(train_x):
        train_fold = torch.utils.data.Subset(train_dataset, train_idx)
        val_fold = torch.utils.data.Subset(train_dataset, val_idx)
        
        train_loader = DataLoader(train_fold, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_fold, batch_size=64, shuffle=False)
        
        model = model_class(**kwargs)
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        train_loss, train_accuracy, true_accuracy = train_model(model, train_loader, criterion, optimizer, device, 1, val_fold)
        
        # 收集每一折的验证集准确率（true_accuracy）
        accuracies.append(true_accuracy)

    # 计算交叉验证的平均准确率和标准差
    mean_accuracy = np.mean(accuracies)  # 计算平均准确率
    std_accuracy = np.std(accuracies)    # 计算标准差
    
    print(f"Cross-validation Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

use_dropout = True 
normalization_method = 'batchnorm'  
use_l2_reg = True  

model = CNNModel(use_dropout=use_dropout, normalization_method=normalization_method, use_l2_reg=use_l2_reg)
num_epochs = 20
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

train_loss, train_accuracy,true_accuracy = train_model(model, train_loader, criterion, optimizer, device, num_epochs,test_dataset)

plot_metrics(train_loss, train_accuracy, true_accuracy,num_epochs)

cross_validate(CNNModel, num_folds=5, use_dropout=use_dropout, normalization_method=normalization_method, use_l2_reg=use_l2_reg)