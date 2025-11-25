import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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

train_x = train_x.astype(np.float32) / 255.0
test_x = test_x.astype(np.float32) / 255.0

train_y = train_y.astype(np.int64)
test_y = test_y.astype(np.int64)

train_x = train_x.reshape(-1, 3, 32, 32)
test_x = test_x.reshape(-1, 3, 32, 32)

train_x_tensor = torch.tensor(train_x)
train_y_tensor = torch.tensor(train_y)
test_x_tensor = torch.tensor(test_x)
test_y_tensor = torch.tensor(test_y)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1_weight = nn.Parameter(torch.randn(32, 3, 3, 3))
        self.fc1 = nn.Linear(32 * 16 * 16, 10)

    def conv2d_numpy(self, input, filters, stride=1, padding=1):
        batch_size, in_channels, in_height, in_width = input.shape
        out_channels, _, filter_height, filter_width = filters.shape
        input_padded = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
        out_height = (in_height + 2 * padding - filter_height) // stride + 1
        out_width = (in_width + 2 * padding - filter_width) // stride + 1
        output = np.zeros((batch_size, out_channels, out_height, out_width))

        for b in range(batch_size):
            for c_out in range(out_channels):
                for c_in in range(in_channels):
                    for i in range(out_height):
                        for j in range(out_width):
                            input_slice = input_padded[b, c_in, i * stride:i * stride + filter_height, j * stride:j * stride + filter_width]
                            output[b, c_out, i, j] += np.sum(input_slice * filters[c_out, c_in])

        return output

    def forward(self, x):
        x = x.numpy()
        x = self.conv2d_numpy(x, self.conv1_weight.detach().numpy(), stride=1, padding=1)
        x = np.maximum(x, 0)
        batch_size, channels, height, width = x.shape
        pool_size = 2
        out_height = height // pool_size
        out_width = width // pool_size
        pooled = np.zeros((batch_size, channels, out_height, out_width))

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        pooled[b, c, i, j] = np.max(x[b, c, i*2:i*2+2, j*2:j*2+2])

        pooled = pooled.reshape(batch_size, -1)
        x = torch.tensor(pooled, dtype=torch.float32)
        x = self.fc1(x)
        return x

model = CNNModel()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_x_tensor, train_y_tensor, criterion, optimizer, epochs=10):
    model.train()
    total_samples = len(train_x_tensor)
    batch_size = 64
    
    for epoch in range(epochs):
        indices = np.random.choice(total_samples, size=500, replace=False)
        sampled_train_x = train_x_tensor[indices]
        sampled_train_y = train_y_tensor[indices]
        
        sampled_dataset = TensorDataset(sampled_train_x, sampled_train_y)
        train_loader = DataLoader(sampled_dataset, batch_size=batch_size, shuffle=True)
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                tepoch.set_postfix(loss=running_loss / total, accuracy=100 * correct / total)
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/total:.4f}, Accuracy: {100*correct/total:.2f}%")

train_model(model, train_x_tensor, train_y_tensor, criterion, optimizer, epochs=15)
