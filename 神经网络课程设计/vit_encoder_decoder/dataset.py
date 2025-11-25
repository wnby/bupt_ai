import os
import json
import random
from collections import defaultdict, Counter
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def create_dataset(config):
    """处理数据集,生成词典和训练数据"""
    
    # 读取captions.json（直接从当前目录读取）
    with open('captions.json', 'r') as f:
        captions_dict = json.load(f)
        
    # 统计词频
    vocab = Counter()
    image_captions = defaultdict(list)
    
    for img_name, caption in captions_dict.items():
        # 分词
        tokens = caption.lower().split()
        if len(tokens) <= config.max_len:
            vocab.update(tokens)
            image_captions[img_name].append(tokens)
            
    # 构建词典
    words = [w for w in vocab.keys() if vocab[w] >=  config.min_word_count]
    vocab = {k: v + 1 for v, k in enumerate(words)}
    vocab['<pad>'] = 0
    vocab[''] = len(vocab)
    vocab['<start>'] = len(vocab)
    vocab['<end>'] = len(vocab)
    
    # 保存词典
    os.makedirs('processed', exist_ok=True)
    with open(os.path.join('processed', 'vocab.json'), 'w') as f:
        json.dump(vocab, f)
        
    # 划分数据集
    all_images = list(image_captions.keys())
    random.shuffle(all_images)
    
    train_size = int(0.7 * len(all_images))
    val_size = int(0.15 * len(all_images))
    
    splits = {
        'train': all_images[:train_size],
        'val': all_images[train_size:train_size+val_size],
        'test': all_images[train_size+val_size:]
    }
    
    # 保存数据集
    for split, images in splits.items():
        data = {
            'IMAGES': [],
            'CAPTIONS': []
        }
        
        for img in images:
            # 修改图片路径，确保使用正确的目录
            img_path = os.path.join('images', img)  # 使用相对路径
            captions = image_captions[img]
            
            # 检查文件是否存在
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
                
            # 编码caption
            enc_captions = []
            for c in captions:
                enc_c = [vocab['<start>']] + [vocab.get(w, vocab['']) for w in c] + [vocab['<end>']]
                enc_captions.append(enc_c)
                
            data['IMAGES'].extend([img_path] * len(enc_captions))
            data['CAPTIONS'].extend(enc_captions)
            
        with open(os.path.join('processed', f'{split}_data.json'), 'w') as f:
            json.dump(data, f)
            
    return vocab

class ImageTextDataset(Dataset):
    def __init__(self, data_path, vocab_path, split, max_len=30, transform=None):
        self.split = split
        self.max_len = max_len
        
        # 加载数据
        with open(data_path, 'r') as f:
            self.data = json.load(f)
            
        # 加载词典
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
            
        self.transform = transform
        self.dataset_size = len(self.data['CAPTIONS'])
        
    def __getitem__(self, i):
        img = Image.open(self.data['IMAGES'][i]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            
        caption = self.data['CAPTIONS'][i]
        caplen = len(caption)
        
        caption = torch.tensor(
            caption + [self.vocab['<pad>']] * (self.max_len + 2 - caplen),
            dtype=torch.long
        )
        
        return img, caption, caplen
        
    def __len__(self):
        return self.dataset_size

def get_data_loaders(config):
    """获取数据加载器"""
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_set = ImageTextDataset(
        os.path.join('processed', 'train_data.json'),
        os.path.join('processed', 'vocab.json'),
        'train',
        config.max_len,
        train_transform
    )
    
    val_set = ImageTextDataset(
        os.path.join('processed', 'val_data.json'),
        os.path.join('processed', 'vocab.json'), 
        'val',
        config.max_len,
        val_transform
    )
    
    test_set = ImageTextDataset(
        os.path.join('processed', 'test_data.json'),
        os.path.join('processed', 'vocab.json'),
        'test', 
        config.max_len,
        val_transform
    )
    
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader
