import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import random
from collections import defaultdict, Counter
from PIL import Image
from torch.utils.data import Dataset
from train_decoder import make_trg_mask
import torch.nn.functional as F
# Assuming dataset.py contains ImageTextDataset, get_data_loaders, create_dataset
from torchvision import transforms
import torchvision.models as models

# 1. PyTorch Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, forward_expansion, dropout, max_len):
        super(TransformerDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Embedding(max_len, embed_dim)

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, trg, memory, trg_mask):
        N, seq_len = trg.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(trg.device)
        trg = self.dropout((self.embedding(trg) + self.positional_encoding(positions)))

        for layer in self.decoder_layers:
            trg = layer(trg, memory, memory, trg_mask) # Query, Key, Value

        out = self.fc_out(trg)
        return out

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion, dropout):
        super(TransformerDecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.transformer_block = FeedForwardBlock(embed_dim, forward_expansion, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, value, key, mask):
        # 自注意力层
        attention = self.attention(x, x, x, mask)
        query = self.dropout(self.norm1(attention + x))

        # 交叉注意力层 - 不使用掩码
        # 注意：这里 query 来自解码器，key 和 value 来自编码器
        cross_attention = self.cross_attention(
            values=value,  # 来自编码器
            key=key,      # 来自编码器
            query=query,  # 来自解码器的查询
            mask=None     # 交叉注意力不需要掩码
        )
        query = self.dropout(self.norm3(cross_attention + query))

        # 前馈网络
        forward = self.transformer_block(query)
        out = self.dropout(self.norm2(forward + query))
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (self.head_dim * num_heads == embed_dim), "Embed dim needs to be divisible by heads"

        self.values = nn.Linear(embed_dim, embed_dim)
        self.keys = nn.Linear(embed_dim, embed_dim)
        self.queries = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0)

    def forward(self, values, key, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], key.shape[1], query.shape[1]

        # 线性变换
        values = self.values(values)  # [N, value_len, embed_dim]
        keys = self.keys(key)         # [N, key_len, embed_dim]
        queries = self.queries(query) # [N, query_len, embed_dim]

        # 重塑为多头形式
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)

        # 转置以便进行注意力计算
        values = values.transpose(1, 2)  # [N, num_heads, value_len, head_dim]
        keys = keys.transpose(1, 2)      # [N, num_heads, key_len, head_dim]
        queries = queries.transpose(1, 2) # [N, num_heads, query_len, head_dim]

        # 计算注意力分数
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # energy shape: [N, num_heads, query_len, key_len]

        if mask is not None:
            mask = mask.unsqueeze(1)  # [N, 1, seq_len, seq_len]
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)  # [N, num_heads, query_len, key_len]
        attention = self.dropout(attention)

        # 计算输出
        out = torch.matmul(attention, values)  # [N, num_heads, query_len, head_dim]
        out = out.transpose(1, 2)  # [N, query_len, num_heads, head_dim]
        out = out.reshape(N, query_len, self.embed_dim)  # [N, query_len, embed_dim]

        return self.fc_out(out)

class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim, forward_expansion, dropout):
        super(FeedForwardBlock, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.seq(x)

# 2. ViT Encoder
class ViTEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ViTEncoder, self).__init__()
        if pretrained:
            weights = models.ViT_B_16_Weights.DEFAULT
        else:
            weights = None
        self.model = models.vit_b_16(weights=weights)
        self.model.heads = nn.Identity()
        self.norm = nn.LayerNorm(768)

    def forward(self, x):
        x = self.model.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        
        class_token = self.model.class_token.expand(x.size(0), -1, -1)
        x = torch.cat((class_token, x), dim=1)
        
        x = x + self.model.encoder.pos_embedding
        x = self.model.encoder.dropout(x)
        
        x = self.model.encoder(x)
        x = self.norm(x)
        x = x[:, 1:, :]
        
        return x

# 3. Training Code
class Config:
    data_dir = '.'  # Replace with your data directory if needed
    max_len = 196  # 与训练时保持一致
    min_word_count = 2
    batch_size = 32
    learning_rate = 3e-4
    num_epochs = 1
    embed_dim = 768  
    decoder_num_heads = 8
    decoder_num_layers = 4
    decoder_forward_expansion = 4
    decoder_dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_caption(
    image, encoder, decoder, vocab, config, max_generation_length=100, beam_size=3, repeat_threshold=2
):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        memory = encoder(image.to(config.device))

        start_token_index = vocab.get("<start>")
        end_token_index = vocab.get("<end>")
        if start_token_index is None or end_token_index is None:
            raise ValueError("<start> or <end> token not found in vocabulary.")

        # Initialize the beam with the starting token
        beams = [( [start_token_index], 0.0, {} )]  # (sequence of token indices, log probability, word frequency)

        for _ in range(max_generation_length):
            new_beams = []
            reached_end = False  # Flag to check if <end> token is generated

            for seq, log_prob, word_freq in beams:
                if seq[-1] == end_token_index:  # If the current sequence already ended
                    new_beams.append((seq, log_prob, word_freq))
                    reached_end = True
                    continue

                current_trg_tensor = torch.tensor([seq]).to(config.device)
                trg_mask = make_trg_mask(current_trg_tensor)
                output = decoder(current_trg_tensor, memory, trg_mask)

                # Get probabilities of the next token
                probabilities = F.log_softmax(output[:, -1], dim=-1)  # Use log probabilities

                # Get the top k probabilities and their indices
                topk_probs, topk_indices = torch.topk(probabilities, beam_size)

                for i in range(beam_size):
                    next_token_index = topk_indices[0][i].item()
                    new_log_prob = log_prob + topk_probs[0][i].item()

                    # Update word frequency
                    new_word_freq = word_freq.copy()
                    word = {idx: w for w, idx in vocab.items()}.get(next_token_index, "")
                    if word and word not in {"the", "a", "and", "of", "is", "her", "on", "his", "with", "its", "there", "<start>", "<end>"}:  # Exclude common words
                        new_word_freq[word] = new_word_freq.get(word, 0) + 1

                    # Penalize if word frequency exceeds threshold
                    if new_word_freq.get(word, 0) > repeat_threshold:
                        new_log_prob -= 1.0  # Penalize repeated words

                    new_seq = seq + [next_token_index]
                    new_beams.append((new_seq, new_log_prob, new_word_freq))

            # Sort the new beams by probability and keep the top beam_size
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]

            # Stop if any sequence reaches <end>
            if reached_end:
                break

        # Select the beam with the highest probability
        best_beam = beams[0]

        # Convert the best beam's token indices to words
        inverse_vocab = {idx: word for word, idx in vocab.items()}
        predicted_sentence = [inverse_vocab.get(idx) for idx in best_beam[0] if idx not in [start_token_index, end_token_index]]

        return " ".join(predicted_sentence)






def load_and_preprocess_image(image_path, device):
    # 定义与训练时相同的图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载和预处理图片
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 添加批次维度
    return image.to(device)

if __name__ == "__main__":
    config = Config()
    print(dir(models))
    try:
        from dataset import ImageTextDataset, get_data_loaders, create_dataset
    except ImportError:
        print("Error: Make sure you have 'dataset.py' in the same directory with the necessary Dataset and DataLoader implementations.")
        exit()

    vocab = create_dataset(config)
    train_loader, val_loader, test_loader = get_data_loaders(config)

    encoder = ViTEncoder(pretrained=False).to(config.device) # Load with pretrained=False initially
    decoder = TransformerDecoder(
        vocab_size=len(vocab),
        embed_dim=config.embed_dim,
        num_heads=config.decoder_num_heads,
        num_layers=config.decoder_num_layers,
        forward_expansion=config.decoder_forward_expansion,
        dropout=config.decoder_dropout,
        max_len=config.max_len + 2 # Account for <start> and <end> tokens
    ).to(config.device)

    # Load the trained decoder weights
    try:
        decoder.load_state_dict(torch.load('decoder.pth', map_location=config.device))
        print("Loaded decoder weights from decoder.pth")
    except FileNotFoundError:
        print("Error: 'decoder.pth' not found. Make sure you have trained the model and saved the weights.")
        exit()

    # Since the encoder was frozen during training, we might not have saved its weights separately.
    # If you need to load specific encoder weights, you'll need to modify the training script.
    # For now, we'll use the default pretrained weights (if available) or the initialized weights.
    encoder = ViTEncoder(pretrained=True).to(config.device)
    for param in encoder.parameters():
        param.requires_grad = False # Freeze encoder for inference

    # 为特定图片生成描述
    image_path = os.path.join('images', 'WOMEN-Rompers_Jumpsuits-id_00007408-02_2_side.jpg')  # 使用相对路径
    image = load_and_preprocess_image(image_path, config.device)
    generated_caption = generate_caption(image, encoder, decoder, vocab, config, beam_size=3) # You can adjust beam_size
    print("Generated Caption (Beam Search):", generated_caption)