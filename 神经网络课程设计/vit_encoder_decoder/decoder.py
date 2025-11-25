import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights

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
        trg = self.dropout((self.embedding(trg) + self.positional_encoding(positions)))  # [N, seq_len, embed_dim]

        for layer in self.decoder_layers:
            trg = layer(trg, memory, memory, trg_mask)  # [N, seq_len, embed_dim]

        out = self.fc_out(trg)  # [N, seq_len, vocab_size]
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
        N, seq_len, embed_dim = x.shape  # x: [32, 61, 768]

        # Self-attention layer
        attention = self.attention(values=x, key=x, query=x, mask=mask)  # Corrected
        #print(f'Self-Attention Output Shape: {attention.shape}')  # [32,8,61,96]

        # Reshape attention output
        attention = attention.transpose(1, 2).contiguous().view(N, seq_len, embed_dim)  # [32,61,768]
        #print(f'Attention Reshaped Shape: {attention.shape}')  # [32,61,768]

        # Add & Normalize
        query = self.dropout(self.norm1(attention + x))  # [32,61,768]
        #print(f'After Add & Norm (Self-Attention): {query.shape}')  # [32,61,768]

        # Print value and key shapes
        #print(f'Value shape: {value.shape}')  # [32,61,768]
        #print(f'Key shape: {key.shape}')      # [32,61,768]

        # Cross-attention layer
        cross_attention = self.cross_attention(values=value, key=key, query=query, mask=None)
        #print(f'Cross-Attention Output Shape: {cross_attention.shape}')  # [32,61,768]

        # Add & Normalize
        query = self.dropout(self.norm3(cross_attention + query))  # [32,61,768]
        #print(f'After Add & Norm (Cross-Attention): {query.shape}')  # [32,61,768]

        # Feedforward network
        forward = self.transformer_block(query)  # [32,61,768]
        #print(f'FeedForward Output Shape: {forward.shape}')  # [32,61,768]

        # Add & Normalize
        out = self.dropout(self.norm2(forward + query))  # [32,61,768]
        #print(f'After Add & Norm (FeedForward): {out.shape}')  # [32,61,768]

        return out  # [32,61,768]


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
        self.dropout = nn.Dropout(0.1)

    def forward(self, values, key, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], key.shape[1], query.shape[1]

        #print(f'Before Linear Transformation:')
        #print(f'Values shape: {values.shape}')  # Should be [32, 61, 768]
        #print(f'Keys shape: {key.shape}')        # Should be [32, 61, 768]
        #print(f'Queries shape: {query.shape}')    # Should be [32, 61, 768]

        # Linear transformations and reshape
        values = self.values(values).view(N, value_len, self.num_heads, self.head_dim).transpose(1, 2)  # [N, num_heads, value_len, head_dim]
        keys = self.keys(key).view(N, key_len, self.num_heads, self.head_dim).transpose(1, 2)          # [N, num_heads, key_len, head_dim]
        queries = self.queries(query).view(N, query_len, self.num_heads, self.head_dim).transpose(1, 2)  # [N, num_heads, query_len, head_dim]

        #print(f'After Linear Transformation and Reshape:')
        #print(f'Values shape: {values.shape}')    # Should be [32, 8, 61, 96]
        #print(f'Keys shape: {keys.shape}')        # Should be [32, 8, 61, 96]
        #print(f'Queries shape: {queries.shape}')  # Should be [32, 8, 61, 96]

        # Compute energy scores
        energy = torch.matmul(queries, keys.transpose(-2, -1))  # [N, num_heads, query_len, key_len]
        #print(f'Energy Shape: {energy.shape}')  # Should be [32, 8, 61, 61]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Softmax and dropout
        attention = torch.softmax(energy / (self.embed_dim ** 0.5), dim=-1)  # [N, num_heads, query_len, key_len]
        attention = self.dropout(attention)

        # Compute output
        out = torch.matmul(attention, values)  # [N, num_heads, query_len, head_dim]
        #print(f'Attention Output Shape: {out.shape}')  # Should be [32, 8, 61, 96]

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(N, query_len, self.embed_dim)  # [N, query_len, embed_dim]
        #print(f'After Transpose and View:')
        #print(f'Out shape: {out.shape}')  # Should be [32, 61, 768]

        # Final linear layer
        out = self.fc_out(out)  # [N, query_len, embed_dim]
        #print(f'Final Output Shape: {out.shape}')  # Should be [32, 61, 768]

        return out
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

class ViTEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ViTEncoder, self).__init__()
        if pretrained:
            weights = ViT_B_16_Weights.DEFAULT
        else:
            weights = None
        self.model = models.vit_b_16(weights=weights)
        # 移除分类头
        self.model.heads = nn.Identity()

        # 手动添加 LayerNorm 层
        self.norm = nn.LayerNorm(768)  # embed_dim = 768

    def forward(self, x):
        # 提取补丁嵌入
        x = self.model.conv_proj(x)  # [batch_size, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        
        # 添加分类令牌
        class_token = self.model.class_token.expand(x.size(0), -1, -1)  # [batch_size, 1, embed_dim]
        x = torch.cat((class_token, x), dim=1)  # [batch_size, 1 + num_patches, embed_dim]
        
        # 加上位置嵌入
        x = x + self.model.encoder.pos_embedding
        x = self.model.encoder.dropout(x)
        
        # 通过 transformer 编码器
        x = self.model.encoder(x)  # [batch_size, 1 + num_patches, embed_dim]
        
        # 应用手动添加的 LayerNorm
        x = self.norm(x)  # [batch_size, 1 + num_patches, embed_dim]
        
        # 移除分类令牌，只保留补丁嵌入
        x = x[:, 1:, :]  # [batch_size, num_patches, embed_dim]
        
        #print(f'ViTEncoder Output Shape: {x.shape}')  # 应为 [32, num_patches, 768]
        return x