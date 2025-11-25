# generate_caption.py

import torch
import torch.nn as nn
from encoder import VisionTransformerEncoder
from decoder import TransformerDecoder
from dataset import create_dataset
from PIL import Image
import torchvision.transforms as transforms
import os
import torch.nn.functional as F
class Config:
    max_len = 196  # 根据 ViT 的 num_patches 设置，假设为 196
    embed_dim = 768  # 应与 ViT 输出维度匹配
    decoder_num_heads = 8
    decoder_num_layers = 4
    decoder_forward_expansion = 4
    decoder_dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder_pth_path = r'decoder_epoch10.pth'  # 替换为您的解码器 .pth 文件路径
    encoder_pth_path = r'encoder_epoch10.pth'  # 替换为您训练好的编码器 .pth 文件路径


def tokens_to_sentence(tokens, vocab):
    """
    将令牌ID转换为句子。
    tokens: [seq_len]
    vocab: 字典 {word: id, ...}
    """
    inverse_vocab = {id_: word for word, id_ in vocab.items()}
    words = []
    for token in tokens:
        word = inverse_vocab.get(token, '<UNK>')  # Remove .item()
        if word == '<END>':
            break
        if word not in ['<START>', '<PAD>']:
            words.append(word)
    sentence = ' '.join(words)
    return sentence


def generate_caption(image_path, encoder, decoder, vocab, transform, max_generation_length=50, beam_size=3, repeat_threshold=2):
    """
    生成图像的标题。
    image_path: str, 图像路径
    encoder: 编码器模型
    decoder: 解码器模型
    vocab: 词汇表字典
    transform: 图像预处理
    config: 配置（包含设备信息）
    max_generation_length: 最大生成长度
    beam_size: Beam size
    repeat_threshold: 重复词频阈值
    """
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        # 读取并处理图像
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(config.device)  # [1, 3, H, W]

        memory = encoder(image)  # 编码器输出 [1, num_patches, embed_dim]

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

def make_trg_mask(trg_input):
    batch_size, sz = trg_input.size()
    mask = torch.tril(torch.ones(sz, sz, device=trg_input.device)).bool()
    mask = mask.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)  # [batch_size, 1, sz, sz]
    return mask

if __name__ == "__main__":
    config = Config()
    vocab = {'the': 1, 'lower': 2, 'clothing': 3, 'is': 4, 'of': 5, 'long': 6, 'length.': 7, 'fabric': 8, 'cotton': 9, 'and': 10, 'it': 11, 'has': 12, 'plaid': 13, 'patterns.': 14, 'his': 15, 'tank': 16, 'top': 17, 'sleeves': 18, 'cut': 19, 'off,': 20, 'pure': 21, 'color': 22, 'neckline': 23, 'round.': 24, 'pants': 25, 'this': 26, 'man': 27, 'wears': 28, 'are': 29, 'with': 30, 'sweater': 31, 'sleeves,': 32, 'stripe': 33, 'lapel.': 34, 'gentleman': 35, 'a': 36, 'pants.': 37, 'solid': 38, 'shirt': 39, 'short': 40, 'crew': 41, 'neckline.': 42, 'person': 43, 'its': 44, 'denim,': 45, 'lapel': 46, 'pants,': 47, 'short-sleeve': 48, 't-shirt': 49, 'patterns': 50, 'round': 51, 'wearing': 52, 'long-sleeve': 53, 'outer': 54, 'denim': 55, 'upper': 56, 'there': 57, 'an': 58, 'accessory': 59, 'on': 60, 'wrist.': 61, 'fabric.': 62, 'square': 63, 'guy': 64, 'trousers.': 65, 'trousers': 66, 'medium': 67, 'medium-sleeve': 68, 'v-shape.': 69, 'hat': 70, 'in': 71, 'head.': 72, 'shorts': 73, 'shorts.': 74, 'lattice': 75, 'also': 76, 'clothing,': 77, 'waist.': 78, 'sleeveless': 79, 'cotton.': 80, 'pattern': 81, 'color.': 82, 'v-shape': 83, 'trousers,': 84, 'cotton,': 85, 'crew.': 86, 'complicated': 87, 'length': 88, 'block': 89, 'striped': 90, 'leather': 91, 'stand': 92, 'stand.': 93, 'belt.': 94, 'hat.': 95, 'sunglasses.': 96, 'knitting': 97, 'graphic': 98, 'glasses': 99, 'hands': 100, 'or': 101, 'clothes.': 102, 'pair': 103, 'other': 104, 'ring': 105, 'finger.': 106, 'mixed': 107, 'other.': 108, 'mixed,': 109, 'neckwear.': 110, 'neck.': 111, 'other,': 112, 'ring.': 113, 'belt': 114, 'mixed.': 115, 'complicated.': 116, 'no': 117, 'leather,': 118, 'stripe.': 119, 'leather.': 120, 'three-quarter': 121, 'denim.': 122, 'socks.': 123, 'plaid.': 124, 'lattice.': 125, 'socks': 126, 'shoes.': 127, 'floral': 128, 'block.': 129, 'off': 130, 'graphic.': 131, 'three-point': 132, 'shorts,': 133, 'striped.': 134, 'square.': 135, 'floral.': 136, 'lady': 137, 'chiffon': 138, 'her': 139, 'female': 140, 'woman': 141, 'chiffon,': 142, 'chiffon.': 143, 'suspenders.': 144, 'suspenders': 145, 'leggings.': 146, 'furry': 147, 'knitting,': 148, 'knitting.': 149, 'furry.': 150, 'furry,': 151, 'eyeglasses.': 152, 'glasses.': 153, 'skirt.': 154, 'skirt': 155, 'skirt,': 156, '<pad>': 0, '': 157, '<start>': 158, '<end>': 159}

    # 初始化编码器
    encoder = VisionTransformerEncoder(
        img_size=224,
        patch_size=16,
        in_c=3,
        embed_dim=config.embed_dim,
        depth=4,
        num_heads=config.decoder_num_heads,
        mlp_ratio=config.decoder_forward_expansion,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=config.decoder_dropout,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm
    ).to(config.device)

    # 加载编码器权重
    encoder.load_state_dict(torch.load(config.encoder_pth_path, map_location=config.device))
    encoder.eval()

    # 初始化解码器并加载权重
    decoder = TransformerDecoder(
        vocab_size=len(vocab),
        embed_dim=config.embed_dim,
        num_heads=config.decoder_num_heads,
        num_layers=config.decoder_num_layers,
        forward_expansion=config.decoder_forward_expansion,
        dropout=config.decoder_dropout,
        max_len=config.max_len + 2  # 考虑 <START> 和 <END> 令牌
    ).to(config.device)

    decoder.load_state_dict(torch.load(config.decoder_pth_path, map_location=config.device))
    decoder.eval()

    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 示例生成
    image_path = os.path.join('images', 'WOMEN-Rompers_Jumpsuits-id_00007408-02_2_side.jpg')  # 替换为您要生成标题的图像路径
    caption = generate_caption(image_path, encoder, decoder, vocab, transform)
    print(f"Generated Caption: {caption}")
