# train_encoder.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from dataset import create_dataset, get_data_loaders
from encoder import VisionTransformerEncoder
from decoder import TransformerDecoder  # 假设 decoder.py 中定义了 TransformerDecoder
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
decoder_pth_path = r'decoder.pth'  # 替换为您的解码器 .pth 文件路径
encoder_save_path = r'encoder.pth'  # 编码器保存路径
class Config:
    data_dir = '.'  # 替换为您的数据目录
    max_len = 196  # 根据 ViT 的 num_patches 设置，假设为 196
    min_word_count = 2
    batch_size = 32
    learning_rate = 3e-4
    num_epochs = 11
    embed_dim = 768  # 应与 ViT 输出维度匹配
    decoder_num_heads = 8
    decoder_num_layers = 4
    decoder_forward_expansion = 4
    decoder_dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_trg_mask(trg_input):
    batch_size, sz = trg_input.size()
    mask = torch.tril(torch.ones(sz, sz, device=trg_input.device)).bool()
    mask = mask.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)  # [batch_size, 1, sz, sz]
    return mask

def calculate_accuracy(output, target):
    """
    计算令牌级别的准确率。
    output: [batch_size * seq_len, vocab_size]
    target: [batch_size * seq_len]
    """
    _, preds = torch.max(output, dim=1)  # [batch_size * seq_len]
    correct = (preds == target).float()
    acc = correct.sum() / (target != 0).sum()  # 忽略填充的令牌
    return acc

def save_image_with_caption(image_tensor, caption, save_path):
    """
    将图像张量转换为图像并添加标题，然后保存。
    image_tensor: [3, H, W]
    caption: str
    save_path: str
    """
    image = image_tensor.cpu().clone().detach()
    image = image * 0.5 + 0.5  # 反归一化（假设使用了mean=0.5, std=0.5）
    image = image.numpy().transpose(1, 2, 0) * 255
    image = image.astype('uint8')
    image = Image.fromarray(image)

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=15)
    except IOError:
        font = ImageFont.load_default()
    text_position = (10, 10)
    text_color = (255, 0, 0)  # 红色
    draw.text(text_position, caption, fill=text_color, font=font)

    image.save(save_path)

def tokens_to_sentence(tokens, vocab):
    """
    将令牌ID转换为句子。
    tokens: [seq_len]
    vocab: 字典 {word: id, ...}
    """
    inverse_vocab = {id_: word for word, id_ in vocab.items()}
    words = [inverse_vocab.get(token.item(), '<UNK>') for token in tokens]
    sentence = ' '.join(words)
    return sentence

def train_encoder():
    config = Config()
    vocab = {'the': 1, 'lower': 2, 'clothing': 3, 'is': 4, 'of': 5, 'long': 6, 'length.': 7, 'fabric': 8, 'cotton': 9, 'and': 10, 'it': 11, 'has': 12, 'plaid': 13, 'patterns.': 14, 'his': 15, 'tank': 16, 'top': 17, 'sleeves': 18, 'cut': 19, 'off,': 20, 'pure': 21, 'color': 22, 'neckline': 23, 'round.': 24, 'pants': 25, 'this': 26, 'man': 27, 'wears': 28, 'are': 29, 'with': 30, 'sweater': 31, 'sleeves,': 32, 'stripe': 33, 'lapel.': 34, 'gentleman': 35, 'a': 36, 'pants.': 37, 'solid': 38, 'shirt': 39, 'short': 40, 'crew': 41, 'neckline.': 42, 'person': 43, 'its': 44, 'denim,': 45, 'lapel': 46, 'pants,': 47, 'short-sleeve': 48, 't-shirt': 49, 'patterns': 50, 'round': 51, 'wearing': 52, 'long-sleeve': 53, 'outer': 54, 'denim': 55, 'upper': 56, 'there': 57, 'an': 58, 'accessory': 59, 'on': 60, 'wrist.': 61, 'fabric.': 62, 'square': 63, 'guy': 64, 'trousers.': 65, 'trousers': 66, 'medium': 67, 'medium-sleeve': 68, 'v-shape.': 69, 'hat': 70, 'in': 71, 'head.': 72, 'shorts': 73, 'shorts.': 74, 'lattice': 75, 'also': 76, 'clothing,': 77, 'waist.': 78, 'sleeveless': 79, 'cotton.': 80, 'pattern': 81, 'color.': 82, 'v-shape': 83, 'trousers,': 84, 'cotton,': 85, 'crew.': 86, 'complicated': 87, 'length': 88, 'block': 89, 'striped': 90, 'leather': 91, 'stand': 92, 'stand.': 93, 'belt.': 94, 'hat.': 95, 'sunglasses.': 96, 'knitting': 97, 'graphic': 98, 'glasses': 99, 'hands': 100, 'or': 101, 'clothes.': 102, 'pair': 103, 'other': 104, 'ring': 105, 'finger.': 106, 'mixed': 107, 'other.': 108, 'mixed,': 109, 'neckwear.': 110, 'neck.': 111, 'other,': 112, 'ring.': 113, 'belt': 114, 'mixed.': 115, 'complicated.': 116, 'no': 117, 'leather,': 118, 'stripe.': 119, 'leather.': 120, 'three-quarter': 121, 'denim.': 122, 'socks.': 123, 'plaid.': 124, 'lattice.': 125, 'socks': 126, 'shoes.': 127, 'floral': 128, 'block.': 129, 'off': 130, 'graphic.': 131, 'three-point': 132, 'shorts,': 133, 'striped.': 134, 'square.': 135, 'floral.': 136, 'lady': 137, 'chiffon': 138, 'her': 139, 'female': 140, 'woman': 141, 'chiffon,': 142, 'chiffon.': 143, 'suspenders.': 144, 'suspenders': 145, 'leggings.': 146, 'furry': 147, 'knitting,': 148, 'knitting.': 149, 'furry.': 150, 'furry,': 151, 'eyeglasses.': 152, 'glasses.': 153, 'skirt.': 154, 'skirt': 155, 'skirt,': 156, '<pad>': 0, '': 157, '<start>': 158, '<end>': 159}

    train_loader, val_loader, test_loader = get_data_loaders(config)

    # 初始化编码器
    encoder = VisionTransformerEncoder(
        img_size=224,
        patch_size=16,
        in_c=3,
        embed_dim=config.embed_dim,
        depth=4,
        num_heads=config.decoder_num_heads,  # 保持与解码器一致
        mlp_ratio=config.decoder_forward_expansion,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=config.decoder_dropout,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm
    ).to(config.device)

    # 加载已有的编码器权重
    encoder.load_state_dict(torch.load(encoder_save_path, map_location=config.device))
    encoder.train()  # 保证编码器是训练模式

    # 初始化解码器并加载已有权重
    decoder = TransformerDecoder(
        vocab_size=len(vocab),
        embed_dim=config.embed_dim,
        num_heads=config.decoder_num_heads,
        num_layers=config.decoder_num_layers,
        forward_expansion=config.decoder_forward_expansion,
        dropout=config.decoder_dropout,
        max_len=config.max_len + 2  # 考虑 <start> 和 <end> 令牌
    ).to(config.device)

    decoder.load_state_dict(torch.load(decoder_pth_path, map_location=config.device, weights_only=True))
    decoder.train()  # 设置为评估模式
    for param in decoder.parameters():
        param.requires_grad = True  # 冻结解码器参数

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 假设 0 是填充索引
    optimizer = optim.Adam(encoder.parameters(), lr=config.learning_rate)

    # 创建保存图像的目录
    os.makedirs('saved_images_encoder', exist_ok=True)

    # 用于保存损失和准确率
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(config.num_epochs):
        encoder.train()
        total_loss = 0
        total_acc = 0

        loop = tqdm(train_loader, total=len(train_loader), desc=f'Epoch [{epoch+1}/{config.num_epochs}]', leave=False)

        for i, (images, captions, lengths) in enumerate(loop):
            images = images.to(config.device)
            captions = captions.to(config.device)

            # 前向传播
            memory = encoder(images)  # [batch_size, num_patches, embed_dim]

            trg_input = captions[:, :-1]  # [batch_size, max_len]
            trg_expected = captions[:, 1:]  # [batch_size, max_len]
            trg_mask = make_trg_mask(trg_input)  # [batch_size, 1, max_len, max_len]

            output = decoder(trg_input, memory, trg_mask)  # [batch_size, max_len, vocab_size]

            # 重塑以计算损失
            output = output.reshape(-1, output.shape[2])  # [batch_size * max_len, vocab_size]
            trg_expected = trg_expected.reshape(-1)       # [batch_size * max_len]

            loss = criterion(output, trg_expected)
            acc = calculate_accuracy(output, trg_expected)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()

            loop.set_postfix(loss=loss.item(), accuracy=f'{acc.item()*100:.2f}%')

        average_loss = total_loss / len(train_loader)
        average_acc = total_acc / len(train_loader)
        print(f'Epoch [{epoch+1}/{config.num_epochs}], Average Loss: {average_loss:.4f}, Average Accuracy: {average_acc*100:.2f}%')

        # 保存训练损失和准确率
        train_losses.append(average_loss)
        train_accuracies.append(average_acc)

        # 验证步骤
        encoder.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            val_loop = tqdm(val_loader, total=len(val_loader), desc=f'Validation [{epoch+1}/{config.num_epochs}]', leave=False)
            for images, captions, lengths in val_loop:
                images = images.to(config.device)
                captions = captions.to(config.device)

                memory = encoder(images)  # [batch_size, num_patches, embed_dim]

                trg_input = captions[:, :-1]
                trg_expected = captions[:, 1:]
                trg_mask = make_trg_mask(trg_input)

                output = decoder(trg_input, memory, trg_mask)

                output = output.reshape(-1, output.shape[2])
                trg_expected = trg_expected.reshape(-1)

                loss = criterion(output, trg_expected)
                acc = calculate_accuracy(output, trg_expected)

                val_loss += loss.item()
                val_acc += acc.item()

                val_loop.set_postfix(loss=loss.item(), accuracy=f'{acc.item()*100:.2f}%')

            average_val_loss = val_loss / len(val_loader)
            average_val_acc = val_acc / len(val_loader)
            print(f'Epoch [{epoch+1}/{config.num_epochs}], Validation Loss: {average_val_loss:.4f}, Validation Accuracy: {average_val_acc*100:.2f}%')

            # 保存验证损失和准确率
            val_losses.append(average_val_loss)
            val_accuracies.append(average_val_acc)

        # 每5轮保存一次模型
        if (epoch + 1) % 5 == 0:
            torch.save(encoder.state_dict(), f"encoder_epoch{epoch+1}.pth")
            torch.save(decoder.state_dict(), f"decoder_epoch{epoch+1}.pth")
            print(f"Models saved at epoch {epoch+1}")

        # 保存图像
        plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies)

    # 最后保存模型
    torch.save(encoder.state_dict(), encoder_save_path)
    torch.save(decoder.state_dict(), decoder_pth_path)
    print(f"Encoder model saved to {encoder_save_path}")
    print(f"Decoder model saved to {decoder_pth_path}")

def plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies):
    # 绘制损失和准确率的图像
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # 绘制损失图
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 保存图像
    plt.tight_layout()
    plt.savefig('loss_accuracy.png')
    plt.close()

if __name__ == "__main__":
    #print(torch.cuda.is_available())  # 返回 True 表示可以使用 GPU
    #print(torch.cuda.device_count())  # 打印可用 GPU 的数量
    #print(torch.cuda.get_device_name(0))  # 打印第一个 GPU 的名称
    train_encoder()
