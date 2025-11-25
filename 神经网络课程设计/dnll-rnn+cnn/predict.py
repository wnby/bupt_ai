import os
import json
import paddle
from PIL import Image
from paddle.vision import transforms
from model import ARCTIC
from config import config

class ImageDescriptionGenerator:
    def __init__(self, checkpoint_path, vocab_path):
        # 加载词典
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
            
        # 创建词索引到词的映射
        self.idx2word = {v: k for k, v in self.vocab.items()}
        
        # 创建模型
        self.model = ARCTIC(
            config.image_code_dim,
            self.vocab,
            config.word_dim,
            config.attention_dim,
            config.hidden_size,
            config.num_layers
        )
        
        # 加载模型参数
        checkpoint = paddle.load(checkpoint_path)
        self.model.set_state_dict(checkpoint['model'])
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def generate_description(self, image_path):
        """为输入图片生成描述"""
        # 读取并预处理图片
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0)
        except Exception as e:
            return f"Error loading image: {str(e)}"
            
        # 生成描述
        with paddle.no_grad():
            texts = self.model.generate_by_beamsearch(image, config.beam_k, config.max_len)
            
        # 将索引转换为词
        description = []
        for idx in texts[0]:
            if idx == self.vocab['<end>']:
                break
            if idx not in [self.vocab['<start>'], self.vocab['<pad>'], self.vocab['<unk>']]:
                description.append(self.idx2word[idx])
                
        return ' '.join(description)

def main():
    # 设置设备
    paddle.device.set_device('gpu:0' if paddle.device.is_compiled_with_cuda() else 'cpu')
    
    # 初始化生成器
    generator = ImageDescriptionGenerator(
        checkpoint_path='checkpoints/best_model.pdparams',
        vocab_path='processed/vocab.json'
    )
    
    # 测试图片路径
    image_path = os.path.join('data', 'images', 'MEN-Denim-id_00000080-01_7_additional.jpg')
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist!")
        return
        
    # 生成描述
    print("\nGenerating description for:", image_path)
    description = generator.generate_description(image_path)
    print("\nGenerated description:")
    print(description)
    
    # 打印原始描述（从captions.json中获取）
    try:
        with open('data/captions.json', 'r') as f:
            captions = json.load(f)
            original = captions.get(os.path.basename(image_path))
            if original:
                print("\nOriginal description:")
                print(original)
    except Exception as e:
        print("Could not load original description:", str(e))

if __name__ == '__main__':
    main() 