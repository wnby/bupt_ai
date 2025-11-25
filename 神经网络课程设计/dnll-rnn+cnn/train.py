import os
import json
import paddle
import paddle.nn as nn
from nltk.translate.bleu_score import corpus_bleu

from config import config
from dataset import create_dataset, get_data_loaders
from model import ARCTIC
from metrics import Metrics, save_metrics_to_file

class CrossEntropyLoss(nn.Layer):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, predictions, targets, lengths):
        preds = []
        gts = []
        for i in range(len(lengths)):
            preds.append(predictions[i, :lengths[i], :])
            gts.append(targets[i, :lengths[i]])
        preds = paddle.concat(preds, axis=0)
        gts = paddle.concat(gts, axis=0)
        return self.loss_fn(preds, gts)

def filter_useless_words(sent, filtered_words):
    return [w for w in sent if w not in filtered_words]

def evaluate(data_loader, model, vocab, epoch=None, mode='val'):
    model.eval()
    metrics = Metrics()
    filtered_words = {vocab['<start>'], vocab['<end>'], vocab['<pad>'], vocab['<unk>']}
    
    # 创建词索引到词的映射
    idx2word = {v: k for k, v in vocab.items()}
    
    for i, (imgs, caps, caplens) in enumerate(data_loader):
        with paddle.no_grad():
            texts = model.generate_by_beamsearch(imgs, config.beam_k, config.max_len)
            
            # 处理每个样本
            for j, (text, cap) in enumerate(zip(texts, caps)):
                candidate = [w for w in text if w not in filtered_words]
                reference = [w for w in cap.tolist() if w not in filtered_words]
                
                candidate = [idx2word[idx] for idx in candidate]
                reference = [idx2word[idx] for idx in reference]
                
                metrics.add_sample(f"{i}_{j}", ' '.join(candidate), ' '.join(reference))
    
    scores = metrics.compute_scores()
    
    # 保存结果，包含epoch信息
    save_metrics_to_file(scores, epoch=epoch, mode=mode)
    
    model.train()
    return scores['BLEU-4']

def main():
    paddle.device.set_device('gpu:0')
    
    # 创建必要的目录
    os.makedirs('processed', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # 处理数据集
    if not os.path.exists(os.path.join('processed', 'vocab.json')):
        vocab = create_dataset(config)
    else:
        with open(os.path.join('processed', 'vocab.json'), 'r') as f:
            vocab = json.load(f)
            
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(config)
    
    # 创建模型
    model = ARCTIC(
        config.image_code_dim,
        vocab,
        config.word_dim,
        config.attention_dim,
        config.hidden_size,
        config.num_layers
    )
    
    # 优化器
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(),
        learning_rate=config.learning_rate
    )
    
    # 损失函数
    loss_fn = CrossEntropyLoss()
    
    # 训练
    best_bleu = 0
    for epoch in range(config.num_epochs):
        model.train()
        
        for i, (imgs, caps, caplens) in enumerate(train_loader):
            optimizer.clear_grad()
            
            predictions, alphas, sorted_captions, lengths, _ = model(imgs, caps, caplens)
            
            loss = loss_fn(predictions, sorted_captions[:, 1:], lengths)
            loss += config.alpha_weight * ((1. - alphas.sum(axis=1)) ** 2).mean()
            
            loss.backward()
            
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                
            optimizer.step()
            
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{config.num_epochs}], '
                      f'Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
                      
            if (i + 1) % config.evaluate_step == 0:
                # 传入当前epoch
                bleu = evaluate(val_loader, model, vocab, epoch=epoch+1, mode='val')
                print(f'Validation BLEU-4: {bleu:.4f}')
                
                if bleu > best_bleu:
                    best_bleu = bleu
                    paddle.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'bleu': bleu
                    }, config.best_checkpoint)
                    
    # 测试
    checkpoint = paddle.load(config.best_checkpoint)
    model.set_state_dict(checkpoint['model'])
    # 最终测试结果
    test_bleu = evaluate(test_loader, model, vocab, mode='test')
    print(f'Test BLEU-4: {test_bleu:.4f}')

if __name__ == '__main__':
    main() 