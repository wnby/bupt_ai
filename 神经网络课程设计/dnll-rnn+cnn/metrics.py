import os
import json
import numpy as np
from collections import defaultdict
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
from nltk import ngrams

class Metrics:
    def __init__(self):
        self.refs = defaultdict(list)
        self.cands = {}
        
    def add_sample(self, image_id, candidate, reference):
        """添加一个样本的预测结果和参考描述"""
        self.refs[image_id].append(reference)
        self.cands[image_id] = candidate
        
    def compute_scores(self):
        """计算所有评估指标"""
        scores = {}
        
        # 计算BLEU-4
        refs_list = [[ref] for refs in self.refs.values() for ref in refs]
        cands_list = [self.cands[image_id] for image_id in self.refs.keys()]
        scores['BLEU-4'] = self._compute_bleu(cands_list, refs_list)
        
        # 计算CIDEr-D
        scores['CIDEr-D'] = self._compute_cider(self.cands, self.refs)
        
        # 计算SPICE
        scores['SPICE'] = self._compute_spice(self.cands, self.refs)
        
        return scores
        
    def _compute_bleu(self, candidates, references):
        """计算BLEU-4分数"""
        return corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))
        
    def _compute_cider(self, candidates, references, n=4, sigma=6.0):
        """计算CIDEr-D分数"""
        # 计算TF-IDF权重
        doc_freq = defaultdict(float)
        for refs in references.values():
            for ref in refs:
                for ng in range(1, n+1):
                    for gram in ngrams(ref, ng):
                        doc_freq[gram] += 1
                        
        # 计算IDF
        num_refs = sum(len(refs) for refs in references.values())
        for gram, freq in doc_freq.items():
            doc_freq[gram] = np.log(num_refs / freq)
            
        # 计算每个样本的CIDEr得分
        scores = []
        for image_id, cand in candidates.items():
            score = 0
            refs = references[image_id]
            
            for ng in range(1, n+1):
                # 计算候选描述的TF向量
                cand_vec = defaultdict(float)
                for gram in ngrams(cand, ng):
                    cand_vec[gram] += 1
                
                # 计算参考描述的TF向量
                ref_vecs = []
                for ref in refs:
                    vec = defaultdict(float)
                    for gram in ngrams(ref, ng):
                        vec[gram] += 1
                    ref_vecs.append(vec)
                    
                # 应用TF-IDF权重
                for gram, freq in cand_vec.items():
                    cand_vec[gram] = freq * doc_freq.get(gram, 0)
                for vec in ref_vecs:
                    for gram, freq in vec.items():
                        vec[gram] = freq * doc_freq.get(gram, 0)
                        
                # 计算余弦相似度
                for vec in ref_vecs:
                    # 计算分子
                    numerator = sum(cand_vec[gram] * vec[gram] for gram in set(cand_vec) & set(vec))
                    # 计算分母
                    denominator = np.sqrt(sum(val**2 for val in cand_vec.values()) * 
                                        sum(val**2 for val in vec.values()))
                    if denominator:
                        score += numerator / denominator
                        
            score = np.exp(-(score / len(refs) - 1)**2 / (2 * sigma**2))
            scores.append(score)
            
        return np.mean(scores)
        
    def _compute_spice(self, candidates, references):
        """计算SPICE分数(简化版本)"""
        # 这里实现一个简化版的SPICE
        # 实际的SPICE需要进行语义解析和场景图匹配，实现较为复杂
        scores = []
        for image_id, cand in candidates.items():
            refs = references[image_id]
            
            # 将描述转换为单词集合
            cand_words = set(word_tokenize(cand.lower()))
            
            # 计算与每个参考描述的F1分数
            f1_scores = []
            for ref in refs:
                ref_words = set(word_tokenize(ref.lower()))
                
                # 计算精确率和召回率
                if len(cand_words) == 0:
                    precision = 0
                else:
                    precision = len(cand_words & ref_words) / len(cand_words)
                    
                if len(ref_words) == 0:
                    recall = 0
                else:
                    recall = len(cand_words & ref_words) / len(ref_words)
                    
                # 计算F1分数
                if precision + recall == 0:
                    f1 = 0
                else:
                    f1 = 2 * precision * recall / (precision + recall)
                    
                f1_scores.append(f1)
                
            # 取最高的F1分数
            scores.append(max(f1_scores))
            
        return np.mean(scores)

def save_metrics_to_file(scores, filename='evaluation_results.txt', epoch=None, mode='test'):
    """将评估结果保存到文件
    
    Args:
        scores: 评估分数字典
        filename: 输出文件名
        epoch: 当前轮次
        mode: 'train', 'val' 或 'test'
    """
    # 如果文件不存在，创建文件并写入标题
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("Image Description Generation Evaluation Results\n")
            f.write("===========================================\n\n")
    
    # 追加写入模式
    with open(filename, 'a') as f:
        if epoch is not None:
            f.write(f"\nEpoch {epoch} - {mode}\n")
            f.write("-" * 20 + "\n")
        else:
            f.write(f"\n{mode.capitalize()} Results\n")
            f.write("-" * 20 + "\n")
            
        for metric, score in scores.items():
            f.write(f"{metric}: {score:.4f}\n")
        f.write("\n") 