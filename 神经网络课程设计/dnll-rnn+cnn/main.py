import os
import paddle
from train import main

if __name__ == '__main__':
    # 设置随机种子
    paddle.seed(42)
    
    # 创建必要的目录
    os.makedirs('processed', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # 开始训练
    main() 