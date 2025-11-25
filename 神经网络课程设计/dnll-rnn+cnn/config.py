from argparse import Namespace

config = Namespace(
    # 数据参数
    data_dir = './data',
    max_len = 30,
    min_word_count = 5,
    batch_size = 32,
    
    # 模型参数
    image_code_dim = 2048,
    word_dim = 512,
    hidden_size = 512,
    attention_dim = 512,
    num_layers = 1,
    
    # 训练参数
    learning_rate = 0.0005,
    num_epochs = 10,
    grad_clip = 5.0,
    alpha_weight = 1.0, # 注意力正则化权重
    evaluate_step = 100,
    
    # 模型保存
    checkpoint = None,
    best_checkpoint = 'checkpoints/best_model.pdparams',
    last_checkpoint = 'checkpoints/last_model.pdparams',
    
    # 生成参数
    beam_k = 5
) 