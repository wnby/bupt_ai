## 题目：基于编解码框架的图像描述生成

> 2024 秋季北京邮电大学深度学习与神经网络课程设计

## 一、任务说明

- 详细题目：设计并实现一个图像描述模型，输入图像，后图像描述模型能够自动为图片生成流畅关联的自然语言描述
- 实现模型：CNN+GRU、网络/区域表示、Transformer 编码器+Transformer 解码器和视觉 Transformer+Transformer 解码器
- 评价标准：BLEU-4、CIDEr-D 和 SPICE
- 任选任务：利用训练的服饰图像描述模型和多模态大语言模型，为真实背景的服饰图像数据集增加服饰描述和背景描述，构建全新的服饰图像描述数据集
- 额外说明：后两个模型有一定耦合性，所以我们是这么实现的：首先写解码器，然后写视觉 Transformer+Transformer 解码器，然后写网格表示加Transformer 编码器
- 训练过程（亮点）：我们用到了微调的技术，在CNN+GRU里，我们用到了预训练的ResNet101模型去训练GRU，在后两个模型里，我们首先用预训练的vit去训练编码器，然后再用训练好的编码器去训练两个解码器。这样可以非常有效的加快了训练的速度，压缩epoch，尤其是在一轮要一小时的情况下。

## 二、所用方法及模型

### 2.1.0 实训平台

实训平台跑不了这个任务，会出现各种问题，比如上传数据解压之后数据会丢失，以及一些包的版本问题，下载又巨慢，以及评估的时候模型一大就加载不了这种问题。

```
RuntimeError: CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 9.77 GiB total capacity; 915.03 MiB already allocated; 6.69 MiB free; 1006.00 MiB reserved in total by PyTorch)
```

没有服务器的算力，我们想了很多办法去减少算力开销，上面的训练过程也有所提及。

### 2.1.1 数据处理

很友好的图片，尺寸224乘224，和很多模型刚刚好能对上。

将 DeepFashion-MultiModel 数据集中 image 和 textual description 的数据存下来，编写一个处理数据集的脚本dataset.py，生成词典和训练数据：

> https://github.com/yumingj/DeepFashion-MultiModal

脚本函数设计如下：

```Python
def create_dataset(config):
    """处理数据集,生成词典和训练数据"""
    
    # 读取captions.json
    with open(os.path.join(config.data_dir, 'captions.json'), 'r') as f:
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
    words = [w for w in vocab.keys() if vocab[w] >= config.min_word_count]
    vocab = {k: v + 1 for v, k in enumerate(words)}
    vocab['<pad>'] = 0
    vocab['<unk>'] = len(vocab)
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
            img_path = os.path.join(config.data_dir, 'images', img)
            captions = image_captions[img]
            
            # 编码caption
            enc_captions = []
            for c in captions:
                enc_c = [vocab['<start>']] + [vocab.get(w, vocab['<unk>']) for w in c] + [vocab['<end>']]
                enc_captions.append(enc_c)
                
            data['IMAGES'].extend([img_path] * len(enc_captions))
            data['CAPTIONS'].extend(enc_captions)
            
        with open(os.path.join('processed', f'{split}_data.json'), 'w') as f:
            json.dump(data, f)
            
    return vocab
```

------

### 2.2 CNN+GRU

#### 2.2.1图像编码器

**核心思想：**图像编码器部分主要使用了基于ResNet101的卷积神经网络来提取图像的特征表示。ResNet101是一种深度残差网络，其设计理念是通过使用残差学习的框架来促进更深网络的训练。在本实现中，我们采用了预训练的ResNet101模型来初始化权重，这有助于提高模型在特定任务上的学习效率和效果

代码实现：

```Python
class ImageEncoder(nn.Layer):
    """图像编码器类
    使用预训练的ResNet101模型提取图像特征
    """
    def __init__(self, finetuned=True):
        super(ImageEncoder, self).__init__()
        # 加载预训练的ResNet101模型，去掉最后两层
        model = models.resnet101(pretrained=True)
        self.grid_representation_extractor = nn.Sequential(*(list(model.children())[:-2]))
        # 设置是否微调预训练模型的参数
        for param in self.grid_representation_extractor.parameters():
            param.requires_grad = finetuned
            
    def forward(self, images):
        # 提取图像的网格特征表示
        return self.grid_representation_extractor(images)
```

#### 2.2.2 注意力解码器

![img](\图片\4b9f7bf27bda033f46512179381445a7.png)
x_t：当前时刻输入信息
h_t-1：上一时刻隐藏信息
h_t:传递到下一时刻的隐藏状态
~h_t：候选隐藏状态
r_t：重置门
z_t：更新门
**GRU中的核心就是两个门——重置门和更新门**
**重置门决定了如何将新的输入信息与前面的记忆相结合**
![4797d04b16eb9168e041bf8288ab6deb](图片\4797d04b16eb9168e041bf8288ab6deb.png)

**更新门用于控制前一时刻的状态信息被带入到当前状态中的程度，简单来说就是更新记忆**
![img](图片\03808590791e648b4933a2228690e3f0.png)

**核心思想：**在注意力解码器部分，我们采用了门控循环单元（GRU）作为核心的RNN结构来进行文本的序列生成。GRU是一种有效的序列模型，它能够通过其更新门和重置门来解决长期依赖问题。

代码实现：

```Python
class AttentionDecoder(nn.Layer):
    """带注意力机制的解码器类
    用于生成图像描述
    """
    def __init__(self, image_code_dim, vocab_size, word_dim, attention_dim, hidden_size, num_layers, dropout=0.5):
        super(AttentionDecoder, self).__init__()
        # 初始化各个组件
        self.embed = nn.Embedding(vocab_size, word_dim)    # 词嵌入层
        self.attention = AdditiveAttention(hidden_size, image_code_dim, attention_dim)  # 注意力层
        self.init_state = nn.Linear(image_code_dim, num_layers*hidden_size)  # 初始隐状态生成层
        self.rnn = nn.GRU(word_dim + image_code_dim, hidden_size, num_layers, time_major=False)  # GRU层
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)       # 输出层
        self.vocab_size = vocab_size
```

---

### 2.3 Transformer 解码器

#### 2.3.0 失败案例



#### 2.3.1 解码器设计



我们代码里有很多print，那是因为写的时候老是对不上，需要一层一层的对......

#### 1. `TransformerDecoder` 类

- **功能**：负责将目标序列（如文本）转换为输出，通过多层解码器块（`TransformerDecoderBlock`）与编码器的记忆（`memory`）进行交互。
  - **嵌入层** (`nn.Embedding`)：将目标词汇映射到嵌入空间。
  - **位置编码** (`nn.Embedding`)：为每个位置添加位置信息，帮助模型理解序列顺序。
  - **解码器层** (`TransformerDecoderBlock`)：包含多个解码器块，逐层处理输入。
  - **输出层** (`nn.Linear`)：将解码器的输出映射到词汇表大小，生成最终预测。
  - **Dropout 层**：防止过拟合，提升模型泛化能力。

#### 2. `TransformerDecoderBlock` 类

- **功能**：构成解码器的基本单元，负责自注意力和跨注意力机制，以及前馈网络处理。
  - **自注意力层** (`MultiHeadAttention`)：允许解码器自身在生成时关注目标序列的不同部分。
  - **跨注意力层** (`MultiHeadAttention`)：使解码器能够关注编码器的输出（记忆），实现信息融合。
  - **归一化层** (`nn.LayerNorm`)：对每一步的输出进行规范化，提升训练稳定性。
  - **前馈网络** (`FeedForwardBlock`)：进一步处理注意力输出，增强模型表达能力。
  - **Dropout 层**：应用于各个子层，防止过拟合。

#### 3. `MultiHeadAttention` 类

- **功能**：实现多头注意力机制，允许模型在不同的子空间中并行关注信息。
  - **线性变换层** (`nn.Linear`)：分别生成查询（Q）、键（K）和值（V）向量。
  - **多头分割**：将嵌入维度分为多个头，每个头独立计算注意力。
  - **注意力计算**：通过点积计算注意力得分，并应用softmax归一化。
  - **输出线性层**：将多个头的输出拼接并线性变换回原始嵌入维度。
  - **Dropout 层**：应用于注意力权重，防止过拟合。

#### 4. `FeedForwardBlock` 类

- **功能**：实现前馈神经网络，增强模型的非线性表达能力。

  - **线性层** (`nn.Linear`)：两层全连接网络，中间使用ReLU激活函数。

  - **Dropout 层**：应用于激活后，防止过拟合。

    

#### 2.3.2 连接预训练模型



##### 1.预训练模型的输入输出



ViT_B_16 模型本来是用来图像分类的（虽然这不意味着它只会图像分类），因此它不能直接进入解码器的交叉多头里面。我们需要修改和定制 ViT 模型的输出，使其更适合作为特征提取器而不是分类器。



**输入形状**：输入张量`x`的形状为`torch.Size([32, 3, 224, 224])`，表示批量大小为32的RGB图像，每张图像的尺寸为224x224。

**卷积投影**：通过`conv_proj`层后，张量形状变为`torch.Size([32, 768, 14, 14])`，将图像划分为14x14的补丁，每个补丁被投影到768维的嵌入空间。

**展平与转置**：将卷积输出进行展平和转置处理后，张量形状变为`torch.Size([32, 196, 768])`，其中196代表补丁数量（14x14），768为嵌入维度。

**添加分类令牌**：通过扩展分类令牌并与补丁嵌入拼接，得到形状为`torch.Size([32, 197, 768])`的张量。

**位置嵌入与Dropout**：将位置嵌入加到张量上，并应用Dropout，保持形状不变。

**Transformer编码器**：经过Transformer编码器处理后，输出形状仍为`torch.Size([32, 197, 768])`。

**应用LayerNorm**：对编码器输出应用LayerNorm，保持形状不变。

**移除分类令牌**：最后移除分类令牌，仅保留补丁嵌入，得到最终输出形状为`torch.Size([32, 196, 768])`。

**代码输出如下：**

输入 x 的形状: torch.Size([32, 3, 224, 224])
经过 conv_proj 后的形状: torch.Size([32, 768, 14, 14])
经过 flatten 和 transpose 后的形状: torch.Size([32, 196, 768])
class_token 的形状: torch.Size([32, 1, 768])
添加 class_token 后的形状: torch.Size([32, 197, 768])
加上位置嵌入后的形状: torch.Size([32, 197, 768])
经过 dropout 后的形状: torch.Size([32, 197, 768])

经过 transformer 编码器后的形状: torch.Size([32, 197, 768])
应用 LayerNorm 后的形状: torch.Size([32, 197, 768])
移除 class_token 后的最终输出形状: torch.Size([32, 196, 768])



##### 2.交叉多头自注意力



```
memory = encoder(images)
output = decoder(trg_input, memory, trg_mask)
```

**Q**：来自解码器的当前输入，即`trg_input`。

**键和值**：来自编码器的输出，即`memory`。



#### 2.3.3 训练函数



##### config参数设置

在训练之前，我们需要定义一些重要的配置参数。这些参数定义了模型、数据和训练过程的各个方面，包括：

*   `data_dir`: 数据集的根目录。
*   `max_len`: 目标序列的最大长度。这个值应该与 ViT 编码器的 `num_patches` 相匹配，并加上 <start> 和 <end> 标记。
*   `min_word_count`: 词汇表中词语的最小出现次数。
*   `batch_size`: 每个训练批次的大小。
*   `learning_rate`: 优化器的学习率。
*   `num_epochs`: 训练的总轮数。
*   `embed_dim`: 嵌入层的维度。它应该与 ViT 编码器的输出维度匹配。
*   `decoder_num_heads`: Transformer 解码器中多头注意力机制的头数。
*   `decoder_num_layers`: Transformer 解码器的层数。
*   `decoder_forward_expansion`: 解码器前馈网络的扩展因子。
*   `decoder_dropout`: 解码器中 dropout 的概率。
*   `device`: 模型运行的设备 (GPU 或 CPU)。

##### 掩码张量

为了让 Transformer 解码器在生成字幕时不会“偷看”未来的单词，我们使用了一个掩码张量。`make_trg_mask(trg_input)` 函数用于生成这个掩码，具体步骤如下：
1. 创建一个下三角矩阵，确保每个位置只能看到自己及之前的位置。
2. 将该矩阵扩展到 `[batch_size, 1, seq_len, seq_len]` 的形状。

```
def make_trg_mask(trg_input):
batch_size, sz = trg_input.size()
\# 创建一个下三角矩阵，确保每个位置只能看到自己及之前的位置
mask = torch.tril(torch.ones(sz, sz, device=trg_input.device)).bool()
\# 扩展为 [batch_size, 1, sz, sz]
mask = mask.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1) # [batch_size, 1, sz, sz]
return mask
```

##### 冻结编码器

为了节省计算资源和时间，我们冻结了 ViT 编码器的参数。这意味着在训练过程中，编码器的权重不会被更新。这一步通过设置 `param.requires_grad = False` 实现。

```
for param in encoder.parameters():
param.requires_grad = False
```

##### 在已有模型上训练

直接加载已经训练的模型，在此基础上再训练，不浪费之前训练的结果，节约算力。比如10个epoch的基础上训5个epoch，就是15个epoch。

##### 损失函数

用 `nn.CrossEntropyLoss` 作为损失函数，并使用 `ignore_index=0` 来忽略填充的令牌，因为它不应该影响损失的计算。

```
criterion = nn.CrossEntropyLoss(ignore_index=0)
```

##### 优化器

使用 `optim.Adam` 作为优化器，只优化解码器的参数。

```
optimizer = optim.Adam(decoder.parameters(), lr=config.learning_rate)
```



#### 2.3.4 生成函数

用了束优化+重复词权重降低+生成重复词语





**束优化**是一种在序列生成任务中广泛使用的搜索策略，初始束仅包含一个序列，即起始标记`<start>`。

```
beams = [( [start_token_index], 0.0, {} )]
```

每个束包含三个部分：

**序列**：当前生成的词汇索引列表。

**对数概率**：当前序列的累积对数概率。

**词频**：记录当前序列中各个词汇的出现频率，用于后续的重复词惩罚。

然后迭代生成扩展，最后束合并：

如果任意束生成了结束标记`<end>`，则提前终止生成过程。

```
new_beams.sort(key=lambda x: x[1], reverse=True)
beams = new_beams[:beam_size]
```

为了避免生成过程中出现重复词汇，在束优化过程中引入了**重复词权重降低**机制。

不过有些词不重复是不可能的，我们专门为它们打了个表，比如is，on这种。

---

### 2.4 视觉 Transformer 编码器

#### 2.4.0 参考课程

难度太大了，就自己找了个课程

https://www.bilibili.com/video/BV1AL411W7dT

#### 2.4.1 视觉 Transformer（ViT）结构

可以看到，虽然vit和网格表示是两个任务，但是他们都用到了CNN作为提取图像特征的工具。

![fba01ac1b74efef84897a542deac353](图片\fba01ac1b74efef84897a542deac353.jpg)

#### 2.4.2 简略代码实现



大神的Vit有很多的小trick以及细节。

最惊艳的还是大神对于q，k和v的处理。有几个地方，按理来说三层才能实现的东西，大神几行代码变过来变过去一起解决了。处理的很优美，就是有点难懂。

#### 1. `PatchEmbed` 类

将2D图像分割成固定大小的补丁，并将这些补丁嵌入到高维空间中。

- `img_size`: 输入图像的尺寸（默认224x224）。
- `patch_size`: 每个补丁的尺寸（默认16x16）。
- `in_c`: 输入图像的通道数（默认3，RGB图像）。
- `embed_dim`: 嵌入维度（默认768）。
- `norm_layer`: 归一化层（可选）。

#### 2. `Attention` 类

实现多头自注意力机制，用于捕捉序列中不同位置之间的依赖关系。

- **初始化参数**：
  - `dim`: 嵌入维度。
  - `num_heads`: 注意力头的数量（默认8）。
  - 其他参数用于控制偏置、缩放因子和Dropout。
- **主要操作**：
  1. 通过线性层生成QKV。
  2. 分头计算注意力权重，并应用softmax。
  3. 加权求和值向量，合并各头输出，并通过线性层投影回原始维度。
- **前向传播**：
  - 输入形状 `[B, N, C]` 处理后输出相同形状的张量。

#### 3. `Mlp` 类

实现前馈神经网络，增强模型的非线性表达能力。

- `in_features`: 输入特征维度。
- `hidden_features`: 隐藏层维度（默认是4倍的输入维度）。
- `out_features`: 输出特征维度（默认与输入相同）。
- 激活函数和Dropout率。

#### 4. `Block` 类

构成Transformer编码器的基本单元，包含自注意力和前馈网络。

1. 自注意力层
   - 对输入进行LayerNorm归一化。
   - 通过多头自注意力机制处理输入。
   - 残差连接和Dropout。
2. 前馈网络
   - 对自注意力输出进行LayerNorm归一化。
   - 通过MLP处理。
   - 残差连接和Dropout。

#### 5. `VisionTransformerEncoder` 类

整合上述组件，构建完整的ViT编码器，将输入图像转换为特征表示。



------

### 2.5 网格表示+Transformer 编码器

#### 2.5.1 它的结构

![image-20250105032110143](C:\Users\49452\AppData\Roaming\Typora\typora-user-images\image-20250105032110143.png)

#### 2.5.2 简要代码实现

因为已经写过一遍编码器了，这个和vit的差别主要是一开始特征提取的地方，所以为了减少工作量，直接调包了。

主要还是区域表示的部分和上面不一样。

将输入的图像通过一个卷积神经网络进行特征提取，并将维度和Transformer模型对其，再加入位置嵌入，这些思路中期报告都有写就不再多说了。

首先用预训练的ResNet50提取特征：

```
self.cnn = models.resnet50(pretrained=True)
self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])  # 输出尺寸：[batch_size, 2048, H/32, W/32]
```

我们移除ResNet50的最后两个层（平均池化层和全连接层），从而保留ResNet50输出的特征图。

输出的尺寸是 `[batch_size, 2048, H/32, W/32]`，其中 `H/32` 和 `W/32` 表示输入图像经过多层卷积和池化后，空间分辨率被降低了32倍。

然后1x1卷积将2048通道调整为嵌入维度 `embed_dim`

```
self.conv = nn.Conv2d(2048, embed_dim, kernel_size=1)  # 输出尺寸：[batch_size, embed_dim, H/32, W/32]
```

然后展平空间维度为序列

```
features = features.flatten(2).permute(0, 2, 1)  # [batch_size, num_tokens, embed_dim]
```

处理输入序列长度后，加入位置嵌入：

```
features = features + self.positional_encoding[:, :features.size(1), :]
```

最后转换为Transformer输入格式

```
features = features.permute(1, 0, 2)
```

Transformer的输入格式要求是 `[num_tokens, batch_size, embed_dim]`，所以需要将特征的维度从 `[batch_size, num_tokens, embed_dim]` 转换为 `[num_tokens, batch_size, embed_dim]`。

---

### 2.6 EX任选任务

#### 2.6.1 设计思路

##### 数据增强与描述生成

为了更真实地表达服饰的细节和背景，我们设计了两种描述生成方式：

- **基于训练模型的服饰描述生成**：使用自定义的服饰图像描述模型生成服饰相关的描述。
- **基于外部多模态大语言模型的背景描述生成**：通过调用外部API，生成图像背景的描述。

最终将两种描述合并，为每张图片生成综合性的描述。

##### 模型训练与评估

在生成描述后，将新的服饰图像描述数据集用于重新训练服饰描述模型，并通过实验验证新数据集对模型性能的提升效果。

##### 模块化设计

为了提高代码的可维护性和扩展性，设计了以下功能模块：

- 图像描述生成模块
- 数据处理与保存模块
- 图像预处理模块

#### 2.6.2 代码结构与功能说明

##### 主代码框架

```python
if __name__ == '__main__':
    image_folder = os.path.join("data", "images")  # 替换为你的图片文件夹路径
    output_json = "image_labels.json"  # 输出 JSON 文件名
    generate_labels_json(image_folder, output_json)
```

主代码通过调用 `generate_labels_json` 函数完成对图像文件夹的遍历，并生成描述文件。

##### 图像描述生成模块

```python
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
```

此部分代码定义了 `ImageDescriptionGenerator` 类，负责加载训练好的服饰描述模型，并将输入图像转换为对应的文本描述。

##### 外部API调用

```python
def get_image_description(image_path, access_token):
    url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/image2text/fuyu_8b?access_token={access_token}"
    image_base64 = encode_image_to_base64(image_path)
    if not image_base64:
        return None
    payload = json.dumps({
        "prompt": "What is the picture's background like",
        "image": image_base64
    })
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, headers=headers, data=payload)
        response_data = response.json()
        result = response_data.get("result")
        if result:
            result = ' '.join(result.strip().splitlines())  # 移除中间和结尾的换行符
    except Exception as e:
        print(f"Error calling API: {str(e)}")
        result = None

    return result
```

此部分通过调用外部API获取图像背景描述，采用图像编码为Base64的方式传递数据。

##### 数据处理与保存模块

```python
def generate_labels_json(folder_path, output_file):
    access_token = get_access_token()
    if not access_token:
        print("Failed to retrieve access token.")
        return

    labels = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            description = get_image_description(file_path, access_token)
            if description:
                labels[filename] = description
                print(f"Processed {filename}: {description}")
            else:
                print(f"Failed to process {filename}")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=4)
        print(f"Labels saved to {output_file}")
    except Exception as e:
        print(f"Error saving JSON: {str(e)}")
```

此函数遍历文件夹中的所有图像文件，并将生成的描述保存到JSON文件中。

##### 图像预处理模块

```python
def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Image file not found: {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None
```

该模块负责将图像转换为Base64格式，以便与外部API兼容。

#### 2.6.3 问题与分析

在实验中遇到以下几个问题，并进行了针对性的分析和解决：

##### 图片的格式在Payload中需要Base64编码

API的图像输入要求是Base64字符串，而不是常见的JPEG或PNG格式。

所以需要 `encode_image_to_base64` 函数，将图像文件读取后转换为Base64格式。

Base64编码能够将图像二进制数据以字符串形式表示，便于通过HTTP协议传输，同时减少因编码格式问题导致的解析错误。

##### API调用时图片被认为违规时返回空值

某些图片因敏感内容被API判定为违规，导致API输出为空，无法获取背景描述。
在代码中加入异常处理和日志记录机制，当API返回空值时，将错误原因记录并跳过该图片。此问题由于API的安全策略造成，异常处理即可。

##### 字符串连接问题


生成的服饰描述和背景描述需要整合为一段完整的文字，但两部分内容格式不统一，并且会出现换行符。 
使用Python字符串方法对两部分描述进行格式化，将换行符和多余空格移除后合并。

字符串连接问题在多模态描述生成中较为常见，需要统一输出的格式，确保描述语义完整且可读。

### 2.7 评价标准

#### 2.7.1BLEU-4


BLEU是用于评估模型生成的句子(candidate)和实际句子(reference)的差异的指标.
它的取值范围在0.0到1.0之间, 如果两个句子完美匹配(perfect match), 那么BLEU是1.0, 反之, 如果两个句子完美不匹配(perfect mismatch), 那么BLEU为0.0

**公式**
n-gram层面的准确率：
$$
\text{BLEU}_n (a, b) = \frac{\sum_{w_n \in a} \min \left( c_a(w_n), \max_{j=1}^{|b|} c_{b_j}(w_n) \right)}{\sum_{w_n \in a} c_a(w_n)}
$$

- a：候选句子（生成句子）
- b：参考句子集（真实句子集、标注句子集）
- w_n：n-gram
- c_x(y_n)：n-gram y_n在句子x中出现的次数

选用n=4

**代码实现**

```Python
    def compute_scores(self):
        """计算所有评估指标"""
        scores = {}
        # 计算BLEU-4
        refs_list = [[ref] for refs in self.refs.values() for ref in refs]
        cands_list = [self.cands[image_id] for image_id in self.refs.keys()]
        scores['BLEU-4'] = self._compute_bleu(cands_list, refs_list)
......
         return scores
    def _compute_bleu(self, candidates, references):
        """计算BLEU-4分数"""
        return corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))
```

------

#### 2.7.2CIDEr-D

**1.核心逻辑**
CIDEr-D 是基于 n-grams 和 TF-IDF 权重的加权余弦相似度，具体计算方式如下：

1. **多个生成描述的处理**：
   - 如果 `generated_captions` 中每张图片有多个生成描述（如 `["desc1", "desc2", ...]`），CIDEr-D 通常会取这些描述的平均分数。
   - 每个生成描述会与所有参考描述（`reference_captions`）比较，计算单个生成描述的 CIDEr-D 分数。
   - 对于同一张图片的多个生成描述，计算分数后取平均值。
   
2. **最终分数的计算**：
   - 对数据集中的所有图片，分别计算每张图片的平均分数。
   - 再对所有图片的平均分数取整体的均值，作为最终的 CIDEr-D 分数。

**2.公式**

设 \( S(i, j) \) 是第 \( i \) 张图片的第 \( j \) 个生成描述与所有参考描述的 CIDEr-D 分数：

$$
\text{CIDEr-D}_i = \frac{1}{N_i} \sum_{j=1}^{N_i} S(i, j)
$$

其中 \( N_i \) 是第 \( i \) 张图片生成描述的个数。

最终：

$$
\text{CIDEr-D} = \frac{1}{M} \sum\_{i=1}^M \text{CIDEr-D}\_i
$$

其中 \( M \) 是数据集中的图片总数。

**3.代码实现：**

```Python
class Cider:
    def __init__(self, n=4, sigma=6.0):
        """
        初始化 Cider 类
        参数:
            n: n-grams 的最大长度
            sigma: 高斯权重的标准差
        """
        self.n = n
        self.sigma = sigma
```

**类的初始化**：

- `n` 是 n-grams 的最大长度（默认为 4）。
- `sigma` 是高斯衰减的标准差（默认为 6.0）。

```Python
    def compute_score(self, gts, res):
......
        """
        计算 CIDEr-D 分数
        参数:
            gts: 参考描述，字典 {image_id: [参考描述1, 参考描述2, ...]}
            res: 生成描述，字典 {image_id: [生成描述]}
        返回:
            score: 平均 CIDEr-D 分数
            scores: 每张图片的分数
        """
        # 提取所有描述
        all_references = []
        all_hypotheses = []

        for image_id in gts.keys():
            all_references.extend(gts[image_id])  # 参考描述
            all_hypotheses.extend(res[image_id])  # 生成描述

        # 计算 TF-IDF 矩阵
        tfidf_vec = TfidfVectorizer(ngram_range=(1, self.n))
        tfidf_matrix = tfidf_vec.fit_transform(all_references + all_hypotheses)

        # 分离参考描述和生成描述的矩阵
        reference_matrix = tfidf_matrix[:len(all_references), :]
        hypothesis_matrix = tfidf_matrix[len(all_references):, :]

        # 计算余弦相似度
        cosine_sim = hypothesis_matrix @ reference_matrix.T
        scores = cosine_sim.max(axis=1).toarray().flatten()  # 对每个生成描述取最大相似度

        # 应用高斯权重
        weights = np.exp(-(scores ** 2) / (2 * self.sigma ** 2))
        scores *= weights

        # 计算平均 CIDEr-D 分数
        score = np.mean(scores)
        return score, scores
```

**`compute_score` 方法**：

- `gts`: 参考描述，格式为字典，键是图片 ID，值是一个包含多条参考描述的列表。
- `res`: 生成描述，格式为字典，键是图片 ID，值是一个包含生成描述的列表。

**TF-IDF 矩阵**：

- 使用 `TfidfVectorizer` 生成 n-grams 的 TF-IDF 矩阵。

**余弦相似度计算**：

- 通过矩阵乘法计算生成描述与参考描述之间的余弦相似度。
- 每个生成描述取与所有参考描述的最大相似度。

**高斯权重**：

- 对分数应用高斯权重，使得重复的 n-grams 贡献较小。

```Python
# 示例数据
generated = {
    0: ["The lower clothing is of long length. The fabric is cotton and it has plaid patterns."]
}
references = {
    0: ["The lower clothing is of long length. The fabric is cotton and it has plaid patterns."]
}

# 计算 CIDEr-D
cider = Cider()
score, scores = cider.compute_score(references, generated)
print(f"CIDEr-D Score: {score:.4f}")
print(f"Individual Scores: {scores}")
```

**输出**：

- `score`: 数据集的平均 CIDEr-D 分数。
- `scores`: 每张图片的 CIDEr-D 分数。

---

#### 2.7.3 SPICE

**1. 核心逻辑**

1. **场景图提取**：
- 使用 `_extract_ngrams` 方法将文本分解为 n-grams（默认是 1-grams）。
   - 构建生成描述和参考描述的 n-grams 计数器，表示每个描述的场景图。
   
2. **相似性计算**：

   - 对生成描述与参考描述逐一计算 F1-score（基于 n-grams 的精确率和召回率）。
   - 返回最大 F1-score，作为该生成描述与参考描述的语义相似性。

3. **平均分数**：
   - 对数据集中所有图片的 SPICE 分数取平均值，作为整体 SPICE 分数。

**2. 方法解析**

1. **`_extract_ngrams`**：

   - 将输入文本分词后提取 n-grams，用于表示场景图。
   - 返回一个 `Counter` 对象，记录每个 n-gram 出现的次数。

2. **`_compute_similarity`**：

   - 对每个生成描述与所有参考描述的场景图计算 F1-score。
   - 返回最大 F1-score，作为当前生成描述的 SPICE 分数。

3. **`compute_score`**：
   - 遍历每张图片的生成描述和参考描述，计算 SPICE 分数。
   - 返回平均 SPICE 分数和每张图片的 SPICE 分数列表。

**3.spice 代码实现**

```Python

class Spice:
    def __init__(self):
        """
        初始化 SPICE 类
        """
        nltk.download('punkt')

    def _extract_ngrams(self, text, n=1):
        """
        提取 n-grams
        参数:
            text: 输入文本
            n: n-gram 的长度
        返回:
            Counter: n-grams 计数
        """
        tokens = nltk.word_tokenize(text.lower())
        ngrams = zip(*[tokens[i:] for i in range(n)])
        return Counter([" ".join(ngram) for ngram in ngrams])

    def _compute_similarity(self, generated_graph, reference_graphs):
        """
        计算生成描述与参考描述场景图之间的语义相似性
        参数:
            generated_graph: 生成描述的场景图 Counter
            reference_graphs: 参考描述的场景图列表 [Counter, Counter, ...]
        返回:
            float: SPICE 分数
        """
        # 计算生成描述与每个参考描述的 F1-score
        f1_scores = []
        for ref_graph in reference_graphs:
            overlap = sum((generated_graph & ref_graph).values())
            precision = overlap / sum(generated_graph.values()) if generated_graph.values() else 0
            recall = overlap / sum(ref_graph.values()) if ref_graph.values() else 0
            f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
            f1_scores.append(f1)
        return max(f1_scores)  # 返回最大 F1-score

    def compute_score(self, gts, res):
        """
        计算 SPICE 分数
        参数:
            gts: 参考描述，字典 {image_id: [参考描述1, 参考描述2, ...]}
            res: 生成描述，字典 {image_id: [生成描述]}
        返回:
            score: 平均 SPICE 分数
            scores: 每张图片的分数
        """
        scores = []
        for image_id in gts.keys():
            # 提取生成描述和参考描述的场景图
            generated_graph = self._extract_ngrams(res[image_id][0], n=1)  # 假设每个生成描述只有一个
            reference_graphs = [self._extract_ngrams(ref, n=1) for ref in gts[image_id]]

            # 计算当前图片的 SPICE 分数
            spice_score = self._compute_similarity(generated_graph, reference_graphs)
            scores.append(spice_score)

        # 计算平均分数
        average_score = np.mean(scores)
        return average_score, scores
```

## 三、运行结果及分析

### 3.1CNN+GRU

**图片生成描述：**![image-20250104182511939](图片\image-20250104182511939.png)

**Accuracy and Loss：**
![accuracy](图片\1accuracy.png)

![loss](图片\1loss.png)

**评估结果：**

```
Test Results
--------------------
BLEU-4: 0.5630
CIDEr-D: 0.9738
SPICE: 0.6125
```

------

### 3.2 区域表示、Transformer编码器+Transformer解码器

训练结果（编码器解码器一起训，解码器以及CNN是预训练的）：

![aad16bb282d7a2ef95c3aa24b24ca4f](图片\aad16bb282d7a2ef95c3aa24b24ca4f.png)

生成结果：

![9f4af3d6f2776cd8b48639d5b8de0bd](图片\9f4af3d6f2776cd8b48639d5b8de0bd.png)

------

### 3.3 ViT+Transformer解码器

训练结果（编码器单独训，解码器是vit预训练模型）：

![0503a635f1f88c616cdab53a81c2dc4](图片\0503a635f1f88c616cdab53a81c2dc4.png)

训练结果（编码器解码器一起训，解码器是预训练的）：

![d1ad0a20effac7ec220c24b114d600c](图片\d1ad0a20effac7ec220c24b114d600c.png)

生成结果：

![0c51840f741d2b6f639722b32252dd2](图片\0c51840f741d2b6f639722b32252dd2.png)

---

### 3.4 EX任务

### 1. 描述生成效果

对于每张输入图像，生成了两部分描述：

- **服饰描述**：包括服饰类型、颜色、纹理等。
- **背景描述**：包括背景物体、环境信息等。

生成的综合描述样例如下：

```
"A man wearing a plaid shirt and jeans. The background shows a sunny park with green trees and a bench."
```

![68b5b11c-a108-4699-8c1b-ed7b38f796b1](图片\68b5b11c-a108-4699-8c1b-ed7b38f796b1.png)

效果图
![alt text](图片\任选loss.png)
![alt text](图片\任选accuracy.png)

### 2. 新数据集构建

新数据集中包含了丰富的服饰与背景描述，共计生成描述10000条，一条大概两三秒时间。

### 3. 性能评估

在新数据集上重新训练模型后，与原始模型相比，BLEU、CIDEr等指标均有显著提升，证明了新数据集的有效性。

## 四、分工，总结与感受

总结

算力是我们最焦虑的问题，ex任务有一万条数据要跑，刚好又遇到了期末周，我们就想尽了办法去减少算力需求和浪费，又是减少层数又是砍数据集又是分开训练，好在训练的结果还不错。

感受

如果没用b站的vit课程的话，vit模型我一辈子都写不出来，实在是太复杂了，每个层都能拆成一个个小层，在保证类的非耦合性的时候，还得要求类复用后维度能对上....

vit训到后面Loss函数已经下降到了一个很低的值，可能因为样本太少过拟合了。后面我们一直在想，能不能换掉交叉熵，不过后面还是没换，因为对语言模型的训练的trick不太熟悉，时间也不允许我们这么做。

分工和贡献度

刘逸（39）：网格表示＋trasformer编码器，vit 编码器，transformer解码器

罗苏涵（35）：CNN+GRU，数据处理，测评指标

史博文（26）：报告编写，任选任务，模型训练

