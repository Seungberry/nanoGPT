# nanoGPT

## 项目简介
本项目基于 Andrej Karpathy 的 NanoGPT 开源库，旨在复现并实验中型 GPT 模型的训练与推理过程。NanoGPT 是一个针对 GPT 架构最简化的实现，代码遵循 PyTorch 设计哲学，去除了冗余的抽象，适合深入理解 Transformer 机制及大规模语言模型的预训练逻辑。

---

## 环境要求
* Python 3.8+
* PyTorch 2.0+ (推荐使用以支持 torch.compile)
* numpy, transformers, datasets, tiktoken, wandb, tqdm

## 快速开始

### 1. 数据准备
以 Shakespeare 数据集为例，执行以下命令进行下载并生成训练所需的二进制文件：
```bash
python data/shakespeare_char/prepare.py
```

### 2. 模型训练
使用单卡 GPU 进行训练的示例命令：
```bash
python train.py --dataset=shakespeare_char --device=cuda --compile=True --eval_iters=20 --log_interval=10
```
若在 CPU 环境下运行，请将 `--device` 设置为 `cpu` 并关闭 `--compile`。

### 3. 模型推理
训练生成的权重将保存在 `out/` 目录中。运行 `sample.py` 进行文本生成：
```bash
python sample.py --out_dir=out --start="ROMEO:" --num_samples=5 --max_new_tokens=200
```

## 核心文件说明
* **model.py**: 包含 GPT 类及 Transformer 核心组件（自注意力、多层感知机）的定义。
* **train.py**: 训练主循环，支持混合精度训练（AMP）与分布式数据并行（DDP）。
* **sample.py**: 采样推理脚本，支持从本地 checkpoint 或 GPT-2 预训练模型加载。
* **configurator.py**: 参数解析工具，允许通过命令行或配置文件（.py）修改模型超参数。

## 关键参数说明
* `n_layer`: Transformer 层数（默认 12）
* `n_head`: 注意力头数（默认 12）
* `n_embd`: 嵌入维度（默认 768）
* `block_size`: 最大上下文序列长度（默认 1024）
* `batch_size`: 训练批次大小

## 运行结果
<img width="449" height="555" alt="{6DF9424C-6C6E-40B2-A0CD-F489A2FBD23E}" src="https://github.com/user-attachments/assets/946d0103-2857-454e-bd0e-72528e774992" />

---

## 1. 使用《天龙八部》数据集训练模型


* **数据清洗：** 确保 `tianlong.txt` 采用 UTF-8 编码。建议去除文本中的页码、广告词或重复的章节标题，只保留正文。
* **Tokenization（分词）：** 对于中文，初学者通常建议使用 **Character-level（字符级）**。即将每一个汉字、标点符号作为一个 Token。这能有效避免词表过大导致的显存爆炸。
* **准备脚本：** 运行 `prepare.py` 将文本转换为 `train.bin` 和 `val.bin`。

---

## 2. 核心参数解析与实验建议

在 `config/train_poemtext_char.py` 中，参数的选择直接决定了模型的“智商”和生成风格。可以尝试修改以下关键参数：

| 参数名称 | 含义 | 修改建议与预期效果 |
| :--- | :--- | :--- |
| **`n_layer`** | Transformer 层的数量。 | **实验：** 从 4 增加到 8。模型会拥有更强的逻辑感，能记住更长的人物关系。 |
| **`n_head`** | 多头注意力的头数。 | **实验：** 确保能被 `n_embd` 整除。增加头数有助于模型在同一位置关注不同的语义信息。 |
| **`n_embd`** | 词向量的维度（Embedding size）。 | **实验：** 调大此值（如从 128 升至 384）。模型对汉字的表达会更细腻，但计算量增加。 |
| **`block_size`** | 上下文窗口长度（一次看多少个字）。 | **实验：** 设为 256 或 512。武侠小说描写较长，较大的窗口能避免模型“前言不搭后语”。 |
| **`learning_rate`** | 学习率。 | **实验：** 如果 Loss 震荡，尝试减小 LR；如果 Loss 下降太慢，适度调大。 |



---

## 3. 模型推理过程解析 (`sample.py`)

模型的推理（Inference）本质上是一个自回归（Autoregressive）的过程。简单来说，就是“根据已有的字，预测下一个字”。

---

## 1. 初始化与环境准备
在开始“写文章”之前，模型需要配置好它的“大脑”：
* **加载模型状态**：通过 `init_from` 判断是从本地检查点（`resume`）加载训练好的权重，还是直接调用预训练的 `gpt2` 模型。
* **设置计算精度**：代码中使用了 `bfloat16` 或 `float16`。这是一种低精度计算技术，可以在不显著牺牲准确性的情况下，大幅提升推理速度并减少显存占用。
* **编译优化**：如果 `compile = True`，则会调用 PyTorch 2.0 的编译器对模型进行图形优化，使前向传播更快。

---

## 2. 文本编码 (Encoding)
计算机看不懂文字，它只认识数字。
* **Tokenization**：代码通过 `tiktoken`（GPT-2 默认）或自定义的 `meta.pkl` 将输入的字符串 `start` 转换成一系列整数 ID。
* **Tensor 转换**：将这些 ID 转换成 PyTorch 的张量（Tensor），并移动到 GPU (`cuda`) 上。

---

## 3. 核心循环：自回归生成
这是推理最关键的部分。模型通过调用 `model.generate` 函数执行以下逻辑：


### 1）. 前向传播 (Forward Pass)
模型将当前的 Token 序列输入 Transformer。Transformer 内部的注意力机制（Attention）会分析序列中每个词之间的关系，并为词表中的每一个可能的词算出一个分数（Logits）。

### 2）. 概率缩放 (Temperature & Top-K)
在得到分数后，代码引入了两个关键的超参数来控制“创造力”：
* **Temperature (温度)**：
    * $T < 1.0$：模型变得保守，总是选概率最高的。
    * $T > 1.0$：模型变得更有“创意”（也更乱），增加了低概率词被选中的机会。
* **Top-K 采样**：只从概率最高的前 $k$ 个词中挑选，直接扔掉那些长尾的、不合理的词，防止模型“胡言乱语”。

### 3）. 采样 (Sampling)
模型不再仅仅是找“概率最大的词”，而是根据调整后的概率分布进行**随机采样**。

### 4）. 拼接与迭代
选出下一个 Token 后，将其拼接到原序列的末尾，然后**重新喂给模型**。
> **输入：** "今天天气" -> **预测：** "很"
> **输入：** "今天天气很" -> **预测：** "好"
> 这个过程会一直持续，直到达到 `max_new_tokens` 设定的上限。

---

## 4. 解码与输出 (Decoding)
最后，`decode(y[0].tolist())` 将模型输出的一串数字 ID 重新转换回人类可读的文字，并打印出来。
