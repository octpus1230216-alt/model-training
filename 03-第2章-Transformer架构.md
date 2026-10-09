# 第 2 章 Transformer 架构

> 对应原文：`docs/chapter2/第二章 Transformer架构.md`　|　本章定位：全书最核心的理论基础。讲清"注意力机制"为何取代 RNN、其数学形式与每一步维度变化，并完整搭建出可计算的 Encoder-Decoder Transformer，为第 3 章预训练语言模型与第 5 章动手搭建 LLM 打地基。

> 衔接语：第 1 章我们留下四道未解难题——稀疏性、词序缺失、局部上下文、长距离依赖。本章主角 Transformer 用一个名为"注意力"的运算把它们一次性化解。它的诞生要从 RNN 的失败说起，而理解 RNN 为何失败，正是理解 Transformer 为何成功的前提。读完本章，你会真正明白"为什么现代大模型几乎都长成 Transformer 的样子"。

## 本章目标

- 理解 RNN/LSTM 的两大缺陷，以及 Transformer 为何"完全由注意力机制构成"。
- 能从零推导缩放点积注意力公式 $softmax(QK^T/\sqrt{d_k})V$，说清每个符号与每一步维度变化。
- 区分自注意力、掩码自注意力、多头注意力三种形态及其实现要点。
- 掌握 Encoder/Decoder 内部组件：FNN、Layer Norm、残差连接、Pre-Norm 结构。
- 理解正余弦位置编码公式及其相对位置可表达性的数学推导。
- 能通读完整 Transformer 代码，核对 `(B,T,d)` 的数据流与参数量统计。

---

## 2.1 注意力机制

**为什么先讲注意力，而不是直接讲 Transformer？** 因为 Transformer 的"灵魂"就是注意力，其余组件（前馈网络、归一化、残差）都只是让它更好训练、更深更稳的"辅助结构"。只要把注意力吃透，Transformer 不过是把注意力按一定规则堆叠、拼接起来的工程实现。本节我们遵循"为什么需要它 → 它是什么 → 怎么算 → 怎么验证"的主线，把注意力从直觉一路推到公式与代码。

### 2.1.1 什么是注意力机制

**（1）三类经典神经网络架构**

在引入注意力之前，先回顾处理数据的三种主流网络，它们分别对应不同的数据形状：

- **前馈神经网络（FNN，Feedforward Neural Network）**：数据从输入层单向流到输出层、无循环、以全连接 MLP 为典型，适合固定尺寸输入（如一张图片、一个向量）。直觉上它像一条单行道，信息只往前走。
- **卷积神经网络（CNN）**：用参数量远小于全连接的卷积层提取局部特征，擅长图像这类有空间局部相关的数据。直觉上它像"滑动的放大镜"，每次只看一小块。
- **循环神经网络（RNN）**：带环与自重复、能把历史信息作为输入的神经网络，专为序列/时序数据设计。直觉上它像"边读边记"的读书笔记员，每读一个新词，都把前面的记忆带进来。

文本正是一串序列，因此 RNN（及其变体 LSTM）长期是 NLP 霸主——第 1 章开创预训练思想的 ELMo 用的就是双向 LSTM。在注意力机制出世之前，RNN/LSTM 是 NLP 领域当之无愧的霸主。

**（2）为什么是 RNN，RNN 又为什么不够**

RNN 能建模时序，这让它一度无可替代，但它有两点难以弥补的缺陷（这是理解 Transformer 的钥匙）：

1. **无法并行计算**——序列须逐词输入、依序计算，GPU 那成千上万个核心只能干等前一个词算完。序列越长，这种"串行"越浪费硬件，时间成本极高。
2. **难以捕捉长序列相关关系**——距离越远的输入越难被捕捉；且 RNN 需要把整条序列读入内存依次计算，限制了序列长度。虽然 LSTM 用"门机制"（遗忘门、输入门、输出门）缓解了不少，但对较远距离的相关关系（例如一句话开头的主语和结尾的动词）捕捉依旧不理想。

> 一句话总结 RNN 的悖论：它为了"记住顺序"而串行，却因串行而慢、因串行而难以看到远处。Transformer 要同时解决"慢"和"看不远"两件事。

**（补充）三种架构的取舍，决定了注意力的出场时机。** 回顾开头的三类网络：FNN 不能处理变长序列，CNN 擅长局部但难建模长距全局依赖，RNN 能建模序列却牺牲了并行。注意力机制的巧妙在于，它既像 CNN 那样"一次性看到整张画"（全序列并行），又像 RNN 那样"每个位置都能参考任意远处的信息"（任意两位置直接相连），同时还摆脱了 RNN 的串行锁链。换句话说，注意力是把 CNN 的"全局感受野"与 RNN 的"序列建模能力"结合，并加上了 GPU 友好的并行性——这正是它能成为大模型基石的底层原因。读者在后续第 5 章亲手搭 LLaMA 时会再次确认：去掉 RNN、只保留注意力，模型反而更快更强。

**Transformer** 由 Vaswani 等学者在 2017 年提出。他们参考了在计算机视觉领域被提出的**注意力机制（Attention）**：让模型在处理某位置时"回头看"序列其他位置并按需加权汇聚信息的机制，直觉上如读句时目光自动聚焦到最相关的几个词。Transformer 创新性地用"完全由注意力构成的神经网络"替代了 RNN，成为 LLM 的鼻祖与核心架构。值得注意：注意力虽在 NLP 被发扬光大，但它最早源于计算机视觉——优秀的思想会跨领域流动，这正是研究者需要开阔视野的原因。

**（3）三要素与直觉案例**

注意力的三个核心变量是：

- **Query（查询值，Q）**：代表"当前想要查询什么信息"的向量，直觉上像你心里提出的问题。
- **Key（键值，K）**：代表"每个位置能提供什么信息"的向量，直觉上像字典的目录词条。
- **Value（真值，V）**：代表"每个位置实际携带的内容"的向量，直觉上像目录对应的正文。

一个新闻报道案例可帮助理解：想从一篇报道里找"报道时间"，Q 类似"时间/日期"的向量（实际是稠密向量），K 与 V 是整个文本；先拿 Q 与每个 K 算相似度，得到**注意力权重**（每个 token 应分配的注意力大小），再把权重与 V 加权求和，即得到"从时间视角看整篇报道"的结果。一句话定义：**通过 Query 与 Key 的相关性为 Value 加权求和，拟合序列中每个词同其他词的相关关系**。

> 直觉延伸：把 Q/K/V 想象成"查字典"——你拿着问题（Query）翻目录（Key），找到相关条目后取出内容（Value）。不同之处在于，这里不是精确匹配，而是按相关程度"软性"地同时取多个 Value 并加权混合。这正是注意力比硬查字典更强大的地方：它允许"既看 apple 又看 banana，按相关度各取一点"。

### 2.1.2 深入理解注意力机制（公式推导全流程）

本节我们像剥洋葱一样，从最朴素的"查字典加权"逐步推导出标准注意力公式。每一步都保留直观含义，避免直接抛出最终式让读者摸不着头脑。这也是"自洽"红线的体现：公式不是天降的，而是每一步都有理由。

**第一步：字典类比。** 把 Key=键、Value=值看作字典 `{"apple":10, "banana":5, "chair":2}`。精确匹配时 Q="apple" 直接命中；但 Q 若是"fruit"这类含多个 Key 的概念，应同时匹配 apple 与 banana，按权重组合 Value：

$$value = 0.6 \times 10 + 0.4 \times 5 + 0 \times 2 = 8$$

这里的 0.6、0.4、0 即**注意力分数**——为查询 Q 应赋予每个 Key 多少注意力。注意"fruit"既匹配 apple 又匹配 banana，恰说明注意力是"软匹配"而非"硬查找"；chair 得 0 分说明它与"fruit"无关，被自然排除。

**第二步：用点积度量相似度。** 由第 1 章，合理训练的词向量能表征语义，可用点积度量相似：

$$v \cdot w = \sum_{i} v_i w_i$$

- 符号：$v,w$ 为两个词向量；$v_i,w_i$ 为第 $i$ 维分量；$\sum_i$ 对全部维度求和。
- 判定：语义相似的两词点积应 $>0$（方向相近），不相似应 $<0$（方向相背）。故 Key 与 Query 越相关，点积越大，应赋越大权重。

> 为什么用点积而不是欧氏距离？点积带有"方向一致则大"的几何含义，且能写成高效的矩阵乘法（一次算完所有对），这正是后面能做并行、能上 GPU 的根本。

**第三步：Query 与所有 Key 的相似度（矩阵化）。** 设 Q 为 "fruit" 对应向量 $q$，所有 Key 堆叠成矩阵 $K=[v_{apple}\ v_{banana}\ v_{chair}]$：

$$x = q K^T$$

- 符号：$K$ 为所有 Key 词向量堆叠矩阵（每行一个 Key）；$x$ 为 $q$ 与每个 Key 的点积（由矩阵乘法定义）。一次性算完所有 Key，正是并行化的雏形——一句话求出"fruit 对 apple/banana/chair 分别多相关"。

**第四步：Softmax 归一化为权重。**

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$

- 符号：$x_i$ 第 $i$ 位相似度；$e^{x_i}$ 指数化（放大幅度差）；分母 $\sum_j e^{x_j}$ 为所有位之和。作用：既反映相似程度，又使权重和为 1，可直接当"混合比例"。例如 $[0.6,0.4,0]$ 就是归一化后的样子。

**第五步：单 Query、标量 Value 的基本公式。**

$$attention(Q,K,V) = \text{softmax}(q K^T) v$$

此时 $V$ 是标量列表 $v$，加权求和得到单个数值（即第一步算出的 8）。

**第六步：向量化——多 Query、$d_v$ 维 Value。** 把多个 Query 堆叠成矩阵 $Q$，Value 也变为 $d_v$ 维向量，一次算整批：

$$attention(Q,K,V) = \text{softmax}(Q K^T) V$$

**第七步：缩放点积（最终标准公式）。** 若 $Q,K$ 维度 $d_k$ 较大，$QK^T$ 的值会很大，经 Softmax 后梯度趋于陡峭、训练不稳，需缩放：

$$attention(Q,K,V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

- **符号总表**：

| 符号 | 含义 |
| --- | --- |
| $Q$ | 查询矩阵（多个 Query 词向量堆叠），形状 $(..., T_q, d_k)$ |
| $K$ | 键矩阵（多个 Key 词向量堆叠），形状 $(..., T_k, d_k)$ |
| $V$ | 值矩阵（多个 Value 词向量堆叠），形状 $(..., T_k, d_v)$ |
| $d_k$ | Key（即 Query）向量维度 |
| $d_v$ | Value 向量维度 |
| $\sqrt{d_k}$ | 缩放因子（Scaled Dot-Product Attention 的由来） |
| $QK^T$ | Query 与所有 Key 的点积相似度矩阵，形状 $(..., T_q, T_k)$ |
| $\text{softmax}(\cdot)$ | 沿末维归一化为和 1 的注意力权重 |
| 与 $V$ 相乘 | 按权重对 Value 加权求和 |

> 缩放的直觉：点积的方差随 $d_k$ 线性增长，数值过大使 Softmax 进入饱和区（输出接近 one-hot），梯度趋零、训练困难；除以 $\sqrt{d_k}$ 把方差拉回 1 附近，稳定梯度。这就是**注意力机制的核心计算公式**——请把它刻进记忆，全书的注意力都长这样。

**（补充）从公式看"注意力到底在算什么"。** 把整条公式连起来读：`QKᵀ` 得到"每个位置对每个位置的相似度表"，`softmax` 把每一行变成"和为 1 的关注比例"，再拿这个比例去给 `V` 做加权求和，得到的就是"融合了全序列信息的、属于当前位置的新的表示"。注意一个易被忽略的细节：注意力**不改变信息总量，只重新分配权重**——它不创造新特征，而是把 Value 按相关度重新混合。因此单独的注意力层表达能力有限，必须配合后面的前馈网络（做非线性变换、创造新特征）才完整。这也是为什么 Transformer 里"注意力 + 前馈"总是成对出现，缺一不可。

**（补充）维度约定的统一约定。** 为便于后续阅读，本书统一：$(B,T,d)$ 表示批大小、序列长度、模型维度；自注意力时 $Q,K,V$ 三者序列长度常相等为 $T$；交叉注意力时 $Q$ 长度可与 $K,V$ 不同（如 Decoder 长度为 $T_q$、Encoder 输出长度为 $T_k$）。记不住时，回看本节第七步的符号总表与各节维度变化链即可。

### 2.1.3 注意力机制的实现

把公式落到代码只需几行。下面的 `attention` 是最朴素版本，假设 `query/key/value` 已是经投影后的矩阵（对应公式里的 $Q,K,V$）。

```python
'''注意力计算函数：实现 softmax(QK^T/√d_k)V'''
def attention(query, key, value, dropout=None):
    # 获取键向量维度 d_k（Q/K 同维，所以取 query 最后一维）
    d_k = query.size(-1)
    # 计算 Q·K^T 并除以根号 d_k 缩放
    # 维度：(B, T, d_k) · (B, d_k, T) -> (B, T, T)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 沿最后一维（每个 query 对所有 key）做 Softmax，得到注意力权重
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)            # 训练时随机置零部分权重，防过拟合
    # 权重与 V 加权求和：(B, T, T) · (B, T, d_v) -> (B, T, d_v)
    return torch.matmul(p_attn, value), p_attn
```

要点：输入 `q,k,v` 是已投影矩阵；`key.transpose(-2,-1)` 即 $K^T$；返回 `(加权结果, 权重)`，权重可用于可视化"模型在看哪"。务必记住：`scores` 形状 `(B,T,T)` 中第 $i$ 行第 $j$ 列，表示"第 $i$ 个位置对第 $j$ 个位置的注意力大小"——这是理解后面一切的基础。

### 2.1.4 自注意力（Self-Attention）

注意力本质是"对两段序列的元素依次算相似度、加权分配注意力"，这两段序列即 $Q,K,V$ 的来源。**经典（交叉）注意力**中 $Q$ 来自一个序列、$K,V$ 来自另一个——例如 Transformer 的 Decoder 用自身 $Q$ 去查 Encoder 输出的 $K,V$，从而融合"已生成的历史"与"原文信息"。

**自注意力（Self-Attention）** 定义为：计算**同一序列**中每个元素对其他元素的注意力分布，即 $Q,K,V$ **都由同一输入经不同参数矩阵得到**。在 Encoder 中，$Q,K,V$ 分别由输入对参数矩阵 $W_q,W_k,W_v$ 做积得到，从而建模"每个 token 对全部 token 的关系"。代码上仅一行：

```python
# attention 为上文定义函数；x 为同一输入 → Q=K=V 同源即自注意力
attention(x, x, x)
```

> 关键洞察：自注意力让"我喜欢你"中"我"能直接注意到"你"、"喜欢"，且这种关注是一次性、全序列并行的。任意两个位置（无论隔多远）之间都有一条"直接通路"，距离不再是障碍——这正一举化解了第 1 章的"长距离依赖"与"并行"两难。

**（补充）自注意力 vs 交叉注意力：何时用哪个？** 一句话口诀："自己看自己用自注意力，看别人用交叉注意力。"Encoder 内部、Decoder 的第一层都是"自己序列内部互看"，所以用自注意力；Decoder 的第二层要"参考原文"，Q 来自已生成的部分、K/V 来自 Encoder 输出，这就是交叉注意力。记住这个口诀，后面读 BERT（只有 Encoder，全程自注意力）和 GPT（只有 Decoder，自注意力+掩码自注意力）的架构时会非常轻松——它们只不过是把这两种注意力按不同方式组合罢了。

### 2.1.5 掩码自注意力（Mask Self-Attention）

**（1）动机。** Transformer 也以"类 n-gram 语言模型"方式学习：根据前文预测下一 token（如 `【BOS】I like you【EOS】` 逐位预测）。但训练时若真串行（先算 Step1 才能算 Step2），就违背了 Transformer 可并行的优势。

**（2）掩码方案。** 生成一串上三角掩码遮蔽未来信息，让所有位置**并行**输入模型，各自只对未遮蔽 token 预测下一词：

```
<BOS> 【MASK】【MASK】【MASK】【MASK】
<BOS>    I   【MASK】 【MASK】【MASK】
<BOS>    I     like  【MASK】【MASK】
<BOS>    I     like    you  【MASK】
<BOS>    I     like    you   </EOS>
```

每一行模型仍只看到前面的 token、预测下一个，但这些行可一起并行喂入，模型对每行根据未遮蔽 token 预测下一词，从而实现"并行的语言模型"。

**（3）掩码矩阵与维度。** 掩码是与序列等长的**上三角矩阵**。输入为 `(batch_size, seq_len, hidden_size)` 时，Mask 维度一般为 `(1, seq_len, seq_len)`（靠广播作用于整批不同样本）。

**（4）代码实现。**

```python
# 生成上三角掩码：(1, L, L) 全 -inf，再取上三角（主对角线及以上）
mask = torch.full((1, args.max_seq_len, args.max_seq_len), float("-inf"))
mask = torch.triu(mask, diagonal=1)     # diagonal=1 → 主对角及以上为 -inf，下三角为 0
```

```python
# 应用到注意力分数
scores = scores + mask[:, :seqlen, :seqlen]            # 上三角加 -inf
scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # -inf 经 softmax 严格变 0 → 忽略未来
```

原理：上三角区域分数变 $-inf$，下三角不变；Softmax 后 $-inf$ 被置 0，从而实现注意力遮蔽。这一步是"并行语言模型"得以成立的关键——同一矩阵同时算所有位置的预测，却互不泄露未来。

> 为什么用 `-inf` 而不是 0？若用 0，Softmax 仍会给该位置分配一点点概率（权重不为 0），未来信息会"漏"进来；用 `-inf` 经 `exp(-inf)=0`，权重被彻底清零，模型在数学上严格看不到未来。

### 2.1.6 多头注意力（Multi-Head Attention）

**（1）动机。** 一次注意力只能拟合一种相关关系（一种"关注视角"）。但语言信息是多层次的：一个头可能关注语法依存，另一个头可能关注语义角色，再一个头可能关注指代。方案：对同语料做多次注意力、各拟合不同关系，再拼接输出，更全面刻画语言。原论文可视化也证实：不同头确实捕捉不同层次信息。

**（2）公式。**

$$\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h}) W^O$$
$$\text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)$$

- 符号：$h$ 头数；$\mathrm{head}_i$ 第 $i$ 头输出；$W^Q_i,W^K_i,W^V_i$ 第 $i$ 头各自的 Q/K/V 投影；$W^O$ 输出投影，把 $h\cdot d_{head}$ 维映射回 $d_{model}$。每个头在"降维后的子空间"里算注意力，计算量并不随头数线性暴涨。

**（3）两种实现思路。** 朴素做法用 $n$ 组独立矩阵，各做一遍注意力再拼——但时空复杂度高。本文用**三个组合矩阵**代替 $n$ 个参数矩阵：因为"矩阵内积再拼接 ≡ 拼接矩阵再内积"，用一次大线性层即可并行算出所有头，效率更高。

**（4）完整实现。**

```python
import torch.nn as nn
import torch, math

class MultiHeadAttention(nn.Module):
    def __init__(self, args: ModelArgs, is_causal=False):
        super().__init__()
        assert args.dim % args.n_heads == 0          # 隐藏维度须被头数整除
        self.head_dim = args.dim // args.n_heads      # 每头维度 = 模型维 / 头数
        self.n_heads = args.n_heads
        # 用组合矩阵同时算所有头的 Q/K/V：(n_embd -> n_heads*head_dim)
        self.wq = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)  # 输出投影
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.is_causal = is_causal
        if is_causal:                                # 因果掩码，多一维以便广播
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)       # 注册为缓冲区，不更新梯度、随模型移动

    def forward(self, q, k, v):
        bsz, seqlen, _ = q.shape                     # (B, T, n_embd)
        # 线性投影：(B,T,n_embd) -> (B,T,n_heads*head_dim)
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)
        # 拆多头：(B,T,n_heads,head_dim) -> 交换为 (B,n_heads,T,head_dim)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        # 注意力分数：(B,nh,T,hd)·(B,nh,hd,T) -> (B,nh,T,T)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if self.is_causal:
            scores = scores + self.mask[:, :, :seqlen, :seqlen]   # 加因果掩码
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.attn_dropout(scores)
        # 加权求和：(B,nh,T,T)·(B,nh,T,hd) -> (B,nh,T,hd)
        output = torch.matmul(scores, xv)
        # 合并头：(B,nh,T,hd) -> (B,T,n_heads*hd)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)                     # 投影回 (B,T,dim)
        output = self.resid_dropout(output)
        return output
```

**（5）维度变化链。**

| 步骤 | 形状 |
| --- | --- |
| 输入 q/k/v | `(B, T, n_embd)` |
| wq/wk/wv 投影 | `(B, T, n_heads*head_dim)` |
| view + transpose | `(B, n_heads, T, head_dim)` |
| scores = xq·xkᵀ/√d | `(B, n_heads, T, T)` |
| + mask（因果） | `(B, n_heads, T, T)` |
| softmax + dropout | `(B, n_heads, T, T)` |
| output = scores·xv | `(B, n_heads, T, head_dim)` |
| 合并头 + wo | `(B, T, dim)` |

工程细节：`view` 必须"先按 B×T×n_head×C/n_head 展开再互换 1、2 维"；`contiguous()` 因先 transpose 再 view 需重开内存（否则 PyTorch 报错）；`register_buffer` 使掩码随模型 `.to(device)` 移动但不参与梯度；因果掩码比 §2.1.5 多一维 `(1,1,L,L)` 以广播到 `(B,nh,T,T)`。

**（补充）头数为什么要整除模型维度？** `assert args.dim % args.n_heads == 0` 不是随意的限制：多头注意力把模型维度 `dim` 均匀切成 `n_heads` 份，每份 `head_dim = dim / n_heads`。若不能整除，"切份"会留下不完整的一截，无法用整齐的矩阵运算表达。实践中常取 `n_heads` 为 2 的幂（如 8、12、32），`dim` 取 512、768、4096 等可被整除的值。这个约束在后面第 5 章搭 LLaMA 时同样适用，请提前记住。

**（补充）多头到底带来了什么？** 单个注意力头只能学到一种"关注模式"，而不同头在不同子空间里独立计算，相当于模型同时用多种视角读同一句话：有的头关注相邻词（局部），有的头关注主语与动词（长距），有的头关注标点配对。最终 `Concat + Wᵒ` 把这些视角融合回统一表示。这解释了为什么现代大模型普遍用很多头（如 32、64）——头越多，并行的"关注视角"越丰富，但每个头的维度变窄，计算总量基本不变，这正是组合矩阵实现省时空的来源。

### 2.1.7 如何验证你真的懂了注意力

学完本节，请用三句话自检（这也是后面章节的基石）：① 注意力用一个 `(B,T,T)` 的权重矩阵表达"谁看谁"；② 缩放 $\sqrt{d_k}$ 是为让梯度稳定；③ 自注意力 Q=K=V 同源、掩码自注意力用 `-inf` 屏蔽未来、多头用组合矩阵并行再拼接。若这三句能脱口而出，本节达标。

---

## 2.2 Encoder-Decoder

> 过渡：上面我们把"注意力"这只发动机造好了。但发动机要装进车里——Transformer 中，使用注意力的两个核心组件是 **Encoder（编码器）** 和 **Decoder（解码器）**。后续基于 Transformer 的预训练模型，几乎都是对 Encoder-Decoder 这一部分做增删：只留 Encoder 得 BERT，只留 Decoder 得 GPT。所以理解下面三块经典组件，等于理解了所有主流 LLM 的"积木"。

**（补充）为什么是"编码器 + 解码器"这个组合？** 编码器的职责是把"已知道的输入"压缩成富含双向上下文的表示（它看得到整句话的前后），解码器的职责是"根据已知信息一步步生成输出"（它只能看已生成的部分，不能偷看未来）。这种"理解输入 + 自回归生成输出"的分工，天然契合机器翻译、问答、摘要等几乎所有生成类任务。即便后来的 GPT 只保留解码器、BERT 只保留编码器，其内部的子层结构（自注意力、前馈、归一化、残差）仍是本节要讲的内容——所以这一节是"一通百通"的关键。

### 2.2.1 Seq2Seq 模型

**Seq2Seq（Sequence-to-Sequence）**：输入序列 $input=(x_1,...,x_n)$ 输出可能不等长序列 $output=(y_1,...,y_m)$ 的模型。符号：$x_i$ 输入第 $i$ 个 token，$y_j$ 输出第 $j$ 个 token，$n,m$ 一般不等。一个核心论断：**几乎所有 NLP 任务都可视为 Seq2Seq**——文本分类输出长度 1（$m=1$），词性标注输出与输入等长（$m=n$）。经典实例机器翻译：输入"今天天气真好"→输出"Today is a good day."。编码把输入序列变身为表征语义的向量/矩阵（可理解为"更复杂的词向量"），解码再把它变回目标序列。Transformer 由 Encoder 与 Decoder 组成，各含 6 层（原论文 N=6），Encoder 顶层输出喂给 Decoder 每一层。

**（补充）"几乎一切都是 Seq2Seq"对理解 LLM 为何重要？** 这个论断把五花八门的 NLP 任务统一到一个框架里：只要能设计"输入序列"和"输出序列"的映射，就能用同一套 Transformer 架构去解决。它解释了为什么一个预训练好的大模型可以通过"提示词（prompt）"被改造成分类器、翻译器、问答器——因为对模型而言，它们都只是"给一段序列、补一段序列"的游戏。这也是第 4 章讲 LLM"上下文学习"与"指令理解"时的底层逻辑：任务边界被序列到序列的生成能力抹平了。读者在此先建立这个统一视角，后面会省力很多。

### 2.2.2 前馈神经网络（FNN / MLP）

每个 Encoder Layer 含一个注意力层与一个前馈网络。FNN 由**两层线性变换 + 中间 ReLU 激活 + Dropout** 组成：

```python
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)   # (B,T,d) -> (B,T,d_ff)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # (B,T,d_ff) -> (B,T,d)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # 先线性+ReLU 升维非线性，再线性降回；dropout 防过拟合
        return self.dropout(self.w2(F.relu(self.w1(x))))   # 维度 (B,T,d)->(B,T,d_ff)->(B,T,d)
```

注：本仓库 `EncoderLayer` 调用 `MLP(args.dim, args.dim, args.dropout)`，即 `hidden_dim=dim`。**Dropout 只在训练开启、推理关闭**，所以许多结构图不画它。前馈网络把每个位置独立地做一次非线性变换，与注意力（跨位置交互）互补——一个负责"想"，一个负责"联"。

### 2.2.3 层归一化（Layer Norm）

**为什么要归一化**：深层网络中，前层参数变化会改变后层输入分布（internal covariate shift），造成预测误差，需把每层输入归一化为近似标准正态分布。两种主流：**批归一化（Batch Norm）**：在一个 mini-batch 的样本维上统计均值方差，与**层归一化（Layer Norm）**：在每个样本的特征维上统计均值方差。Batch Norm 有四缺陷：① 小 batch 失效（均值方差不能反映全局）；② 对 RNN 无意义（同分布假设破）；③ 测试时变长句无统计量；④ 开销大。故 Transformer 用 Layer Norm——归一化公式与 Batch Norm 相同，只是**统计维度不同**（每个样本自身），不受 batch 大小与序列长度影响。

Batch Norm 三步：

$$\mu_j = \frac{1}{m}\sum^{m}_{i=1}Z_j^{i},\quad \sigma^2 = \frac{1}{m}\sum^{m}_{i=1}(Z_j^i - \mu_j)^2,\quad \widetilde{Z_j} = \frac{Z_j - \mu_j}{\sqrt{\sigma^2 + \epsilon}}$$

- 符号：$Z_j^i$ 样本 $i$ 第 $j$ 维值；$m$ batch 大小；$\mu_j,\sigma^2$ 第 $j$ 维 batch 均值/方差；$\widetilde{Z_j}$ 归一化结果；$\epsilon$ 防除零极小量。

Layer Norm 实现（在末维统计）：

```python
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))   # 可学习增益（还原表达力）
        self.b_2 = nn.Parameter(torch.zeros(features))  # 可学习偏置
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)                 # (B,T,1) 沿特征维求均值
        std = x.std(-1, keepdim=True)                   # (B,T,1) 沿特征维求标准差
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

要点：`mean(-1)`/`std(-1)` 在特征维统计，使每个样本每位置独立归一化；`a_2`、`b_2` 是可学习仿射参数（初值 1、0），避免归一化"削平"模型表达能力；`eps=1e-6` 即公式 $\epsilon$。

### 2.2.4 残差连接（Residual Connection）

**残差连接**：下一层输入不仅是上层输出，还包含上层输入（恒等映射分支），让最底层信息直达最高层、高层专注学残差，避免深模型退化。Encoder 中两个子层均为 Pre-Norm 形式：

$$x = x + \mathrm{MultiHeadSelfAttention}(\mathrm{LayerNorm}(x))$$
$$output = x + \mathrm{FNN}(\mathrm{LayerNorm}(x))$$

```python
h = x + self.attention.forward(self.attention_norm(x))   # 自注意力 + 残差（恒等支路 x）
out = h + self.feed_forward.forward(self.fnn_norm(h))     # 前馈 + 残差
```

> 为什么需要残差：深层网络梯度反向传播易消失/爆炸，残差支路提供"高速公路"，保证梯度能直抵浅层，这也是 Transformer 能堆到几十上百层、进而扩大参数量的基础。

### 2.2.5 Encoder

Encoder 由 N 个 Encoder Layer 组成，每层含一个（无掩码）自注意力与一个 FNN：

```python
class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention_norm = LayerNorm(args.n_embd)
        self.attention = MultiHeadAttention(args, is_causal=False)  # Encoder 无需掩码
        self.fnn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)
    def forward(self, x):
        norm_x = self.attention_norm(x)
        h = x + self.attention.forward(norm_x, norm_x, norm_x)      # Q=K=V 同源=自注意力
        out = h + self.feed_forward.forward(self.fnn_norm(h))
        return out

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)            # N 层后再加一次 Layer Norm
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
```

要点：Encoder 注意力 `is_causal=False`（可见全序列）；`attention(norm_x,norm_x,norm_x)` 即自注意力；N 层后追加 Layer Norm。"通过 Encoder 的输出，就是输入编码之后的结果"——它把整句压缩为富含双向上下文的表示。

### 2.2.6 Decoder

Decoder 每层由**两个注意力层 + 一个 FNN** 组成：① 掩码自注意力（`is_causal=True`，只看历史）；② 交叉注意力（用本层输出作 Q，Encoder 输出作 K,V）；③ FNN。

```python
class DecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention_norm_1 = LayerNorm(args.n_embd)
        self.mask_attention = MultiHeadAttention(args, is_causal=True)  # 掩码自注意力
        self.attention_norm_2 = LayerNorm(args.n_embd)
        self.attention = MultiHeadAttention(args, is_causal=False)      # 交叉注意力
        self.ffn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)
    def forward(self, x, enc_out):
        norm_x = self.attention_norm_1(x)
        x = x + self.mask_attention.forward(norm_x, norm_x, norm_x)     # 掩码自注意力
        norm_x = self.attention_norm_2(x)
        h = x + self.attention.forward(norm_x, enc_out, enc_out)        # Q=Decoder, K/V=Encoder
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)
    def forward(self, x, enc_out):
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)
```

要点：每层三个 LayerNorm（掩码前、交叉前、FFN 前）；交叉注意力 `attention(norm_x, enc_out, enc_out)` 即 Q 来自 Decoder、K/V 来自 Encoder——这正是"翻译时边生成边参考原文"的数学实现。

> 过渡：Encoder/Decoder 搭好即完成核心。把它们与 Embedding、位置编码拼接，就得到完整 Transformer——下一步我们补齐"输入怎么变向量"与"顺序怎么补回来"两件事。

---

## 2.3 搭建一个 Transformer

### 2.3.1 Embedding 层

**Embedding 层**：一个存固定大小词表的嵌入向量查找表，把 token 转为向量。前置需 **tokenizer（分词器）**：把自然语言切分为 token 并映射为固定 index 的组件（详见第 1 章子词切分）。输入形状 `(batch_size, seq_len, 1)`（值为 index），内部是可训练权重矩阵 `(vocab_size, embedding_dim)`，输出 `(batch_size, seq_len, embedding_dim)`。实现直接调用：

```python
self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)   # 词表大小 × 模型维度
```

> 说明：实际大模型分词比"词/字"更细（子词切分，见第 1 章），词表常达数万至数十万；本章示例用 `bert-base-chinese` 仅作演示。Embedding 层的权重就是第 1 章说的"更聪明的词向量"，而它最终会由注意力与训练共同优化。

**（补充）Embedding 与位置编码的先后关系。** 注意数据流：先 `wte`（词向量），再 `wpe`（加位置），二者是**逐位相加**而非拼接。相加的好处是维度不变（仍是 `(B,T,d)`），且不增加后续计算量；代价是位置信息与语义信息混在同一向量里——但正余弦编码的可加性（相对位置可由绝对位置线性表出）保证了这种混合不会破坏顺序信息。这也是为什么前面 §2.3.2 花大力气证明"相对位置可表达性"：它保证了"加"这个操作在数学上是安全的。若改用拼接（如早期某些模型把位置拼到词向量后面），维度会加倍、参数量上升，因此现代Transformer 普遍选择相加。

### 2.3.2 位置编码（Positional Encoding）

**（1）为什么需要。** 注意力机制中每个 token 对其他位置"一视同仁"，导致"我喜欢你"与"你喜欢我"结果相同——**序列顺序信息丢失**。RNN/LSTM 依序递归天然保留顺序，而注意力需显式注入位置。直觉：注意力是"无序集合上的运算"，必须有人告诉它"谁在第几位"。

**（2）正余弦（Sinusoidal）绝对位置编码。**

$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

- 符号：$pos$ 词在句中的位置；$2i,2i+1$ 偶数/奇数维度索引（奇偶用不同函数，使每个维度编码不同频率）；$i$ 维度对索引；$d_{model}$ 模型维度；底数 10000 为实验结果（不是理论推导值）。

**（3）手算示例（$d_{model}=4$，句 "I like to code"）。** 原词向量矩阵 $x$ 每行一词，加 PE 后（偶数维 `sin`、奇数维 `cos`，代码验证用 `n=100`）：

$$x_{PE} = x + PE,\quad PE_{k,2i}=\sin\frac{k}{10000^{2i/d}},\ PE_{k,2i+1}=\cos\frac{k}{10000^{2i/d}}$$

位置 0 行全为 $\sin0=0,\cos0=1$；位置 1 行约 $\sin1\approx0.84,\cos1\approx0.54$，与原文 numpy 输出一致。把 PE 直接与词向量逐位相加，即把"位置信息"融进"语义向量"。

**（4）两大好处。** ① **可外推到更长句子**：公式法能为训练时未见过的第 21 位直接计算 PE，无需重新训练；② **易算相对位置**：固定间距 $k$ 时，$PE(pos+k)$ 可由 $PE(pos)$ 线性表出（由 $\sin(A+B)$、$\cos(A+B)$ 和角公式）。

**（5）相对位置的数学推导（完整复现）。** 原始 Transformer Embedding 无位置信息，交换任意两 token 函数值不变：

$$f(\cdots,x_m,\cdots,x_n,\cdots)=f(\cdots,x_n,\cdots,x_m,\cdots)$$

期望加位置向量 $p_m,p_n$ 后：

$$\tilde{f}(\cdots)=f(\cdots,x_m+p_m,\cdots,x_n+p_n,\cdots)$$

在 $m,n$ 处泰勒展开，关键项是含交叉 Hessian 的 $p_m^\top \mathcal{H} p_n$（只有它依赖两个位置）。希望它只表达相对位置：

$$p_m^\top \mathcal{H} p_n = g(m-n)$$

设 $\mathcal{H}=I$，则 $p_m^\top p_n = \langle p_m,p_n\rangle = g(m-n)$。把向量视复数、用 $\langle p_m,p_n\rangle=\mathrm{Re}[p_m p_n^*]$ 与 $p_m p_n^*=q_{m-n}$，解二维情形：

$$p_m = e^{im\theta} \Leftrightarrow \begin{pmatrix}\cos m\theta \\ \sin m\theta\end{pmatrix}$$

由内积线性叠加，高维偶数维即多个二维编码组合，取 $\theta_i=10000^{-2i/d}$（使 $|m-n|$ 增大时内积趋零，base=10000 为实验值）即得上式的编码。即便 $\mathcal{H}$ 非单位阵（视作对角），经积化和差仍出现 $\cos(m-n)\theta_i$ 项，证明**仍能表达相对位置**。

> 这一段推导的"所以然"：它证明了正余弦编码不是拍脑袋定的——它恰好能让"两位置的内积只取决于它们的相对距离"，这正是注意力机制需要的"顺序感"。

**（6）实现。**

```python
class PositionalEncoding(nn.Module):
    def __init__(self, args):
        super().__init__()
        pe = torch.zeros(args.block_size, args.n_embd)
        position = torch.arange(0, args.block_size).unsqueeze(1)        # (L,1)
        div_term = torch.exp(
            torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd)
        )                                                                 # = 10000^{-2i/d}
        pe[:, 0::2] = torch.sin(position * div_term)                    # 偶数维 sin
        pe[:, 1::2] = torch.cos(position * div_term)                    # 奇数维 cos
        pe = pe.unsqueeze(0)                                            # (1, L, d)
        self.register_buffer("pe", pe)                                  # 不参与训练、随模型迁移
    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)           # 加到 Embedding 上
        return x
```

要点：`div_term` 用 `exp/log` 表达 $10000^{-2i/d}$ 避免数值溢出；`register_buffer` 使 PE 固定但随模型移动；`requires_grad_(False)` 显式关闭梯度。

### 2.3.3 一个完整的 Transformer

**（1）整体数据流。** 经 tokenizer 的输出 → Embedding + 位置编码 → N 个 Encoder 与 N 个 Decoder（原模型 N=6）→ 线性层 + Softmax 得最终输出。这也是"为什么→是什么→怎么做"的终局：输入被表示、被注入位置、被双向编码、被自回归解码、被投影回词表。

**（2）Post-Norm vs Pre-Norm。** 原论文图 LayerNorm 在子层**后**（Post-Norm），但发布源码放在子层**前**（Pre-Norm）。**本书实现采用 Pre-Norm**，因现代 LLM 多用它使 loss 更稳定、深层更易训练。这也是为什么前面 EncoderLayer/DecoderLayer 里都是"先 LayerNorm 再子层、再加残差"。

**（3）完整模型实现。**

```python
class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        assert args.vocab_size is not None and args.block_size is not None
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(args.vocab_size, args.n_embd),   # 词 Embedding
            wpe = PositionalEncoding(args),                     # 位置编码
            drop = nn.Dropout(args.dropout),
            encoder = Encoder(args),
            decoder = Decoder(args),
        ))
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)  # 输出投影到词表
        self.apply(self._init_weights)
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=False):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()     # 可选减去 embedding 参数
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 权重 ~ N(0,0.02^2)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()                                       # (B, T) token index
        assert t <= self.args.block_size
        tok_emb = self.transformer.wte(idx)                    # (B,T,n_embd)
        x = self.transformer.drop(self.transformer.wpe(tok_emb))  # +位置编码+dropout
        enc_out = self.transformer.encoder(x)                  # (B,T,n_embd)
        x = self.transformer.decoder(x, enc_out)               # (B,T,n_embd)
        if targets is not None:                                # 训练阶段
            logits = self.lm_head(x)                           # (B,T,vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        else:                                                  # 推理阶段（只取最后一位预测）
            logits = self.lm_head(x[:, [-1], :])               # (B,1,vocab_size)
            loss = None
        return logits, loss
```

**（4）前向维度链。**

| 步骤 | 张量 | 形状 |
| --- | --- | --- |
| 输入 | `idx` | `(b, t)` |
| Embedding | `tok_emb` | `(b, t, n_embd)` |
| + 位置编码 | `pos_emb` | `(b, t, n_embd)` |
| Dropout | `x` | `(b, t, n_embd)` |
| Encoder | `enc_out` | `(b, t, n_embd)` |
| Decoder | `x` | `(b, t, n_embd)` |
| lm_head（训练） | `logits` | `(b, t, vocab_size)` |
| lm_head（推理） | `logits` | `(b, 1, vocab_size)` |

损失用 `F.cross_entropy(..., ignore_index=-1)` 忽略 padding；`t<=block_size` 断言防超长。源码 `forward` 保留若干 `print` 调试语句，可逐层核对形状——这也是最好的"如何验证"手段：跑通一次，看每行的 `(b,t,d)` 是否如表格所示流转。

**（补充）训练 vs 推理两个分支为什么长得不一样？** 训练时我们手里已有完整目标序列 `targets`，模型一次前向就能对所有位置并行算损失（每个位置预测下一个 token），效率极高；推理时我们没有未来，只能拿当前已生成序列、只取最后一个位置 `x[:, [-1], :]` 的预测作为下一个 token，再把它拼回去、循环生成。这正是 §2.1.5 掩码自注意力的价值：训练时用掩码假装"看不到未来"来并行，推理时则真的只能逐个生成——二者在数学上一致，只是计算方式不同。理解这点，你就理解了"GPT 式生成"的引擎盖下发生了什么。

> 收尾：至此我们已从零"手搓"出完整可计算的 Transformer。本书聚焦 LLM，本章不再细讲如何训练它；后文将类似地从零手搓一个 LLaMA 并带你训练自己的 Tiny LLaMA——而那时你会惊喜地发现，LLaMA 不过是把"Encoder 拿掉、Decoder 留着"的 Transformer。

---

## 关键代码速览

| 模块 | 核心一行 / 维度变化 |
| --- | --- |
| 注意力 | `scores=Q@Kᵀ/√d_k; out=softmax(scores)@V` → `(B,T,T)·(B,T,d_v)` |
| 多头 | 组合矩阵并行：`wq/q·wk/kᵀ` → `(B,nh,T,T)` → `@xv` → `(B,T,dim)` |
| 掩码 | `mask=torch.triu(full(-inf),(1,1,L,L))`；`-inf` 经 softmax→0 |
| EncoderLayer | `h=x+Attn(LN(x)); out=h+FFN(LN(h))`（`is_causal=False`） |
| DecoderLayer | 掩码自注意力 + 交叉注意力(`Q=Dec,K/V=Enc`) + FFN |
| PositionalEncoding | `pe[:,0::2]=sin; pe[:,1::2]=cos; x=x+pe`（`register_buffer`） |
| Transformer.forward | `(b,t)→(b,t,d)→enc→dec→lm_head→(b,t\|1,vocab)` |

---

## 实践清单

| 项目 | 内容 |
| --- | --- |
| 环境依赖 | `docs/chapter2/code/requirements.txt`：torch 2.7.0、transformers 4.52.4、tokenizers 0.21.1、numpy 2.2.6 等 |
| Python 版本 | 3.10 或 3.11 |
| 关键命令 | `python -m venv .venv && source .venv/bin/activate && pip install -r ./docs/chapter2/code/requirements.txt` |
| 运行入口 | `docs/chapter2/code/transformer.py`，`ModelArgs` + `main()` |
| 超参 | n_embd=100, n_heads=10, dim=100, dropout=0.1, max_seq_len=512, vocab_size=1000(被覆盖), block_size=1000, n_layer=2 |
| 硬件 | CPU 或单卡 GPU；分词器 `bert-base-chinese` 需联网下载 |
| 预期结果 | 打印参数量 `X.XXM`、各阶段形状；推理分支 `loss=None`，`logits` 形状 `(b,1,vocab)`；未训练输出为随机 token（验证数据流正确即可） |

---

## 本章小结

- **要点**：① 为何 Transformer——RNN 不可并行、长程差 → 全注意力架构；② 公式链 `softmax(qKᵀ)v`→`softmax(QKᵀ)V`→`softmax(QKᵀ/√d_k)V`（缩放为稳梯度）；③ 三种注意力——自注意力(Q=K=V 同源)、掩码自注意力(上三角 `-inf` 实现并行语言模型)、多头(组合矩阵并行+拼接+$W^O$)；④ Encoder/Decoder 骨架：Embedding+PE → N×EncoderLayer(自注意力+FFN, Pre-Norm+残差) → N×DecoderLayer(掩码自注意力+交叉注意力+FFN) → Linear+Softmax；⑤ 位置编码正余弦公式与相对位置可表达性推导；⑥ 工程要点：Pre-Norm vs Post-Norm、LayerNorm vs BatchNorm、`contiguous()`、`register_buffer`。
- **承上启下**：本章给出了 LLM 的"发动机"——Transformer 的完整数理与代码。第 1 章那些未解矛盾（稀疏、词序、长程）在此被注意力一举化解。下一章（第 3 章 经典预训练语言模型）将告诉我们：如何基于 Encoder/Decoder 的不同组合（BERT 只用 Encoder、GPT 只用 Decoder）构建出强大的预训练模型，并引出正式进入 LLM 的第 4 章。

---

## 习题

1. **概念/推导**：为什么缩放因子是 $\sqrt{d_k}$？不缩放会发生什么？
   - 提示：$d_k$ 大时 $QK^T$ 方差大、Softmax 饱和、梯度趋零；缩放使方差稳定。

2. **辨析**：掩码为何用 `-inf` 而非 0？Softmax 后效果如何？
   - 提示：0 仍保留部分权重；`-inf` 经 Softmax 严格变 0，彻底忽略未来。

3. **推导**：多头为何用"三个组合矩阵"而非 $n$ 组独立矩阵？二者等价吗？
   - 提示：等价（矩阵内积再拼接 ≡ 拼接矩阵再内积），组合矩阵更省时空、可并行。

4. **对比**：Layer Norm 与 Batch Norm 统计量维度不同在哪？为何 LLM 偏爱 Layer Norm？
   - 提示：LN 在每样本特征维、BN 在 batch 维；BN 小 batch/RNN/变长失效，LN 稳定。

5. **辨析**：Pre-Norm 与 Post-Norm 区别？为何现代 LLM 多用 Pre-Norm？
   - 提示：LN 在子层前/后；Pre-Norm 使深层 loss 更稳定、更易训练。

6. **推导**：为何正余弦 PE 可外推到更长序列？相对位置可表达性推导关键步是什么？
   - 提示：公式法可算任意 pos；关键在 $\theta_i=10000^{-2i/d}$ 与积化和差出现 $\cos(m-n)\theta_i$。

7. **代码**：Decoder 中交叉注意力的 Q、K、V 分别来自哪里？
   - 提示：Q 来自 Decoder 本层输出，K、V 来自 Encoder 输出。

8. **开放**：为什么说"几乎所有 NLP 任务都可视为 Seq2Seq"？文本分类、词性标注如何对应？
   - 提示：分类输出长 1、词性标注输出等长；统一为序列到序列框架。

---

## 延伸阅读

- Ashish Vaswani 等 (2017). *Attention Is All You Need.* arXiv:1706.03762（Transformer 原论文）
- Jay Mody. *An Intuition for Attention.* https://jaykmody.com/blog/attention-intuition/
- 配套代码：`docs/chapter2/code/transformer.py`（MultiHeadAttention / LayerNorm / MLP / Encoder / Decoder / PositionalEncoding / Transformer）
