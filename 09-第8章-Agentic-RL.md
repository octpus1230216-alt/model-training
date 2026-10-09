# 第8章 Agentic-RL：让大模型在环境中学会行动

> 对应原文：`docs/chapter8/第八章 大模型强化学习.md`（含配套代码 `docs/chapter8/{grpo,opd,search-r1,retool}/`）　|　本章定位：把第 4 章的 RLHF / 第 7 章的 Agent 工具调用往前推一步——用强化学习让模型**自己学会在环境里行动**（搜索、写代码、吸收更强模型的能力），而非仅靠人工标注的轨迹去模仿。

本章是全书的"前沿实战收口章"。第 2～4 章我们从 Transformer 一路讲到预训练语言模型与大语言模型，第 5～6 章动手训练并微调了一个属于自己的模型，第 7 章又学会了用 RAG 和 Agent 把模型接上外部工具。但到此为止，模型"会不会用工具""要不要搜""搜什么"仍由我们人类在提示词里写死。本章要解决的正是这个问题：**能不能让模型在跟环境交互的过程中，自己摸索出"何时行动、如何行动"**，并且越练越强？这就是 Agentic-RL（智能体强化学习）要回答的。

四个主题按难度递进：GRPO（单轮、结果奖励）→ OPD（单轮、逐 token 稠密信号）→ Search-R1（多轮、搜索环境）→ ReTool（多轮、代码环境）。它们共享同一套 PyTRIO 训练抽象，学完本章，你不仅能跑通四个最小示例，更能掌握"设计任意工具型强化学习任务"的通用框架。

## 本章目标
- 理解 Agentic-RL 是什么，说清它与第 4 章 RLHF、第 7 章 Agent 的承接与区别。
- 掌握 GRPO 的组内相对优势思想，能对照 PPO 解释"为什么可以省掉 critic（价值模型）"。
- 理解 OPD 的 reverse KL 逐 token 训练信号，说清"为什么 Teacher 必须评价 Student 自己生成的同一条轨迹"。
- 掌握 Search-R1 与 ReTool 的多轮轨迹构造，能解释 observation mask / feedback mask 为什么必要。
- 能够照着实践清单跑通四组最小命令，并看懂 reward、退化组、rollout 长度等关键指标。
- 能够用"设计 Agentic-RL 的五问清单"判断一个工具 Demo 是否具备进入强化学习训练的基础。

---

## 8.1 导言：Agentic-RL 是什么、为什么需要它

### 8.1.1 从"对齐的模型"到"会行动的智能体"

**Agentic-RL（智能体强化学习）**：让大语言模型在具备"环境"（如搜索引擎、代码解释器、奖励判题器）的训练框架里，通过反复试错、用环境反馈作为奖励来更新策略，从而自己学会"何时调用工具、调用什么、如何根据结果继续行动"的一类强化学习范式。直觉上，它就像让一个实习生**先上手干活、再拿结果复盘**：不是你手把手教他每一步怎么做，而是告诉他"做对了一起吃饭、做错了扣钱"，他自己摸索出高效套路。

要理解它为什么重要，先回顾我们已经学过的两块：

- **第 4 章的 RLHF**（基于人类反馈的强化学习）：用一个奖励模型把模型回答对齐到"人类偏好"，信号来自偏好比较，交互是**单轮、静态**的——模型吐一段回答，环境给一个偏好分，结束。
- **第 7 章的 Agent**：我们写了一个 `Tiny-Agent`，用 Function Calling 循环让模型"调工具"。但那个循环的**触发逻辑是我们写死的**：什么时候搜、搜什么词、搜几次，全靠系统提示词和固定流程决定，模型本身没有"为了更高奖励而主动优化搜索策略"的能力。

Agentic-RL 正是把这两者合流：保留 RL 的"靠结果反馈自我改进"，但把环境从"偏好判题器"升级为"可交互的真实工具"。原文一句话点明动机：

> "当 Agent 真正进入搜索、代码执行等环境后，仅靠监督微调已经很难覆盖所有可能的交互轨迹。模型需要自己尝试动作，根据最终结果判断一条轨迹是否有效，再把成功经验更新到策略中。"

### 8.1.2 为什么"带工具的强化学习"是必要的

**监督微调（SFT）的盲区**：交互轨迹的空间是组合爆炸的。以搜索为例，第一次搜什么词、看到结果后第二次怎么改写、搜到第几轮收手，可能的轨迹数量远超任何人工标注集能覆盖的范围。SFT 只能模仿"已有范例"，遇到范例之外的状态（分布外）就无能为力。

**强化学习的补位逻辑**：让模型先"随便试"，跑完整条轨迹后只看**最终结果**对不对——答对就鼓励这条轨迹上的所有动作，答错就抑制。这样，模型不必在每一步都有人工示范，它靠"结果信号"自己归纳出哪些动作组合有效。

更深层的原因有两个"能力边界"需要靠工具突破：

1. **知识边界**：模型参数里装的是训练截止时的世界，昨天的论文、实时股价它不知道。搜索工具把"外部知识"拉进上下文，模型只需学会"何时去拿、拿什么"。
2. **计算边界**：大数枚举、符号化简、精确浮点，模型"心算"容易错。代码解释器把"精确计算"外包出去，模型只需学会"什么时候该写程序、写什么程序"。

### 8.1.3 四个主题的衔接：一条清晰的主线

本章四个示例并非并列，而是按"训练信号来源"和"轨迹形态"递进（下表严格来自素材）：

| 序号 | 主题 | 训练信号来源 | 轨迹形态 | 策略更新 loss |
| --- | --- | --- | --- | --- |
| 8.2 | GRPO | 规则判题器（结果奖励） | 单轮 | `importance_sampling` / `ppo` |
| 8.3 | OPD | Teacher 逐 token 概率 | 单轮 | `importance_sampling` |
| 8.4 | Search-R1 | 搜索环境 + 最终答案 EM | 多轮（含 observation） | `importance_sampling` |
| 8.5 | ReTool | 代码解释器 + 最终答案数学等价 | 多轮（含 observation） | `ppo`（非对称 clip） |

递进逻辑可记为：**GRPO（单轮、结果奖励）→ OPD（单轮、逐 token 稠密信号）→ Search-R1 / ReTool（多轮环境交互 + observation mask）**。越往后，轨迹越复杂，但核心抽象始终不变。

### 8.1.4 统一框架：PyTRIO 与五步抽象

**PyTRIO 0.2.6**（本章锁死的训练/采样 SDK）：把**本地**的数据处理、环境交互、训练控制，与**远端**的模型采样、前向反向传播、权重管理连接起来。作者特意选择它而不是 veRL，目的是降低 Agentic-RL 的工程门槛与实验成本——按最小试跑配置，四个示例的基础预算力争控制在 200 元以内。

四个示例虽然任务不同，核心链路始终可以概括为下面 **5 步（本章最重要的抽象，必须记住）**：

1. 用当前策略生成一条或一组轨迹（`rollout`）；
2. 根据结果或 Teacher 反馈计算训练信号（`reward` / `advantage`）；
3. 将 `prompt`、`response`、旧策略 `logprob` 和 `advantage` 对齐为 `Datum`；
4. 调用 `forward_backward()` 与 `optim_step()` 更新策略；
5. 刷新采样客户端，让下一轮 rollout 用上新权重（**on-policy 约束的落点**）。

> **承上启下**：第 4 章已讲过"策略梯度 → 奖励模型 → PPO"的闭环；第 8.2～8.5 节不过是把那个闭环的"环境"换成了工具或更强模型，并把"单轮回答"换成了"多轮轨迹"。读懂第 4 章的读者，本章只需补两个新概念：**组内相对优势**与 **observation mask**。

---

## 8.2 GRPO：群体相对策略优化

### 8.2.1 从语言模型到强化学习策略

先把"语言模型"翻译成"强化学习"的词汇，这是理解一切的前提。自回归语言模型接收 prompt $x$，依次生成 $y=(y_1,y_2,\ldots,y_T)$：

$$\pi_\theta(y\mid x) = \prod_{t=1}^{T} \pi_\theta(y_t\mid x,y_{\lt t}),$$

符号说明：
- $\pi_\theta$：参数为 $\theta$ 的语言模型策略（它就是我们要优化的"智能体大脑"）；
- $y_{\lt t}$：位置 $t$ 之前已经生成的 token（自回归条件，即"历史"）；
- 连乘号表示"整段回答的概率 = 每个词在各自历史下条件概率的乘积"。

**表 8.1 语言模型 ↔ 强化学习概念对应（必背）**

| 强化学习概念 | 大语言模型中的含义 |
| --- | --- |
| 状态 $s_t$ | prompt 与已经生成的 token |
| 动作 $a_t$ | 下一个 token |
| 策略 $\pi_\theta(a_t\mid s_t)$ | 模型对下一个 token 的概率分布 |
| 轨迹 $\tau$ | 一段完整回答，或包含工具调用的多轮交互 |
| 环境 | 数据集判题器、搜索引擎、代码解释器等 |
| 奖励 $R(\tau)$ | 答案正确性、格式、工具执行结果等反馈 |

**rollout**：一次模型生成通常称为一次 rollout（采样轨迹）。训练目标（直观表述）是：提高高奖励轨迹中动作出现的概率，同时降低低奖励轨迹中动作出现的概率。

最朴素、也最本质的策略梯度形式是：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta} \left[ A(\tau) \nabla_\theta \log \pi_\theta(\tau) \right],$$

符号说明：
- $A(\tau)$：**优势函数（advantage）**。可理解为"这条轨迹相对平均水平值不值得鼓励"——$A>0$ 表示值得鼓励，$A<0$ 表示当前策略该减少类似行为；
- $\nabla_\theta \log \pi_\theta(\tau)$：轨迹对数概率对参数的梯度方向，决定"往哪个方向改参数能让这条轨迹更容易/更难出现"。

**为什么大模型 RL 特别难（两个核心问题）**：① 动作空间是整个词表，一段回答可能数百到数千 token，梯度要在海量位置上求和；② 必须控制新旧策略的变化幅度，防止一次更新就毁掉模型已有的语言能力。

> 因此 GRPO 节的真正难点集中在两句话：**① 如何得到稳定的 advantage；② 如何安全地用 advantage 更新策略。**

### 8.2.2 从 PPO 到 GRPO：省掉价值模型

**PPO（近端策略优化）**的做法：额外训练一个 Value Model（价值模型）估计状态价值 $V_\phi(s_t)$，用"回报 − 价值"算 advantage。问题在于：Value Model 往往与策略模型规模接近，带来额外显存、训练与同步成本；而且价值估计本身的误差会一路传到策略更新里。

**GRPO（Group Relative Policy Optimization，群体相对策略优化）**的替代方案非常巧妙——**干脆不训价值模型，改用"同题组内相对比较"**。对同一个问题 $x$，先从旧策略采样 $G$ 个回答：

$$\{y_1,y_2,\ldots,y_G\} \sim \pi_{\theta_{\mathrm{old}}}(\cdot\mid x).$$

判题器分别给出奖励 $r_1,\ldots,r_G$。**经典 GRPO 的标准化形式**为：

$$A_i = \frac{r_i-\mathrm{mean}(r_1,\ldots,r_G)}{\mathrm{std}(r_1,\ldots,r_G)+\varepsilon}.$$

符号说明：
- $r_i$：第 $i$ 条回答的规则奖励（例如答对=1、答错=0）；
- $\mathrm{mean}(\cdot)$ / $\mathrm{std}(\cdot)$：**组内**均值与标准差，注意是"组内"而不是"全训练集全局"；
- $\varepsilon$：防止除零的小常数。

**直觉**：同一道题里的高分回答拿到正 advantage、低分回答拿到负 advantage；**问题本身的难度被组内基线抵消掉**，策略可以更专注地学习"在相同条件下，哪种生成方式更好"。

数值示例（原文给出）：同一道数学题采样 4 个回答，奖励为 $[1,0,0,1]$。组内均值 $0.5$，只做均值中心化时 advantage 为：

$$[0.5,\,-0.5,\,-0.5,\,0.5].$$

**本章代码采用的简化形式**（注意与上式差异）：

$$A_i=r_i-\bar r.$$

> 原文说明：两种写法表达相同的组内相对思想，但**梯度尺度不同**——比较实验时应固定 advantage 的计算方式，否则无法公平对比。

**退化组（degenerate group，重要实践概念）**：当一组回答**全对或全错**时，每个 $A_i$ 都等于 0，这组数据**不产生有效策略梯度**。若训练日志里退化组比例长期偏高，通常要调整：题目难度、采样温度、基础模型能力、或 group size（每题采样数）。代码会在主循环把这种 group 标记为退化组并**跳过**，避免提交没有训练信号的 `Datum`。

### 8.2.3 奖励与可验证强化学习（RLVR）

GRPO 只负责把组内奖励转成相对优势，**奖励本身的质量决定模型最终学到什么**。数学/代码任务可用确定性规则判题器，这类训练即 **RLVR（Reinforcement Learning with Verifiable Rewards，可验证奖励强化学习）**。

本章 GRPO 示例：GSM8K（小学数学）+ `\boxed{}` 抽取答案。最小判题逻辑（`docs/chapter8/grpo/01-demo-sync.py`）：

```python
def extract_boxed(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if not matches:
        return None
    return matches[-1].strip()          # 取最后一个 \boxed{}


def normalize_answer(text: str) -> str:
    return text.replace(",", "").strip().rstrip(".")   # 去逗号/首尾空白/句点


def grade_answer(response: str, ground_truth: str) -> float:
    answer = extract_boxed(response)
    if answer is None:
        return 0.0                                  # 没写 \boxed{} 直接 0 分
    return 1.0 if normalize_answer(answer) == normalize_answer(ground_truth) else 0.0
```

张量/数据维度说明：每个样本产生标量 `reward ∈ {0.0, 1.0}`；一组 `group_size` 个样本拼成长度 `G` 的 `rewards` 列表，再中心化得到长度 `G` 的 `advantages`。

**Reward 设计 5 条检查清单（原文，建议逐条背）**：
1. **答案抽取是否稳健**：同一正确答案可能是整数、小数、分数或带逗号数字；
2. **格式奖励是否压过正确性**：格式只能是辅助约束，不能让"格式正确的错误答案"拿主奖励；
3. **参考答案是否可靠**：错误标签会提供方向相反的训练信号；
4. **判题器是否泄露信息**：模型不应通过 prompt 或工具输出直接读到标准答案；
5. **奖励是否具有区分度**：大量全 0 或全 1 的组都会让 GRPO 失去相对比较信号。

工程建议：正式训练前先把基础模型 rollout 保存下来，**离线人工检查一批"回答—抽取结果—reward"三元组**，这步往往比直接调学习率更早发现问题。

### 8.2.4 策略比率与更新损失：GRPO 与 PPO 的层次

**新旧策略概率比**（rollout 由旧策略生成，更新发生在当前策略）：

$$\rho_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t}\mid x,y_{i,\lt t})}{\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid x,y_{i,\lt t})} = \exp\left(\log\pi_\theta-\log\pi_{\theta_{\mathrm{old}}}\right).$$

符号说明：
- 分子：当前策略在位置 $t$ 对已采样 token 的概率；
- 分母：旧策略（rollout 当时）的同一概率；
- 指数形式说明代码里实际存的是 **logprob 之差**（`Datum.logprobs` 存旧策略 logprob，比较时取指数即可）。

**重要性采样（Importance Sampling）目标**：

$$J_{\mathrm{IS}}(\theta) = \mathbb{E}_{i,t} \left[\rho_{i,t}(\theta)\,A_i\right].$$

动机：若 $\rho$ 偏离 1 太远，少数 token 会主导整条梯度，训练不稳。

**PPO 截断目标**（限制单次更新幅度，防止策略"走太远"）：

$$J_{\mathrm{PPO}}(\theta) = \mathbb{E}_{i,t} \left[\min\left(\rho_{i,t}A_i,\ \mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\,A_i\right)\right].$$

符号说明：
- $\mathrm{clip}(\rho,1-\epsilon,1+\epsilon)$：把概率比夹在 $[1-\epsilon,1+\epsilon]$ 区间；
- $\min$ 的作用——当 $A>0$（该鼓励）时若 $\rho$ 过大则截断（防止过度"讨好"）；当 $A<0$（该打压）时若 $\rho$ 过小则截断（防止过度"打压"），整体形成一个**悲观下界**；
- $\epsilon$：截断超参，经典 PPO 取 0.2。

关于 KL 约束：DeepSeekMath 中的经典 GRPO 目标采用 PPO 风格 clipping，并**加入与参考策略之间的 KL 惩罚项**，防止策略偏离原始模型过远。本章正文未给出该完整 LaTeX，只给出上面的 $J_{\mathrm{PPO}}$ 形式；KL 项指与参考策略 $\pi_{\mathrm{ref}}$ 的 KL 距离。

**GRPO 与 PPO 的层次区分（易混点，务必分清）**：
- **GRPO** 描述如何**对同题分组采样，并由组内奖励得到 advantage**（"优势怎么来"）；
- **importance sampling 或 PPO** 描述如何**把 advantage 转成策略梯度**（"梯度怎么算"）。

本章同步版默认用 PyTRIO 内置 `importance_sampling`（便于观察 `old_logprobs`、`advantages` 与当前策略的数据关系），命令行加 `--loss-fn ppo` 即可切换到内置 PPO loss——两者复用同一批 GRPO `Datum`，只切换 loss 函数。

### 8.2.5 同步版 GRPO 的核心代码骨架

两个配套文件：`01-demo-sync.py`（同步，正文主线）与 `02-demo-async.py`（异步，并发处理同 batch 不同 prompt）。两者用相同的数据、reward、advantage、Datum 和 loss。

#### （1）数据类与配置

```python
@dataclass
class RolloutSample:
    """一条采样结果，以及构造 importance_sampling 所需的旧策略 logprobs。"""
    tokens: list[int]
    logprobs: list[float]
    text: str
    reward: float
    advantage: float
```

**关键约束**：`logprobs` **必须来自 rollout 时的 Student**；策略更新后重算的是"新策略 logprob"，**不能替代**旧策略概率——否则重要性采样比 $\rho$ 就不再是"新旧之比"了。

`GRPOConfig` 关键默认超参（考点）：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--base-model` | `Qwen/Qwen3.5-4B` | PyTRIO 基础模型 |
| `--lora-rank` | `32` | LoRA rank |
| `--steps` | `10` | 优化步数 |
| `--batch-size` | `4` | 每步题目数 |
| `--group-size` | `4` | 每题采样 completion 数 |
| `--max-tokens` | `1024` | 单次采样最大 token |
| `--temperature` | `1.0` | 采样温度 |
| `--learning-rate` | `4e-5` | Adam lr |
| `--beta1`/`--beta2` | `0.9`/`0.95` | Adam 动量 |
| `--loss-fn` | `importance_sampling` | `choices=(importance_sampling, ppo)` |

#### （2）对同一道题采样一组回答（GRPO rollout 核心）

`run_rollout_group()`（`docs/chapter8/grpo/01-demo-sync.py:205`）：

```python
def run_rollout_group(sampling_client, tokenizer, prompt_tokens,
                      ground_truth, sampling_params, group_size) -> list[RolloutSample]:
    result = sampling_client.sample(
        prompt=trio.ModelInput.from_ints(prompt_tokens),
        num_samples=group_size,          # 一次请求返回同题的一组 completion
        sampling_params=sampling_params,
        return_text=True,
    ).result()

    rewards, raw_samples = [], []
    for sequence in result.sequences:
        text = sequence.text or tokenizer.decode(sequence.tokens, skip_special_tokens=True)
        tokens = list(sequence.tokens)                       # 完成 token 序列
        logprobs = [float(v) for v in sequence.logprobs]     # 旧策略在每位的 logprob
        if len(tokens) != len(logprobs):
            raise ValueError("token/logprob 长度不一致")
        reward = grade_answer(text, ground_truth)            # 规则判题 → 0/1
        rewards.append(reward)
        raw_samples.append((tokens, logprobs, text))

    mean_reward = sum(rewards) / len(rewards)
    # 注意：advantage 必须等整组 reward 都算完才能算（组内基线依赖完整组）
    return [RolloutSample(tokens=tokens, logprobs=logprobs, text=text,
                          reward=reward, advantage=reward - mean_reward)
            for (tokens, logprobs, text), reward in zip(raw_samples, rewards, strict=True)]
```

这段代码完成四件事：① 保存 completion token；② 保存采样时旧策略对每个 completion token 的 logprob；③ 用规则判题器得 0/1 reward；④ 计算 `advantage = reward - mean_reward`。

#### （3）右移对齐的 PyTRIO Datum

自回归：用当前位置 token 预测下一个 token。设 prompt 长 $m$，completion 长 $T$，四个数组长度均为 $m+T-1$：

```python
def build_grpo_datum(prompt_tokens: list[int], sample: RolloutSample) -> trio.Datum:
    observation_len = len(prompt_tokens) - 1       # prompt 掩码长度
    input_tokens      = prompt_tokens + sample.tokens[:-1]            # 长度 m+T-1
    target_tokens     = [0]   * observation_len + sample.tokens       # 长度 m+T-1
    padded_logprobs   = [0.0] * observation_len + sample.logprobs     # 长度 m+T-1
    padded_advantages = [0.0] * observation_len + [sample.advantage] * len(sample.tokens)

    return trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens,     dtype=np.int64),
            "logprobs":      np.asarray(padded_logprobs,   dtype=np.float32),
            "advantages":    np.asarray(padded_advantages, dtype=np.float32),
        },
    )
```

维度说明：`input_tokens ∈ ℤ^(m+T-1)`，`target_tokens / logprobs / advantages ∈ ℝ^(m+T-1)`。prompt 区间只提供上下文，三个 loss 字段用 **0 占位**；completion 区间放进真实目标 token、旧策略 logprob、整条回答共享的组内 advantage。**`observation_len = len(prompt)-1`** 的原因：最后一个 prompt token 对应的位置才刚开始预测第一个 completion token。

对齐示意：
```text
prompt       = [x1, x2, x3]      completion   = [y1, y2]
model_input  = [x1, x2, x3, y1]  # 长度 m+T-1 = 4
target       = [ 0,  0, y1, y2]
old_logprob  = [ 0,  0, l1, l2]
advantage    = [ 0,  0,  A,  A]
```

#### （4）串联一轮同步训练

```python
for step in range(effective_steps):
    batch_rows = pick_batch(train_data, step, config.batch_size, config.all_data)
    sampling_client = training_client.save_weights_and_get_sampling_client()  # 每步刷新 → on-policy
    datums, n_degenerate = [], 0
    for row in tqdm(batch_rows, desc=f"GRPO step {step}"):
        prompt_tokens  = build_prompt(tokenizer, row["question"])
        ground_truth   = extract_gsm8k_answer(row["answer"])
        rollout_samples = run_rollout_group(..., group_size=config.group_size)
        if all(s.advantage == 0.0 for s in rollout_samples):   # 退化组
            n_degenerate += 1
            continue
        for sample in rollout_samples:
            datums.append(build_grpo_datum(prompt_tokens, sample))

    fwd = training_client.forward_backward(datums, loss_fn=config.loss_fn)
    training_client.optim_step(adam_params).result()
    fwd.result()
```

**on-policy 落点**：第 $k$ 步更新完后，第 $k+1$ 步用新权重重新生成轨迹（每步都调 `save_weights_and_get_sampling_client()`）。同步版显式等待结果；异步版把 `sample` / `forward_backward` / `optim_step` 换成 `*_async()` 并用 `asyncio.gather()` 并发处理同 batch 内不同 prompt。

**训练观察指标表**

| 指标 | 含义 | 异常现象 |
| --- | --- | --- |
| `reward` | batch 内各 prompt 平均正确率的均值 | 长期不升 → 查 reward、数据难度、lr |
| `frac_degenerate` | 全对/全错 group 比例 | 过高 → 有效相对优势不足 |
| `rollout/avg_gen_len` | completion 平均 token 数 | 接近 `max_tokens` → 查截断 |
| `train_tokens` | advantage 非零的 completion token 数 | 为 0 → 当前 step 无训练信号 |
| `loss_mean` | 当前策略更新的主 loss | 剧烈波动 → 查 lr 与概率比 |

> **本节收束**：当前策略分组采样 → 规则环境给结果奖励 → 组内比较产生 advantage → PyTRIO 完成策略更新。GRPO 把"价值模型"换成了"同题多采样"，这是它最省资源的地方。

---

## 8.3 OPD：On-Policy Distillation（在线策略蒸馏）

### 8.3.1 从离线蒸馏到在线策略蒸馏

**知识蒸馏（Knowledge Distillation）**：用较强的 Teacher 模型 + 较小的 Student 模型，让 Student 去拟合 Teacher 在**固定样本**上的概率分布。直觉上，就像让学霸把自己"做题时的把握程度"传授给学弟，而不是只给最终答案。

**核心问题（原文表述）**：随着 Student 在训练中不断变化，**固定数据与 Student 真正会访问的状态之间会出现偏差**。具体情形：固定蒸馏数据里大多是 Teacher 的高质量轨迹，而 Student 真实生成时**较早出现了一个错误 token**，随后进入的状态可能**从未出现在蒸馏数据中**，Teacher 的指导便覆盖不到这段分布外轨迹。

离线蒸馏目标：设 Teacher 为 $\pi_T$、Student 为 $\pi_\theta$，先固定数据集 $\mathcal D$，在其中状态上最小化分布差异：

$$\mathcal L_{\mathrm{offline}} = \mathbb E_{s\sim\mathcal D} \left[D\bigl(\pi_T(\cdot\mid s),\pi_\theta(\cdot\mid s)\bigr)\right].$$

符号说明：
- $D(\cdot,\cdot)$：分布散度（如 KL）；
- $s\sim\mathcal D$：状态来自**固定数据集**，而非 Student 当前策略——这恰恰是偏差来源。

**OPD 的改动**：状态采样改为**当前 Student**：

$$y\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid x),$$

再让 Teacher 计算**同一条 Student completion** 的条件概率 $\log\pi_T(y_t\mid x,y_{\lt t})$。一次训练迭代三步：① **Student rollout**（当前权重生成回答）；② **Teacher feedback**（Teacher 在完全相同的 prompt 与 completion 上算 logprob）；③ **Student update**（按两者概率差构造逐 token 信号并更新）。

**计算图边界（重要）**：Teacher 只做前向计算；Student 的采样动作**不参与反向传播**，梯度从构造好的训练目标回到当前 Student 参数（停止梯度边界）。这使 OPD 能直接复用现有 rollout 与策略优化基础设施。用途：推理能力迁移、能力恢复、持续学习，且与 RL 数据流天然结合（Student rollout = on-policy 轨迹，Teacher logprob = 逐 token 信号）。

### 8.3.2 Reverse KL 与逐 token 训练信号

本章使用 **reverse KL**（反向 KL 散度）：

$$D_{\mathrm{KL}}\left(\pi_\theta\,\|\,\pi_T\right) = \mathbb E_{y\sim\pi_\theta} \left[\log\pi_\theta(y\mid x)-\log\pi_T(y\mid x)\right].$$

符号说明：
- 期望下标 $y\sim\pi_\theta$：**Student 自己的分布**（这是"reverse"/on-policy 的关键）；对比 forward KL（期望在 Teacher 上），reverse KL 期望在 Student 上，所以信号始终落在 Student 实际会生成的 token 上；
- 被减数是 Student 对数概率，减数是 Teacher 对数概率。

对 Student 实际采样到的第 $t$ 个 token，得到 **Monte Carlo 估计**：

$$d_t = \log\pi_{\theta_{\mathrm{old}}}(y_t\mid x,y_{\lt t})-\log\pi_T(y_t\mid x,y_{\lt t}).$$

符号说明：
- $d_t>0$：Student 对该 token 的偏好**高于** Teacher；
- $d_t<0$：Teacher 给出的概率更高（Student 该向 Teacher 靠拢）。

为最小化 reverse KL，本章把训练 advantage 写为：

$$A_t=-\beta\, d_t.$$

符号说明：
- $\beta$：蒸馏强度系数（命令行 `--kl-penalty-coef`，默认 `1.0`）；
- 负号含义：$d_t>0$（Student 过于自信）→ $A_t<0$ → 抑制；$d_t<0$（Teacher 更认可）→ $A_t>0$ → 鼓励。

**信号密度**：每个 completion token 都有独立训练信号，**远高于只在结尾给一个 0/1 reward**。reverse KL 还有较强 **mode-seeking（寻模）** 特性：Student 会优先集中到 Teacher 已赋予高概率的区域，适合把 Teacher 的高置信行为压缩进小模型。影响效果的因素有 Teacher 与 Student 的**能力差距、推理风格、tokenizer 兼容性**——"更大的 Teacher"只是候选条件，Teacher 还需在 Student 当前轨迹上提供**有价值且可学习**的新信号。

**表 8.2 GRPO vs OPD 数据来源对比（必背）**

| 项目 | GRPO | OPD |
| --- | --- | --- |
| 轨迹来源 | 当前策略 | 当前 Student |
| 反馈来源 | 环境或规则判题器 | Teacher 概率分布 |
| 信号粒度 | 常见为整条轨迹 reward | 逐 completion token |
| 是否需要标准答案 | 通常需要可验证结果 | prompt-only 数据即可 |
| 是否需要额外模型 | 可省去 Value Model | 需要 Teacher 前向计算 |
| 主要风险 | reward hacking、退化组 | Teacher 不兼容、能力差距不合适 |

### 8.3.3 Teacher 必须评价同一条 Student 轨迹

**OPD 的关键数据约束**：Teacher 与 Student 必须在**相同前缀下评价相同目标 token**。反例警示：让 Teacher 自己生成一条回答、再让 Student 学这条回答，会**退回常见合成数据蒸馏流程**（那不是 OPD）。

PyTRIO 实现：先把 prompt 和 Student completion 拼接，再调 Teacher 的 `compute_logprobs()`，最后切片：

```python
def completion_teacher_logprobs(teacher_client, prompt_ids, completion_ids):
    all_ids = prompt_ids + completion_ids
    all_logprobs = teacher_client.compute_logprobs(
        trio.ModelInput.from_ints(all_ids)).result()
    start = len(prompt_ids)
    end = start + len(completion_ids)
    return [float(x) for x in all_logprobs[start:end]]
```

**易错点：logprob 位置偏移（原文明确警告）**：不同 SDK 对 logprob 的位置定义可能**差一位**，不能仅凭数组长度猜切片。最稳妥做法是构造一个很短的已知序列，确认返回值里每个位置究竟是"当前 token 的 logprob"还是"下一个 token 的 logprob"，再做边界测试。本章代码已按 PyTRIO 0.2.6 的返回约定对齐。

**tokenizer 兼容性校验（训练前必做）**：

```python
text = "A short tokenizer alignment test."
assert student_tokenizer.encode(text, add_special_tokens=False) == \
       teacher_tokenizer.encode(text, add_special_tokens=False)
```

同系列但 chat template 不同的模型：应由 **Student tokenizer 渲染一次完整 prompt**，再让 Teacher 直接评价该 token 序列，**不要分别套两份 chat template**（否则两边看到的上下文已不同）。

### 8.3.4 同步版 OPD 的核心代码骨架

#### （1）创建 Student 与 Teacher

```python
service_client  = trio.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model=args.base_model, rank=args.lora_rank, seed=args.seed)   # Student：会更新
tokenizer = training_client.get_tokenizer()

teacher_client = service_client.create_sampling_client(
    base_model=(args.teacher_base_model or args.base_model),
    model_path=args.teacher_model_path)                               # Teacher：只做前向，参数固定
```

默认 Student `Qwen/Qwen3.5-4B`，Teacher `Qwen/Qwen3.6-27B`（实际以 PyTRIO 支持列表为准）。数据来自 DeepMath-103K（`AI-ModelScope/DeepMath-103K`），只保留 `question` 字段（prompt-only）。

#### （2）每步刷新 Student sampler，生成轨迹并算 reverse KL

```python
if student_sampler is None or step % args.sampler_refresh_steps == 0:
    student_sampler = training_client.save_weights_and_get_sampling_client()

result = student_sampler.sample(
    prompt=trio.ModelInput.from_ints(prompt_ids),
    num_samples=args.group_size, sampling_params=sampling_params,
    return_text=False).result()

for sequence in result.sequences:
    completion_ids = sequence.tokens
    if not completion_ids:
        continue
    student_lps = [float(v) for v in sequence.logprobs]            # 旧策略 logprob
    teacher_lps = completion_teacher_logprobs(teacher_client, prompt_ids, completion_ids)
    reverse_kl = np.asarray(student_lps) - np.asarray(teacher_lps)  # d_t 向量
    advantages = -args.kl_penalty_coef * reverse_kl                # A_t = -β d_t
    datums.append(build_opd_datum(prompt_ids, completion_ids, student_lps, advantages))
```

> `group_size` 在 OPD 中的作用：**扩大同一 prompt 下的 Student 状态覆盖范围**。每条轨迹**独立**得 Teacher 信号，**不做 GRPO 式组内中心化**。

#### （3）逐 token OPD Datum（与 GRPO 同 schema，差异仅在 advantage）

```python
def build_opd_datum(prompt_ids, completion_ids, old_logprobs, advantages):
    prompt_loss_len = len(prompt_ids) - 1
    input_ids        = prompt_ids + completion_ids[:-1]
    target_ids       = [0]   * prompt_loss_len + completion_ids
    padded_logprobs  = [0.0] * prompt_loss_len + list(old_logprobs)
    padded_advantages= [0.0] * prompt_loss_len + list(advantages)   # 每个 token 一个独立 A_t
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(input_ids),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_ids,      dtype=np.int64),
            "logprobs":      np.asarray(padded_logprobs, dtype=np.float32),
            "advantages":    np.asarray(padded_advantages, dtype=np.float32),
        },
    )
```

右移规则与 GRPO 完全一致；prompt 区间三字段继续用 0 占位。最后统一用 `loss_fn="importance_sampling"` 做 `forward_backward` + `optim_step`。

**指标**：`opd/reverse_kl_mean`（Student 与 Teacher 概率平均差异）、`opd/reverse_kl_std`（不同 token 信号离散度）、`data/completion_tokens_per_second`（有效吞吐）。**故障排查顺序**：token 对齐 → Teacher 质量 → KL 系数 → 学习率 → sampler 刷新频率 → 截断比例。注意：reverse KL 下降只说明分布接近 Teacher，**任务正确率仍需通过固定的 held-out 评测确认**。

> **本节收束**：OPD 把 GRPO 的"结果奖励"换成"Teacher 逐 token 概率差"，信号密度从"一条轨迹一个分"提升到"每个 token 一个分"，且无需标准答案——这是它能在 prompt-only 数据上做能力迁移的原因。

---

## 8.4 Search-R1：让 LLM 学会搜索

### 8.4.1 从固定检索到自主搜索

**动机**：LLM 内部知识受训练数据时间范围、参数容量、长尾覆盖率限制。**RAG（检索增强生成）**由**系统预先检索**——检索词、次数都是外部定的；**Search-R1** 把**搜索决策交给模型**：何时搜、搜什么、是否据新证据继续搜、何时结束。

**多轮轨迹形式化**：

$$\tau=(x,a_1,o_1,a_2,o_2,\ldots,a_T),$$

符号说明：
- $x$：初始 prompt（用户问题）；
- $a_t$：模型生成的搜索调用或最终回答（**动作**）；
- $o_t$：搜索环境返回的 **observation（观测）**。

模型只有跑完整条轨迹才获得结果奖励，GRPO 再把"成功/失败"的搜索路径区分开。

**数据边界（极重要）**：训练样本**只包含问题和参考答案**，**不含人工写的搜索词、也不含标准搜索轨迹**。若训练集已给固定检索轨迹，任务就退化成"轨迹模仿"而非 Search-R1——搜索 query、次数、证据用法必须由**当前策略在 rollout 中自己探索**。

**表 8.3 固定 RAG vs Search-R1**

| 环节 | 固定 RAG | Search-R1 |
| --- | --- | --- |
| 是否检索 | 由外部流程决定 | 模型根据当前状态决定 |
| 检索词 | 预设或单次生成 | 可根据 observation 多次改写 |
| 检索轮数 | 通常固定 | 由策略决定，受最大次数约束 |
| 训练对象 | 回答模型或检索器 | 搜索动作与最终回答的统一策略 |
| 奖励 | 常见为监督答案 loss | 完整轨迹的结果奖励 |

多跳示例（原文）：问"《小王子》作者出生在哪个国家"→ 模型先搜作者名，看到 observation 后再搜"作者出生地"，最后答"France"。第二次搜索词**依赖第一次 observation**，所以无法在 rollout 前一次性定好——训练代码必须维护环境状态，在每个 assistant turn 后解析模型动作。

**协议选型**：Search-R1 论文用 `<search>...</search>` / `<information>...</information>` 文本标签；本章用 **Qwen3.5-4B 原生 function calling 模板**（减少模型能力与工具格式冲突，且保留结构化工具消息）。解析器 `parse_assistant` 只允许"每轮恰好一次合法工具调用"，调用后不能夹带答案；格式残缺输出标记为 `invalid` 并结束轨迹得格式惩罚。**没有单独"停止"动作**：输出中不存在合法工具调用，本轮即作为最终回答结束。

### 8.4.2 搜索后端怎样变成环境 observation

三个后端：DeepSeek Search（付费高并发）、Wikipedia（免费免密钥，适合验证链路）、知乎（免费但有额度）。统一抽象（`rollout.py` 只依赖这两个 dataclass，不依赖任何 HTTP 响应格式）：

```python
@dataclass(frozen=True)
class SearchItem:
    title: str
    content: str
    source: str | None = None
    url: str | None = None

@dataclass(frozen=True)
class SearchResult:
    ok: bool
    items: list[SearchItem]
    latency: float
    status: int | None = None
    error: str | None = None
```

工厂函数 `create_search_client(backend, ...)` 按 `deepseek` / `wikipedia` / `zhihu` 返回对应客户端。默认并发与超时：`{"deepseek":16, "wikipedia":3, "zhihu":1}`、`{"deepseek":60.0, "wikipedia":15.0, "zhihu":15.0}`。外层 `search()` 设计极健壮：限速 + timeout + 429/5xx 有限重试（指数退避、尊重 `Retry-After`）；累计 `SearchStats`；**失败不抛异常**而是返回 `SearchResult(ok=False, error=...)`，并把错误文本也作为 observation 交给模型（模型可改写 query 重试）。证据 `format_item` 必须给 `content`（只给标题/链接不够），并在接入轨迹前用 `max_tool_response_tokens` / `max_trajectory_tokens` **再次限制**证据长度。

### 8.4.3 多轮 rollout 状态机

**核心数据结构**（`rollout.py`）：

```python
@dataclass
class RolloutConfig:
    group_size: int = 8                 # 同一问题一次采样的轨迹数
    max_search_calls: int = 4           # 一条轨迹最多搜几次
    max_assistant_turns: int = 6        # 最多几轮 Assistant
    max_trajectory_tokens: int = 8192
    max_assistant_tokens: int = 1024
    max_tool_response_tokens: int = 1024
    search_concurrency: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 42

@dataclass
class AssistantTurn:
    prompt_tokens: list[int]      # 本轮生成前的完整输入 token
    completion_tokens: list[int]  # 本轮 assistant 生成的 token（可训练）
    logprobs: list[float]         # 采样时旧策略 logprob
    text: str

@dataclass
class Trajectory:
    example: SearchExample
    group_index: int
    messages: list[dict[str, Any]]
    next_prompt_tokens: list[int] | None = None
    turns: list[AssistantTurn] = field(default_factory=list)
    search_calls: int = 0
    final_text: str = ""
    reward: float = -0.1
    advantage: float = 0.0
    valid_format: bool = False
    exact_match: bool = False
    done: bool = False
```

> 搜索结果存在 `messages` 和下一轮 prompt 中，但**它没有 rollout logprob**（这些 token 来自环境）。

**状态机的明确边界（6 项）**：`max_search_calls`（最多搜几次）、`max_assistant_turns`（最多几轮）、`max_assistant_tokens`（单轮动作上限）、`max_tool_response_tokens`（单次 observation 上限）、`max_trajectory_tokens`（整条上下文上限）、搜索 timeout 与 concurrency。这些既控成本，也**定义了 Agent 能访问的环境范围**——评测时必须保持相同配置，否则更高搜索预算本身就可能提高正确率。

**"共享根节点、随后分叉"的分叉实现**：首轮每个问题只提交一个 prompt，`num_samples=group_size`；对返回的每组序列 `copy.deepcopy(root)` 成独立分支（各自搜索历史独立）。后续轮次各分支 prompt 已不同，`num_samples=1`。这样同时满足：同题轨迹能 GRPO 组内比较，且每条分支能按自己搜索结果继续行动。

```python
# 首轮：num_samples=group_size，深拷贝分叉
responses = asyncio.run(sample_requests_async(sampling_client, first_requests, config, tokenizer))
for request, response in zip(first_requests, responses, strict=True):
    root = roots[request.trajectory_index]
    for group_index, sequence in enumerate(response.sequences):
        branch = copy.deepcopy(root)
        branch.group_index = group_index
        pending = consume_assistant(branch, request.prompt_tokens, sequence, tokenizer, config)
        if pending is not None:
            pending_searches.append(pending)
resolve_searches(pending_searches, search_client, tokenizer, config)   # 线程池并发搜索

# 后续轮次：各分支 num_samples=1，直到全部 done
while any(not t.done for t in trajectories):
    requests = [make_request(...) for t in trajectories if not t.done]
    responses = asyncio.run(sample_requests_async(sampling_client, requests, config, tokenizer))
    # ... consume_assistant + resolve_searches
```

`consume_assistant()` 消费一次 Assistant 输出：解析为工具调用且预算未耗尽则挂起待搜索，否则标记 `done` 作为最终回答。`resolve_searches()` 用 `ThreadPoolExecutor` 并发执行不同轨迹的搜索，`finish_search()` 把 observation 写回 `messages` 与 `next_prompt_tokens`。`make_request()` 用动态 token 预算：`max_tokens = min(max_assistant_tokens, max_trajectory_tokens - len(prompt_tokens))`，预算耗尽则结束轨迹。

### 8.4.4 工具协议与 token 连续性

**多轮训练最易踩的坑**：重新渲染 chat template 后的 token 与真实采样 token 不一致（tokenizer 可能清理空格、规范工具调用文本、自动补轮次结束符）。如果训练阶段重新编码模型文本，**旧 logprob 将无法与目标 token 一一对应**。

本章两条原则：① Assistant 动作始终保留 sampler 返回的原始 token；② 只通过 chat template 计算"新加入的结束符 + 新 observation"的增量 token。`build_next_prompt()` 用空 assistant 消息提取模板闭合 token：

```python
def build_next_prompt(tokenizer, messages_before_assistant, previous_prompt_tokens,
                      completion_tokens, next_tool_message) -> list[int]:
    canonical_prompt = build_prompt(tokenizer, messages_before_assistant)
    empty_assistant_end = _render_chat(tokenizer,
        [*messages_before_assistant, {"role": "assistant", "content": ""}],
        add_generation_prompt=False)
    if empty_assistant_end[:len(canonical_prompt)] != canonical_prompt:
        raise ValueError("chat template 无法从空 assistant 提取结束边界")
    assistant_closing_tokens = empty_assistant_end[len(canonical_prompt):]

    canonical_next_prompt = build_prompt(tokenizer, [*messages_before_assistant,
                                                    {"role": "assistant", "content": "<assistant_text>"},
                                                    next_tool_message])
    observation_tokens = canonical_next_prompt[len(canonical_assistant_end):]
    overlap = _suffix_prefix_overlap(completion_tokens, assistant_closing_tokens)  # 防重复补 token
    return [*previous_prompt_tokens, *completion_tokens,
            *assistant_closing_tokens[overlap:], *observation_tokens]
```

下一轮构造 Datum 时再校验 `turn.prompt_tokens[:len(full_tokens)] == full_tokens`（前缀扩展），尽早发现模板改写、结束符重复、observation 拼接错误。**对 Agentic-RL，token 连续性属于算法正确性的一部分**：轨迹文本层面合理，仍可能因 token 边界错位而产生错误梯度。

### 8.4.5 Reward、组内优势与 observation mask

NQ / HotpotQA 短答案任务，模型最终需输出唯一一行 `Answer: <short answer>`。三档规则奖励：

$$R(\tau)=\begin{cases}1, & \text{格式正确且答案精确匹配},\\ 0, & \text{格式正确但答案错误},\\ -0.1, & \text{最终格式无效}.\end{cases}$$

`reward.py` 归一化：转小写 → 去标点（`unicodedata.category` 以 `P` 开头）→ 去英文冠词（`a/an/the`）→ 合并空格。只评价**最终任务结果**，不为"搜了几次"或"query 好看"加分。

**组内基线（GRPO）**：对同一道题的 $G$ 条完整轨迹：

$$A_i=R(\tau_i)-\frac{1}{G}\sum_{j=1}^{G}R(\tau_j).$$

必须等整组结束才能算 advantage（`assign_group_advantages()` 按 `question_index` 分组，组内中心化；全 0 则计为退化组）。**组内均值不能拆成 micro-batch 后再算**，否则比较基线已被改变。

**observation mask（核心机制）**：advantage 分配给轨迹中**每一轮 Assistant 生成的 token**；搜索 observation 只构成下一步决策的状态、**不是策略做出的动作**，因此其 logprob 与 advantage 都设为 0：

```python
full_tokens.extend(delta_observation)              # 观测区间
old_logprobs_by_token.extend([0.0] * len(delta_observation))
advantages_by_token.extend([0.0] * len(delta_observation))

full_tokens.extend(turn.completion_tokens)        # Assistant 动作区间
old_logprobs_by_token.extend(turn.logprobs)
advantages_by_token.extend([trajectory.advantage] * len(turn.completion_tokens))
```

**两个作用**：① 训练器不会要求模型"预测搜索引擎返回的文本"；② observation **仍保留在上下文**，后续 Assistant token 可对这些证据条件建模。

> **为什么 outcome reward 能训练工具使用策略**：搜索调用和最终回答共享一个**轨迹级 advantage**。答案正确 → 成功的查询规划、搜索词、答案 token 全部被鼓励；答案错误 → 整条动作链被抑制。

### 8.4.6 构造多轮 PyTRIO Datum 与一次参数更新

`build_datum()`（`docs/chapter8/search-r1/train.py:49`）完整核心：

```python
def build_datum(trajectory: Trajectory) -> TrainingDatum:
    full_tokens, old_logprobs_by_token, advantages_by_token = [], [], []
    assistant_token_count = 0
    for turn_index, turn in enumerate(trajectory.turns):
        if turn_index == 0:
            delta_observation = turn.prompt_tokens           # system + 工具定义 + 问题
        elif turn.prompt_tokens[:len(full_tokens)] == full_tokens:
            delta_observation = turn.prompt_tokens[len(full_tokens):]
        else:
            raise ValueError("prompt 不是已有轨迹的前缀扩展，无法对齐 logprob")

        full_tokens.extend(delta_observation)
        full_tokens.extend(turn.completion_tokens)
        old_logprobs_by_token.extend([0.0] * len(delta_observation))   # 观测：logprob=0
        old_logprobs_by_token.extend(turn.logprobs)
        advantages_by_token.extend([0.0] * len(delta_observation))      # 观测：advantage=0
        advantages_by_token.extend([trajectory.advantage] * len(turn.completion_tokens))
        assistant_token_count += len(turn.completion_tokens)

    if assistant_token_count == 0:
        raise ValueError("没有 assistant token，无法构造 Datum")
    # 对完整序列统一右移
    input_tokens  = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    old_logprobs  = old_logprobs_by_token[1:]
    advantages    = advantages_by_token[1:]
    if len(input_tokens) > MAX_TRAIN_CONTEXT_TOKENS:       # 8192
        raise ValueError("Datum 超过最大 token 数")
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens, dtype=np.int64),
            "logprobs":      np.asarray(old_logprobs,  dtype=np.float32),
            "advantages":    np.asarray(advantages,    dtype=np.float32),
        })
```

要点：第一轮 `delta_observation` 是 system+工具+问题，后续轮次是"上一轮结束符 + 搜索结果 + 新一轮 Assistant 前缀"。它们保留真实 target token，但 old logprob/advantage 为 0；**只有 sampler 真生成的 Assistant token 携带 rollout old logprob 与轨迹 advantage**。

**外层循环（`train.py:559`，顺序不可交换）**：

```python
batch = take_batch(examples, step * args.questions_per_batch, args.questions_per_batch)
sampling_client = training_client.save_weights_and_get_sampling_client()   # ① 导出当前权重
trajectories = rollout_batch(sampling_client=sampling_client, ...)          # ② 整组 reward+advantage
datums = build_training_datums(trajectories)                               # 跳过 advantage 全 0 轨迹
micro_batches = pack_micro_batches(datums)
for micro_batch in micro_batches:
    weighted = weight_micro_batch_for_global_mean(micro_batch, total_samples=len(trajectories))
    training_client.forward_backward(weighted, loss_fn="importance_sampling").result()
if micro_batches:
    training_client.optim_step(adam_params).result()                       # ③ 整个 logical batch 只一次
```

三条不可交换原因：① `save_weights` 保证 old logprob 来自更新前当前策略；② `build_training_datums` 已跳过退化组；③ 所有 micro-batch **只累计梯度**，最后只调一次 `optim_step()`。

**micro-batch 加权公式（保证全局样本平均梯度语义）**：若第 $k$ 个 micro-batch 有 $n_k$ 条轨迹，完整 batch 有 $N$ 条，把该批 advantage 乘 $n_k/N$：

$$\sum_k\frac{n_k}{N}\operatorname{mean}(\mathcal L_k)=\operatorname{mean}(\mathcal L_{\mathrm{global}}).$$

只重新包装 `Datum`、**不改 target token 和 old logprob**；动态拆批只改 padding 与显存，不改逻辑 batch 的样本平均梯度。对应地指标合并时 `loss_mean` 用 **sum**（因已预加权），其余用 mean。

> **本节收束**：Search-R1 = GRPO（分组采样 + 组内优势）+ 搜索环境（observation）+ observation mask。模型在 rollout 中自己学会"何时搜、搜什么"，最终答案的 outcome reward 通过共享 advantage 把成功搜索路径强化进策略。

---

## 8.5 ReTool：让 LLM 学会调用代码工具

### 8.5.1 从工具调用格式到工具使用策略

**动机**：搜索工具帮模型获取外部知识，**代码解释器**帮模型完成精确计算、符号操作与枚举验证。模型"会调用"≠"会在合适时机调用"——有些问题口算更快，有些适合写短程序，有些代码执行失败需据错误信息修正。ReTool 用 RL 训练这种"策略性工具使用"能力：模型自主决定**是否调用、写什么代码、如何读结果、何时给最终答案**。环境只按最终数学答案给结果奖励。

⚠ **安全提示（原文）**：配套 `sandbox.py` 只是**带资源限制的本地 subprocess，不是可信安全沙箱**。模型生成的代码仍能读当前账号可访问的文件、继承环境变量、访问网络。真实训练应放一次性容器/低权限虚拟机/专用沙箱服务，并**移除 PyTRIO、SwanLab、SSH 和云凭证**。

典型轨迹：问"满足条件的整数解数量"→ `code_interpreter("count=...; print(count)")` → 返回 `37` → 答 `\boxed{37}`。自我修正轨迹：`print(sum(values))` 报 `NameError` → 改写为 `values=[...]; print(sum(values))` → 返回 `128` → `\boxed{128}`。**RL 视角拆解**：代码、执行错误、数值结果、最终回答**共同构成轨迹**；代码解释器是环境，Assistant 生成的工具调用与自然语言答案都是动作，stdout/stderr 是 observation。不为"调用了工具"单独加分——最终答案 reward 通过**轨迹级 advantage** 传到前面的代码动作。

### 8.5.2 Cold-start、数据与工具协议

**原始 ReTool 两阶段**：① **Cold-start SFT**（用带代码调用与执行反馈的轨迹让模型先掌握协议）；② **RL**（模型自主 rollout，用最终答案 reward 学"何时/如何用工具"）。Cold-start 让初始策略产生一定比例有效工具调用——若基础模型完全不会输出工具协议，早期轨迹几乎全失败，GRPO 难获探索信号。

**【教学实现边界】**：本章选已具原生 function calling 的 Qwen3.5，**直接展示第二阶段 PyTRIO 训练，无 Cold-start SFT**。换成缺工具能力的基础模型时，**应先备 SFT 数据、验证合法调用率达可训练水平，再启 RL**。

RL 数据来自 DAPO-Math-17k，原始题目外带 `Answer: $Answer` 模板与 `\boxed{}` 冲突，需先剥外层：

```python
def strip_dapo_template(question: str) -> str:
    if question.startswith(PROMPT_PREFIX):
        question = question[len(PROMPT_PREFIX):]
    if question.endswith(PROMPT_SUFFIX):
        question = question[: -len(PROMPT_SUFFIX)]
    return question.strip()
```

训练样本**不含标准代码与工具调用次数**，策略须自己探索。`CODE_TOOL` 定义 `code_interpreter`（每次执行是全新独立进程，变量/文件/状态不保留）。`parse_assistant` 只校验调用数量、位置、代码是否非空（Python 可含比较符/引号/多行）；`tool` → 进执行环境，`answer` → 结束受奖励，`invalid` → 结束按错误答案处理。系统不为"格式接近但不完整"的调用做猜测性修复（否则动作定义会变不稳定）。

### 8.5.3 执行结果怎样接回真实 token 轨迹

ReTool 同样不能把 Assistant 文本 decode 后重 tokenize。与 Search-R1 差别是**用占位内容 `"x"` 定位 assistant 结束边界**（Qwen chat template 可能把真实采样文本中的 reasoning 与 content 重新分段，`</think>` 被重构，用真实文本定位不可靠）：

```python
def build_next_prompt(tokenizer, messages_before_assistant, previous_prompt_tokens,
                      completion_tokens, next_tool_message) -> list[int]:
    canonical_prompt = build_prompt(tokenizer, messages_before_assistant)
    placeholder_message = {"role": "assistant", "content": "x"}
    canonical_assistant_end = _render_chat(tokenizer,
        [*messages_before_assistant, placeholder_message], add_generation_prompt=False)
    assistant_closing_tokens = canonical_assistant_end[len(canonical_prompt):]

    canonical_next_prompt = build_prompt(tokenizer,
        [*messages_before_assistant, placeholder_message, next_tool_message])
    observation_tokens = canonical_next_prompt[len(canonical_assistant_end):]
    overlap = _suffix_prefix_overlap(completion_tokens, assistant_closing_tokens)
    return [*previous_prompt_tokens, *completion_tokens,
            *assistant_closing_tokens[overlap:], *observation_tokens]
```

占位消息**只算固定模板边界**，真实模型动作始终来自 `completion_tokens`。下一轮 `prompt_tokens` 必须以前一轮完整 token 为前缀，否则 `train.py` 拒绝构造 Datum。

### 8.5.4 Outcome reward 与 interpreter feedback mask

`reward.py` 取回答末尾 300 字符内最后一个 `\boxed{}`（按花括号配平，支持嵌套），并用 `math-verify` 做数学等价判定：

```python
ANSWER_WINDOW_CHARS = 300

def extract_last_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    index = text.rfind(marker)
    if index < 0:
        return None
    start, depth = index + len(marker), 1
    for position in range(start, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[start:position]
    return None

def answers_equivalent(prediction: str, reference: str) -> bool:
    try:
        return bool(verify(parse(f"${reference}$"), parse(f"${prediction}$")))
    except Exception:
        return False

def score_answer(text: str, reference: str) -> RewardResult:
    answer = extract_last_boxed(text[-ANSWER_WINDOW_CHARS:])
    if answer is None:
        return RewardResult(-1.0, False, False, None)
    correct = answers_equivalent(answer.strip(), reference.strip())
    return RewardResult(1.0 if correct else -1.0, correct, True, answer)
```

奖励函数：

$$R(\tau)=\begin{cases}+1, & \text{最终答案正确},\\ -1, & \text{答案错误或格式无效}.\end{cases}$$

同题中心化 advantage：

$$A_i=R(\tau_i)-\bar R.$$

**interpreter feedback mask 表（必背）**

| token 区域 | 是否留上下文 | old logprob | advantage |
| --- | --- | --- | --- |
| 初始 prompt | 是 | 0 | 0 |
| Assistant 推理与代码调用 | 是 | rollout logprob | 轨迹 advantage |
| stdout / stderr / 超时信息 | 是 | 0 | 0 |
| 最终答案 | 是 | rollout logprob | 轨迹 advantage |

**PPO 更新（非对称概率比边界）**：

```python
# 与官方 recipe 对齐的 PPO-clip 阈值（ε_low=0.2 / ε_high=0.28 的绝对值形式）
PPO_LOSS_CONFIG = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28}

training_client.forward_backward(
    datums, loss_fn="ppo", loss_fn_config=PPO_LOSS_CONFIG).result()
```

解读：$1-\epsilon_{\text{low}}=0.8$（$\epsilon_{\text{low}}=0.2$），$1+\epsilon_{\text{high}}=1.28$（$\epsilon_{\text{high}}=0.28$）。旧策略概率比低于 0.8 或高于 1.28 时 PPO 限制对应更新。**无论何种 clip，rollout sampler 都需在每个训练 step 开始前从当前权重刷新**。链路概括：**GRPO 构造相对优势，PPO clipping 控制策略更新**。

### 8.5.5 代码执行环境与安全边界（四道保险）

`sandbox.py` 设计：模型在 `<tool_call>` 中给的 Python 代码，用 `python -B -c <bootstrap>` 起**全新子进程**执行。list 形式 argv **不经过 shell**，代码经 `repr()` 内嵌引导串——不落脚本文件、无注入面；每次全新进程天然满足"执行独立无状态"。四道保险：

1. **限并发**：`threading.BoundedSemaphore`（默认 8）。**不用 `asyncio.Semaphore`**——它绑定单个事件循环，而训练循环每 step/round 新建 loop，跨 loop 复用会报错。
2. **限时间**：wall-clock 超时（默认 30s）后 `os.killpg` 杀**整个进程组**（`start_new_session=True` 才能整组杀，防 fork 孤儿）；另有引导串里 `RLIMIT_CPU` 作内核级 CPU 兜底。
3. **限输出**：stdout/stderr 写**匿名临时文件**而非 PIPE（防 print 洪水撑爆父进程内存）；读回只取**文件尾部**并按 4096 字符截断——执行结果与 traceback 关键信息都在末尾。
4. **限线程**：子进程把 `OMP/OpenBLAS/MKL/VECLIB` 线程数压到 1。

```python
DEFAULT_TIMEOUT = 30.0
MAX_OUTPUT_CHARS = 4096
_BOOTSTRAP = ("import resource\n"
              "resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))\n"
              "exec(compile({code}, '<sandbox>', 'exec'))\n")
_CHILD_ENV_OVERRIDES = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}

def run_code(self, code: str) -> ExecResult:
    bootstrap = _BOOTSTRAP.format(cpu=int(self.timeout) + 5, code=repr(code))
    env = {**os.environ, **_CHILD_ENV_OVERRIDES}
    with self._semaphore:
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen([self.python, "-B", "-c", bootstrap],
                stdout=stdout_file, stderr=stderr_file, env=env, start_new_session=True)
            try:
                returncode = process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            stdout = _truncate_tail(_read_capped(stdout_file))
            stderr = _truncate_tail(_read_capped(stderr_file))
    ok = returncode == 0
    return ExecResult(ok, stdout, stderr, returncode, timed_out, latency)
```

结果哲学：成功回 stdout（**无 print 则为空串**，官方语义）；失败回**原始 stderr**（完整 traceback）；超时回超时说明。报错不修饰、原样回喂——论文"自我纠错"行为正是从"看错误反馈→改代码"中涌现。`format_tool_content()` 把结果统一转成 tool observation；当前 `run_code()` **不自动补 `print()`**，故 system prompt 要求模型打印想看的值。

**安全要求 6 条（必须遵守）**：① 不在含生产凭证/私钥/敏感数据的主机直接执行模型代码；② 用一次性容器/低权限虚拟机/专用沙箱；③ 默认关网络、设只读文件系统与最小化可见目录；④ 移除无关环境变量与云凭证；⑤ 为 CPU/内存/进程数/磁盘/执行时间设硬限制；⑥ 保存代码、退出状态与资源指标便于审计。"independent execution" 只表示每次调用不共享 Python 状态，**不代表宿主机级安全隔离**。

### 8.5.6 多轮代码 rollout 与训练

三类数据分离：`AssistantTurn`（可训练动作）、`Trajectory`（完整状态）、`PendingExecution`（已生成待执行）。`begin_advance()` 消费 Assistant 输出，可调用且预算未耗尽则挂起 `PendingExecution`，否则标记 `done`。同一 round 内多条轨迹**并发执行**代码，但每条轨迹须等自己 observation 返回才能开始下一轮：

```python
async def execute_pending_async(pendings, sandbox):
    return list(await asyncio.gather(*(sandbox.arun_code(p.code) for p in pendents)))

async def advance_round(responses, requests, trajectories, tokenizer, sandbox, config, cb):
    pendings = [begin_advance(...) for ... if 可调用]
    if not pendings:
        return
    results = await execute_pending_async(pendings, sandbox)
    for pending, result in zip(pendings, results, strict=True):
        finish_advance(pending, result, tokenizer, sandbox, config)   # fit_tool_content 反复保留尾部 70%
```

与 Search-R1 差异：Search-R1 逐条丢弃证据，ReTool **反复保留尾部**（traceback 与最终数值通常在末尾）。`fit_tool_content` 截断时保留 `content[-70%]` 并加 `[... truncated ...]`。首轮 `num_samples=group_size` + 深拷贝分叉，后续 `num_samples=1`；所有轨迹结束后才 `score_trajectory()` + `assign_group_advantages()`。

`build_datum()` 与 8.4.6 同一套前缀扩展规则：`delta_observation`（含初始题、Assistant 结束符、stdout/stderr/timeout 文本）advantage 全 0；Assistant 推理、代码调用与最终答案共享轨迹 advantage——**这就是 interpreter feedback mask 的代码形态**。最后 micro-batch 装箱 + PPO 更新，加权公式与 Search-R1 相同：

$$\sum_k \frac{n_k}{N}\mathrm{mean}(\mathcal L_k) = \mathrm{mean}(\mathcal L_{\mathrm{global}}).$$

**对照实验原则**：text-only 与 ReTool 须用**同一批问题、相同采样参数、相同答案判定器**才能看出工具环境净收益；工具模式还应报告**额外延迟与执行成本**（准确率提升常伴随更多环境调用）。

> **本节收束**：ReTool = GRPO 分组采样 + 代码环境（observation）+ interpreter feedback mask + 非对称 PPO clip。模型在 rollout 中自己学会"何时写程序、写什么、如何从报错自我修正"，最终答案的 outcome reward 通过共享 advantage 把成功代码路径强化进策略。

---

## 关键代码速览

本章四示例共享同一套 PyTRIO 抽象。下表列出**最核心、必须读懂**的代码骨架与相对路径：

| 主题 | 关键函数 / 文件 | 作用 | 核心一行 |
| --- | --- | --- | --- |
| GRPO | `grpo/01-demo-sync.py:run_rollout_group` | 同题一组采样 + 组内 advantage | `advantage = reward - mean_reward` |
| GRPO | `grpo/01-demo-sync.py:build_grpo_datum` | 右移对齐 prompt/completion | `observation_len = len(prompt)-1` |
| OPD | `opd/01-demo-sync.py:completion_teacher_logprobs` | Teacher 评价同一条 Student 轨迹 | `all_logprobs[start:end]` |
| OPD | `opd/01-demo-sync.py` 主循环 | reverse KL → 逐 token advantage | `advantages = -β*(student_lp - teacher_lp)` |
| Search-R1 | `search-r1/rollout.py:consume_assistant` | 解析动作、决定继续搜还是回答 | `can_search = parsed.kind=="tool" and ...` |
| Search-R1 | `search-r1/train.py:build_datum` | 多轮 Datum + observation mask | `obs: logprob=0, adv=0；assistant: 真值` |
| Search-R1 | `search-r1/train.py:559` 外层循环 | rollout→advantage→micro-batch→optim | `save_weights → rollout → optim_step` |
| ReTool | `retool/sandbox.py:run_code` | 四道保险执行代码 | `os.killpg(pid, SIGKILL)` 超时 |
| ReTool | `retool/train.py` 主循环 | feedback mask + 非对称 PPO | `loss_fn="ppo", loss_fn_config={0.8,1.28}` |

**统一 Datum schema（importance_sampling / ppo 共用）**：
```python
trio.Datum(
    model_input=trio.ModelInput.from_ints(input_tokens),   # 长度 L = m+T-1（单轮）或整条轨迹-1（多轮）
    loss_fn_inputs={
        "target_tokens": np.asarray(target_tokens, dtype=np.int64),    # ℤ^L，右移一位
        "logprobs":      np.asarray(old_logprobs, dtype=np.float32),   # ℝ^L，环境 token 填 0
        "advantages":    np.asarray(advantages,   dtype=np.float32),   # ℝ^L，环境 token 填 0
    })
```
张量维度变化：单轮时 `L = m + T - 1`；多轮时 `L = 整条轨迹 token 数 - 1`，其中 prompt/observation 段 `logprobs/advantages` 为 0，仅 Assistant 生成的 token 携带 rollout logprob 与轨迹 advantage。右移（`input=seq[:-1]`, `target=seq[1:]`）使"用位置 $t$ 预测 $t+1$"。

---

## 实践清单

### 环境依赖与版本（考点：pytrio 与 swanlab 锁死）

| 包 | 版本约束 | 用途 |
| --- | --- | --- |
| `pytrio` | `==0.2.6`（**锁死**） | 远端训练/采样 SDK |
| `swanlab` | `==0.9.2`（**锁死**） | 实验记录 |
| `torch` | `>=2.9.1` | 本地张量/类型 |
| `datasets` | `>=5.0.0` | GSM8K / 本地 parquet |
| `modelscope` | `>=1.38.1` | DeepMath-103K、NQ/HotpotQA 下载 |
| `math-verify` | `>=0.9.0` | ReTool 数学等价判定 |
| `deepseek-search` | git+https://github.com/KMnO4-zx/deepseek-search.git@6215c8db | Search-R1 DeepSeek 后端 |
| `httpx` / `matplotlib` / `numpy` / `python-dotenv` / `tqdm` | 见 `requirements.txt` | 搜索后端/绘图/数组/.env/进度条 |

**登录与建环境（从仓库根目录）**：
```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r docs/chapter8/requirements.txt
trio login          # 在线记录实验再 swanlab login
```
查看当前 PyTRIO 支持模型（模型列表持续更新，正式实验前必查）：
```python
import pytrio as trio
print(trio.ServiceClient().get_supported_models())
```

### 四组最小跑通命令

```bash
# 1) GRPO（GSM8K，单步最小）
python docs/chapter8/grpo/01-demo-sync.py \
    --steps 1 --batch-size 1 --group-size 4 --max-tokens 512 \
    --loss-fn importance_sampling --swanlab-mode disabled

# 2) OPD（DeepMath-103K，自动下 1 分片）
python docs/chapter8/opd/01-demo-sync.py \
    --steps 1 --batch-size 1 --group-size 1 --max-tokens 512 \
    --sample-size 20 --num-shards 1 --swanlab-mode disabled

# 3) Search-R1（Wikipedia 免密钥验证链路）
python docs/chapter8/search-r1/prepare_data.py
python docs/chapter8/search-r1/train.py \
    --max-steps 1 --questions-per-batch 1 --group-size 4 \
    --max-search-calls 2 --search-backend wikipedia \
    --search-concurrency 3 --swanlab-mode disabled

# 4) ReTool（先确认执行器在隔离环境！）
python docs/chapter8/retool/prepare_data.py
python docs/chapter8/retool/train.py \
    --max-steps 1 --questions-per-batch 1 --group-size 4 \
    --max-code-calls 2 --sandbox-workers 2 --swanlab-mode disabled
```

### 数据集来源与评测

| 主题 | 训练数据 | 评测数据 | 答案判定 |
| --- | --- | --- | --- |
| GRPO | GSM8K（`openai/gsm8k`） | 同域 held-out | `\boxed{}` 字符串相等 |
| OPD | DeepMath-103K（`AI-ModelScope/DeepMath-103K`） | 独立任务/通用能力 | prompt-only，无标准答案 |
| Search-R1 | NQ + HotpotQA（`zhuangzhuang2023/nq_hotpotqa_train`） | dev 70 条等量少采样 | `Answer:` 行 EM |
| ReTool | DAPO-Math-17k | 固定 30 道 AIME 2025（`yentinglin/aime_2025`） | `\boxed{}` + math-verify |

### 默认超参与指标

四示例共同默认：`--base-model Qwen/Qwen3.5-4B`、`--lora-rank 32`、`--group-size 8`、`--max-tokens 1024`、`--temperature 1.0`、`--top-p 1.0`、`--learning-rate 4e-5`、`--beta1 0.9`、`--beta2 0.95`、`--seed 42`。Search-R1 / ReTool 额外：`--questions-per-batch 8`、`--max-assistant-turns 6`、`--max-trajectory-tokens 8192`；Search-R1 `--max-tool-response-tokens 1024`，ReTool `512` + `--sandbox-timeout 30 --sandbox-workers 8`。

关键观察指标：GRPO 看 `reward / frac_degenerate / rollout/avg_gen_len / train_tokens`；Search-R1 看 `reward/mean / rollout/search_calls / rollout/degenerate_group_rate`；ReTool 看 `Pass@N / format_rate / mean_code_calls / clip fraction / sandbox timeout_rate`。**退化组比例长期偏高**是通用红灯，需调难度/温度/模型/group size。

### 安全要求（ReTool 尤其）

执行模型代码前必须放入隔离环境（容器/低权限 VM/专用沙箱），移除 PyTRIO、SwanLab、SSH、云凭证，关网络、设只读文件系统、限制 CPU/内存/进程/磁盘/执行时间，并保存代码与退出状态便于审计。`sandbox.py` 的本地 subprocess **不是可信安全隔离**。

---

## 本章小结

**核心数据流（收束公式）**：
$$\text{当前策略}\xrightarrow{\text{rollout}}\text{多样化轨迹}\xrightarrow{\text{环境/Teacher}}\text{训练信号}\xrightarrow{\text{token-level advantage}}\text{策略更新}\xrightarrow{}\text{新策略}.$$

**四个模块的分工与共性**：
- **GRPO**：同题多次采样构造组内相对优势，**省掉单独训练 Value Model**；算法三层——策略生成 rollout、环境给 reward、按新旧策略概率比优化。reward 正确性、token 对齐、on-policy sampler 刷新共同决定训练是否有效。
- **OPD**：保留 on-policy rollout，把规则奖励换成 **Teacher 对 Student 轨迹的逐 token 概率反馈**。reverse KL 为每动作提供稠密信号，使小模型能在**自身会访问的状态**上学 Teacher 能力。Teacher/Student 的 tokenizer、推理模式、能力互补需训练前验证。
- **Search-R1 与 ReTool**：把轨迹扩展为**多轮环境交互**。搜索结果/代码结果**进入上下文**却通过 **observation mask / feedback mask 排除在策略 loss 外**；模型生成的搜索词、代码、推理与最终答案**共享轨迹级结果信号**。两者都复用 GRPO 分组采样与相对优势，只替换工具协议、环境实现与 reward。

**与前后章的联系（承上启下）**：
- **承上**：第 4 章的 RLHF / PPO / 奖励模型是本章数学基础——GRPO 仍是策略梯度 + 概率比 clip，只是把"奖励模型"换成"可验证环境"、把"单轮回答"换成"多轮轨迹"；第 7 章的 Agent/Function Calling/工具协议是 Search-R1 与 ReTool 的直接前置，本章只是把"人写死的触发逻辑"交给了"强化学习去自主优化"。
- **启下**：本章打开"模型在环境中自我改进"的大门。后续可扩展 OPSD、DAPO、GSPO、ALFWorld 等更前沿算法（见 agentic-rl-lab），以及把"检索器本身也纳入学习"的多模块 credit assignment。

**设计新 Agentic-RL 任务的五问清单（最有迁移价值）**：
1. Agent 能执行哪些动作，环境返回什么 observation？
2. 一条轨迹在什么条件下结束，预算如何限制？
3. reward 能否稳定反映最终任务目标？
4. 哪些 token 属于策略动作，哪些必须被 mask？
5. rollout、旧 logprob、advantage 与当前权重是否严格对齐？

> 当这五问都有可验证答案时，一个工具 Demo 才具备进入强化学习训练的基础。

---

## 习题

1. **【概念辨析】** 有人说"GRPO 是一种比 PPO 更好的策略优化算法"。这句话哪里不严谨？请结合 8.2.4 节说明 GRPO 与 PPO 各自负责什么。
   > **答案提示**：不严谨。GRPO 描述的是"如何分组采样并由组内奖励得到 advantage"；PPO（或 importance sampling）描述的是"如何把 advantage 转成策略梯度"。两者是不同层次：GRPO 可配 PPO loss（本章 ReTool 正是如此）。更准确说法是"GRPO 用组内相对比较替代了 critic 的价值估计"。

2. **【概念辨析】** 为什么 GRPO / Search-R1 / ReTool 都强调"advantage 必须等整组（整条轨迹集合）结束后才算"，而不能在单条轨迹一生成完就更新？
   > **答案提示**：组内优势依赖"同题所有轨迹的组内均值"作为基线；单条轨迹一结束就更新，基线会随已处理轨迹数变化，比较尺度不一致，梯度方向失真。且 on-policy 要求本轮所有旧 logprob 来自同一份当前权重。

3. **【推导/计算】** 一道题采样 6 个回答，规则奖励为 $[1,1,0,0,0,1]$。用本章简化式 $A_i=r_i-\bar r$ 计算每条轨迹的 advantage，并指出是否存在退化组。
   > **答案提示**：均值 $\bar r=3/6=0.5$。advantage = $[0.5,0.5,-0.5,-0.5,-0.5,0.5]$。非全 0，无退化组（若有组全为 1 或全为 0 则 advantage 全 0，属退化组）。

4. **【推导/计算】** OPD 中某 token 的 Student 旧 logprob 为 $-2.0$，Teacher logprob 为 $-1.0$，$\beta=1.0$。求该 token 的 $d_t$ 与 advantage $A_t$，并说明这一步在鼓励还是抑制该 token。
   > **答案提示**：$d_t=\log\pi_{\theta_{old}}-\log\pi_T=(-2.0)-(-1.0)=-1.0<0$，说明 Teacher 比 Student 更认可该 token；$A_t=-\beta d_t=1.0>0$，鼓励该 token（让 Student 向 Teacher 靠拢）。

5. **【概念辨析】** observation mask / feedback mask 的作用是什么？如果"不 mask"、把搜索结果/代码执行结果的 token 也赋予真实 advantage 会怎样？
   > **答案提示**：mask 让环境返回的 token 保留在上下文（供后续条件建模）但其 logprob/advantage 置 0，训练器不会要求模型"预测外部文本"。若不 mask，模型会被迫去学习"复现搜索引擎/解释器的输出"，既不可能（环境文本非策略动作）又浪费梯度，且会把结果信号错误地归因到环境 token 上。

6. **【动手实践】** 跑通 Search-R1 的 Wikipedia 最小试跑（见实践清单命令 3）。观察日志中 `rollout/search_calls` 与 `reward/mean`：若 `frac_degenerate` 很高，按本章建议你会先调哪几个超参？
   > **答案提示**：优先调 `group_size`、`temperature`、`max-search-calls`，以及检查题目难度是否过易/过难导致全对或全错；也可换更强基础模型。

7. **【动手实践】** ReTool 的安全边界要求把模型代码放进隔离环境。请列出至少 4 条你会在启动 `train.py` 前落实的具体安全措施，并解释 `sandbox.py` 用 `threading.BoundedSemaphore` 而非 `asyncio.Semaphore` 的原因。
   > **答案提示**：移除云凭证/关网络/只读文件系统/限制 CPU 内存与执行时间/一次性容器。用 `threading.BoundedSemaphore` 是因训练循环每 step/round 新建事件循环，`asyncio.Semaphore` 绑定单 loop 跨 loop 复用会报错。

8. **【概念辨析】** 为什么 OPD 强调"Teacher 必须评价 Student 自己生成的同一条轨迹"，而不能让 Teacher 先生成一条回答再让 Student 学？
   > **答案提示**：让 Teacher 生成会退回"合成数据离线蒸馏"，Student 在自己会访问的状态（尤其分布外错误分支）上得不到 Teacher 指导；OPD 的 on-policy 意义正在于信号落在 Student 实际轨迹上。

---

## 延伸阅读

**原教程对应章节**：
- Datawhale happy-llm 第 8 章《大模型强化学习（Agentic RL）》：`docs/chapter8/第八章 大模型强化学习.md`（含 `grpo` / `opd` / `search-r1` / `retool` 四套代码与 README）
- 第 4 章《大语言模型》（RLHF / 奖励模型 / PPO）与第 7 章《大模型应用》（Agent / Function Calling / RAG）是本章前置
- 作者维护的扩展仓库 [agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab)（含 OPSD、DAPO、GSPO、ALFWorld 等）

**关键论文与文档**：
1. Shao, Z. et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. 2024. https://arxiv.org/abs/2402.03300 （GRPO 出处）
2. Schulman, J. et al. *Proximal Policy Optimization Algorithms*. 2017. https://arxiv.org/abs/1707.06347 （PPO clip）
3. Agarwal, R. et al. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*. 2023. https://arxiv.org/abs/2306.13649 （OPD 现象学）
4. Li, Y. et al. *Rethinking On-Policy Distillation of Large Language Models*. 2026. https://arxiv.org/abs/2604.13016
5. Jin, B. et al. *Search-R1: Training LLMs to Reason and Leverage Search Engines with RL*. 2025. https://arxiv.org/abs/2503.09516
6. Feng, J. et al. *ReTool: Reinforcement Learning for Strategic Tool Use in LLMs*. 2025. https://arxiv.org/abs/2504.11536
7. PyTRIO 文档：*Loss Function Guide*（https://docs.pytrio.com/docs/guide/loss_fn）、*Search-R1 Example*（https://docs.pytrio.com/docs/example/search-r1）

**下一步学习建议**：读懂本章后，建议动手把四个最小示例各跑一步，重点观察 `reward` 与 `frac_degenerate` 曲线；再选一个你熟悉的封闭环境（如本地计算器、固定题库）尝试用"五问清单"设计一个自己的 Agentic-RL 任务，体会 reward 设计与 token 对齐为何是成败关键。
