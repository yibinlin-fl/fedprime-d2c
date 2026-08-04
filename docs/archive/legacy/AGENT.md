# FedPRIME-D2C Project Instructions

## Project Goal

This repository implements FedPRIME-D2C: Distribution-Decoupled Communication for PRIME-based Robust Heterogeneous Federated Learning.

The framework has two major parts:

1. Local PRIME robust learning:
   - PRIME is only used as a local robust augmentation module.
   - Do not redesign PRIME as a mechanism probe.
   - Do not add prototype communication.
   - Do not add feature-level alignment unless explicitly requested.

2. D2C communication:
   - Use public logits for model-heterogeneous communication.
   - Implement prior debiasing, class-balanced aggregation, and personalized complementary distillation.
   - Do not use FedAvg parameter aggregation.
   - Do not use FedProto-style prototypes.

## Coding Rules

- Keep changes small and modular.
- Before modifying files, inspect relevant files and explain the plan.
- After modifying files, run the relevant tests.
- Prefer clear, readable PyTorch code over overly abstract code.
- Add type hints where helpful.
- Avoid hard-coded paths.
- Use YAML configs for experiment settings.
- Keep all tensor shapes documented in comments.
- Every new module should have at least one unit test or smoke test.

## Core Formulas

Prior estimation:

pi_hat[k, c] = mean_u softmax(logits_k(u) / tau)[c]

Prior debiasing:

z_tilde[k, u, c] = z[k, u, c] - beta_k * log(pi_bar[k, c] + eps)

Debiased probability:

p_tilde[k, u, c] = softmax(z_tilde[k, u, :] / tau)[c]

Class coverage:

q[k, c] = (pi_bar[k, c] + eps)^eta / sum_j (pi_bar[j, c] + eps)^eta

Sample confidence:

r[k, u] = 1 - H(p_tilde[k, u, :]) / log(C)

Aggregation weight:

a[k, u, c] = q[k, c] * r[k, u]

Teacher:

T[u, c] = normalize_c(sum_k a[k, u, c] * p_tilde[k, u, c])

Complementary weight:

m[i, c] = (1 - pi_bar[i, c])^rho

D2C loss:

L_D2C = tau^2 * sum_u sum_c m[i, c] * T[u, c] * log(T[u, c] / p_i[u, c])

## Validation Commands

Run these when relevant:

```bash
python -m pytest tests -q
python src/main.py --config configs/cifar10_debug.yaml
```



下面把\*\*当前最终版框架\*\*完整整理出来。为了避免再绕回 FedProto、RAHFL 通信或者复杂 PRIME 探针，这版框架严格遵守你的诉求：



> \*\*PRIME 只作为底层鲁棒学习模块；不做原型、不做特征对齐、不做机制探针；真正创新点放在一个轻量的 public-logit 异构通信模块上，并显式支持数据异构。\*\*



框架暂定名：



\# FedPRIME-D2C



\*\*Distribution-Decoupled Communication for PRIME-based Robust Heterogeneous Federated Learning\*\*



中文名：



> \*\*基于 PRIME 与分布解耦通信的模型异构鲁棒联邦学习框架\*\*



\---



\# 1. 框架一句话概括



\*\*FedPRIME-D2C 用 PRIME 作为每个客户端的本地鲁棒训练底座，用 D2C 分布解耦通信模块实现模型异构下的 public logits 蒸馏，并通过先验去偏与个性化互补蒸馏显式缓解 Non-IID 数据异构。\*\*



也就是：



```text

PRIME 负责：抗 common corruptions

D2C 负责：模型异构通信 + 数据异构去偏

```



不是：



```text

PRIME + RAHFL 通信

```



而是：



```text

PRIME + 自己的 Distribution-Decoupled Communication

```



\---



\# 2. 研究问题设定



有 (K) 个联邦客户端，每个客户端 (k) 持有私有数据：



\[

D\_k = {(x\_i^k, y\_i^k)}\_{i=1}^{n\_k}

]



整个任务同时存在三个挑战。



\---



\## 2.1 模型异构



不同客户端模型结构不同：



\[

f\_1 \\neq f\_2 \\neq \\cdots \\neq f\_K

]



例如：



```text

Client 1: ResNet-18

Client 2: MobileNetV2

Client 3: ShuffleNetV2

Client 4: VGG-small

Client 5: CNN-small

Client 6: ViT-Tiny

```



因此不能使用 FedAvg，因为参数维度和结构不一致。



\---



\## 2.2 数据异构



每个客户端的数据分布不同：



\[

P\_k(y) \\neq P\_j(y)

]



典型情况是 Dirichlet label skew：



\[

\\alpha \\in {1.0, 0.5, 0.3, 0.1}

]



(\\alpha) 越小，Non-IID 越严重。



这会导致一个很严重的问题：



> 客户端在公共数据上的 logits 不只是语义知识，还混入了本地类别先验偏置。



比如客户端 A 本地大多数是猫狗，那么它即使面对公共图像，也更容易输出猫狗相关的高概率。这种偏置如果直接参与全局蒸馏，会污染其他客户端。



\---



\## 2.3 数据损坏 / common corruptions



训练数据或测试数据可能受到 common corruptions 影响，例如：



```text

Noise

Blur

Weather

Digital corruption

JPEG compression

Low contrast

Pixelate

Motion blur

Gaussian noise

```



目标是在 clean test 和 corrupted test 上都获得更好性能。



\---



\# 3. 框架整体架构



FedPRIME-D2C 分成两层。



```text

┌─────────────────────────────────────┐

│  上层：D2C 分布解耦异构通信模块       │

│  - public logits communication       │

│  - prior debiasing                   │

│  - class-balanced aggregation        │

│  - personalized complementary KD     │

└─────────────────────────────────────┘

&#x20;                 ↑

&#x20;                 │

┌─────────────────────────────────────┐

│  底层：Local PRIME Robust Learning   │

│  - PRIME augmentation                │

│  - CE loss                           │

│  - JSD consistency loss              │

└─────────────────────────────────────┘

```



底层 PRIME 只负责提升每个客户端自己的 corruption robustness。PRIME 本身是一个中心化 common corruption 鲁棒增强方法，利用 spectral、spatial、color 三类 maximum-entropy primitives 生成语义保持型扰动，并在多个 corruption benchmark 上超过 AugMix。



上层 D2C 是你的核心创新。它不传参数，不传原型，不传特征，只传公共数据上的 logits，因此天然支持模型异构。



\---



\# 4. 整体训练流程



每一轮联邦训练如下：



```text

Round t:



1\. Server 采样一批公共无标签数据 B\_pub

2\. Server 把 B\_pub 发给所有参与客户端

3\. Client 用本地私有数据 + PRIME 做鲁棒训练

4\. Client 在 B\_pub 上输出 logits

5\. Client 上传 public logits

6\. Server 估计每个客户端的预测先验 / 类别偏置

7\. Server 对客户端 logits 做 prior debiasing

8\. Server 构造 class-balanced global teacher

9\. Server 为每个客户端构造 personalized complementary teacher

10\. Client 在 public data 上做 D2C 蒸馏

11\. 进入下一轮

```



核心通信对象只有：



\[

z\_k(u) = f\_k(u)

]



其中 (u \\in B\_{pub})。



因此不同模型只要输出类别数一致，就可以参与通信。



\---



\# 5. 模块一：Local PRIME Robust Learning



每个客户端 (k) 有自己的模型：



\[

f\_k(x;\\theta\_k)

]



对本地样本 ((x,y))，生成两个 PRIME 增强视图：



\[

x\_1' = \\text{PRIME}(x)

]



\[

x\_2' = \\text{PRIME}(x)

]



然后计算：



\[

p\_k(y|x)=\\text{softmax}(f\_k(x))

]



\[

p\_k(y|x\_1')=\\text{softmax}(f\_k(x\_1'))

]



\[

p\_k(y|x\_2')=\\text{softmax}(f\_k(x\_2'))

]



本地鲁棒训练损失为：



\[

\\mathcal{L}\_{local}^{k}

=======================



\\mathcal{L}\*{ce}^{k}

\+

\\lambda\*{jsd}\\mathcal{L}\_{jsd}^{k}

]



其中：



\[

\\mathcal{L}\_{ce}^{k}

====================



CE(f\_k(x), y)

\+

CE(f\_k(x\_1'), y)

]



\[

\\mathcal{L}\_{jsd}^{k}

=====================



JS

\\left(

p\_k(y|x),

p\_k(y|x\_1'),

p\_k(y|x\_2')

\\right)

]



这里借鉴的是 AugMix/PRIME 中常见的一致性思想。AugMix 原文使用随机增强链、mixing 和 Jensen-Shannon consistency loss 来提升分布偏移下的鲁棒性和不确定性估计。



这一部分不是你的最大创新点，它的作用是：



> 让每个客户端先具备基本的 corruption-robust representation / classifier。



\---



\# 6. 模块二：Public Logits 异构通信



由于模型异构，不能聚合参数。



服务器维护一个无标签公共数据集：



\[

D\_{pub}={u\_m}\_{m=1}^{M}

]



每轮采样：



\[

B\_{pub}\\subset D\_{pub}

]



客户端 (k) 在公共数据上输出 logits：



\[

z\_k(u)=f\_k(u)

]



蒸馏温度为 (\\tau)，得到概率：



\[

p\_k(c|u)

========



\\text{softmax}

\\left(

\\frac{z\_k(u)}{\\tau}

\\right)\_c

]



客户端上传：



```text

public logits: z\_k(u), u ∈ B\_pub

```



标准版本不需要上传模型参数、不需要上传特征、不需要上传原型。



\---



\# 7. 模块三：D2C 分布解耦通信



D2C 是核心模块，全称：



> \*\*Distribution-Decoupled Communication\*\*



它解决的问题是：



> Non-IID 下，客户端 public logits 中混入本地类别先验偏置，直接平均会污染全局 teacher。



D2C 包含三个子模块：



```text

1\. Predictive Prior Estimation

2\. Local-prior Logit Debiasing

3\. Class-balanced Consensus Aggregation

4\. Personalized Complementary Distillation

```



严格来说是四步，其中第 1 步是前置估计。



\---



\# 8. 子模块 1：Predictive Prior Estimation



为了避免上传真实本地类别直方图，标准版不直接使用：



\[

\\pi\_{k,c}=

\\frac{n\_{k,c}}{\\sum\_{c'} n\_{k,c'}}

]



而是用客户端在公共数据上的平均预测来估计预测先验：



\[

\\hat{\\pi}\_{k,c}

===============



\\frac{1}{|B\_{pub}|}

\\sum\_{u\\in B\_{pub}}

p\_k(c|u)

]



为了稳定，可以用 EMA 更新：



\[

\\hat{\\pi}\_{k,c}^{(t)}

=====================



\\alpha\_{ema}\\hat{\\pi}\*{k,c}^{(t-1)}

\+

(1-\\alpha\*{ema})

\\frac{1}{|B\_{pub}|}

\\sum\_{u\\in B\_{pub}}

p\_k^{(t)}(c|u)

]



其中 (\\alpha\_{ema}) 可以取 0.9。



这表示客户端当前模型在公共数据上的类别输出倾向。



如果客户端长期偏向某些类别，说明它的 public logits 可能受本地数据分布影响。



\---



\## 8.1 平滑与裁剪



为了避免某些类别概率为 0 导致数值爆炸，对先验做 smoothing 或 clipping：



\[

\\bar{\\pi}\_{k,c}

===============



\\text{clip}(\\hat{\\pi}\*{k,c}, p\*{min}, p\_{max})

]



然后重新归一化：



\[

\\bar{\\pi}\_{k,c}

===============



\\frac{\\bar{\\pi}\*{k,c}}{\\sum\*{c'}\\bar{\\pi}\_{k,c'}}

]



其中：



```text

p\_min = 1e-3 或 1e-4

p\_max = 1.0

```



最终后续都使用 (\\bar{\\pi}\*{k,c})，而不是原始 (\\hat{\\pi}\*{k,c})。



\---



\## 8.2 Oracle 版本



实验中可以额外做一个 Oracle 版本，使用真实本地标签先验：



\[

\\pi\_{k,c}^{oracle}

==================



\\frac{n\_{k,c}}{n\_k}

]



但论文主方法建议使用 public-predicted prior：



\[

\\bar{\\pi}\_{k,c}

]



这样更隐私友好，也更不容易被审稿人攻击。



\---



\# 9. 子模块 2：Local-prior Logit Debiasing



这是 D2C 的灵魂。



对于客户端 (k)，公共样本 (u)，类别 (c)，原始 logit 为：



\[

z\_{k,c}(u)

]



D2C 对其做先验去偏：



\[

\\tilde{z}\_{k,c}(u)

==================



\## z\_{k,c}(u)



\\beta\_k \\log(\\bar{\\pi}\_{k,c}+\\epsilon)

]



然后得到去偏后的概率：



\[

\\tilde{p}\_k(c|u)

================



\\text{softmax}

\\left(

\\frac{\\tilde{z}\_k(u)}{\\tau}

\\right)\_c

]



这里：



\* (\\bar{\\pi}\_{k,c})：客户端预测先验；

\* (\\beta\_k)：去偏强度；

\* (\\epsilon)：防止数值问题；

\* (\\tau)：蒸馏温度。



直观理解：



```text

如果客户端长期偏向某些类别，

说明这些类别的 logit 可能包含本地先验偏置。



D2C 在聚合前减去 log prior，

让服务器看到更接近语义判断的 logits。

```



注意不要在论文里写得太绝对，不要说“严格等价于贝叶斯去先验”。更稳的说法是：



> Inspired by prior-shift correction, we subtract a scaled log predictive prior from client logits to alleviate local label-prior bias induced by Non-IID training data.



中文：



> 受先验偏移校正启发，我们从客户端 logits 中减去缩放后的预测先验项，以缓解 Non-IID 训练数据导致的本地类别先验偏置。



\---



\## 9.1 自适应去偏强度



如果客户端接近 IID，不需要强去偏；如果客户端严重 Non-IID，需要强去偏。



定义预测先验熵：



\[

h\_k

===



\\frac{

H(\\bar{\\pi}\_k)

}{

\\log C

}

]



其中：



\* (h\_k \\approx 1)：客户端预测分布较均衡；

\* (h\_k \\approx 0)：客户端预测严重偏向少数类别。



设置：



\[

\\beta\_k

=======



\\beta\_0(1-h\_k)

]



这样：



```text

客户端越偏，β\_k 越大；

客户端越均衡，β\_k 越小。

```



MVP 阶段可以先用固定 (\\beta)，例如：



\[

\\beta=0.5

]



完整版再加入 adaptive (\\beta\_k)。



\---



\# 10. 子模块 3：Class-balanced Consensus Aggregation



去偏后的客户端预测为：



\[

\\tilde{p}\_k(c|u)

]



服务器要把所有客户端的知识融合成一个全局 teacher：



\[

T(c|u)

]



最简单是平均：



\[

T(c|u)=\\frac{1}{K}\\sum\_k \\tilde{p}\_k(c|u)

]



但这仍然不够，因为不同客户端对不同类别的知识覆盖不同。



所以 D2C 使用 class-balanced aggregation。



\---



\## 10.1 类别覆盖权重



用预测先验 (\\bar{\\pi}\_{k,c}) 表示客户端 (k) 对类别 (c) 的覆盖倾向：



\[

q\_{k,c}

=======



\\frac{

(\\bar{\\pi}\*{k,c}+\\epsilon)^\\eta

}{

\\sum\*{j=1}^{K}

(\\bar{\\pi}\_{j,c}+\\epsilon)^\\eta

}

]



其中：



\[

\\eta \\in \[0,1]

]



\* (\\eta=0)：所有客户端对类别 (c) 权重相同；

\* (\\eta=1)：完全按预测覆盖程度加权；

\* 推荐 (\\eta=0.5)，避免某个客户端垄断某类知识。



这个权重表示：



> 对类别 (c)，哪个客户端更可能有相关知识。



\---



\## 10.2 样本置信度权重



客户端对公共样本 (u) 的置信度定义为：



\[

r\_k(u)

======



\## 1



\\frac{

H(\\tilde{p}\_k(\\cdot|u))

}{

\\log C

}

]



如果客户端对 (u) 的预测熵低，说明它更确定，(r\_k(u)) 更大。



\---



\## 10.3 样本-类别级聚合权重



对客户端 (k)、样本 (u)、类别 (c)，定义：



\[

a\_{k,c}(u)

==========



q\_{k,c}\\cdot r\_k(u)

]



这个权重不是 client-level，而是：



```text

client × sample × class level

```



也就是说，不是说“客户端 A 整体可靠”，而是：



> 对这个公共样本、这个类别，客户端 A 的知识是否更值得采用。



\---



\## 10.4 构造全局 teacher



先计算类别分数：



\[

s(c|u)

======



\\sum\_{k=1}^{K}

a\_{k,c}(u)\\tilde{p}\_k(c|u)

]



然后归一化：



\[

T(c|u)

======



\\frac{

s(c|u)

}{

\\sum\_{c'=1}^{C}s(c'|u)

}

]



这就是服务器生成的全局 teacher。



\---



\# 11. 子模块 4：Personalized Complementary Distillation



全局 teacher (T(c|u)) 不应该原样发给所有客户端。



因为 Non-IID 下，每个客户端缺的类别不同。



客户端 (i) 的预测先验为：



\[

\\bar{\\pi}\_{i,c}

]



定义客户端 (i) 对类别 (c) 的互补学习权重：



\[

m\_{i,c}

=======



(1-\\bar{\\pi}\_{i,c})^\\rho

]



其中 (\\rho > 0)。



如果客户端 (i) 很少预测类别 (c)，说明它可能缺少这个类别的知识，则：



\[

m\_{i,c} \\text{ 大}

]



如果客户端 (i) 已经经常预测类别 (c)，说明它对这个类别已有较多知识，则：



\[

m\_{i,c} \\text{ 小}

]



为了避免 loss scale 不稳定，做归一化：



\[

\\bar{m}\_{i,c}

=============



\\frac{

m\_{i,c}

}{

\\frac{1}{C}\\sum\_{c'=1}^{C}m\_{i,c'}

}

]



\---



\## 11.1 自保护样本门控



为了避免全局 teacher 把客户端已经学得很好的知识拉坏，引入 sample-level gate。



客户端 (i) 对公共样本 (u) 的不确定性：



\[

g\_i(u)

======



\\frac{

H(p\_i(\\cdot|u))

}{

\\log C

}

]



如果客户端对 (u) 很不确定，(g\_i(u)) 大，说明它更需要 teacher。



如果客户端对 (u) 已经很确定，(g\_i(u)) 小，说明不应该被 teacher 强行修改。



MVP 阶段可以先不加 (g\_i(u))，完整版再加。



\---



\## 11.2 个性化 D2C 蒸馏损失



客户端 (i) 的 D2C 蒸馏损失为：



\[

\\mathcal{L}\_{D2C}^{i}

=====================



\\tau^2

\\sum\_{u\\in B\_{pub}}

g\_i(u)

\\sum\_{c=1}^{C}

\\bar{m}\_{i,c}

T(c|u)

\\log

\\frac{

T(c|u)

}{

p\_i(c|u)

}

]



如果 MVP 不使用 sample gate，则令：



\[

g\_i(u)=1

]



得到简化版：



\[

\\mathcal{L}\_{D2C}^{i}

=====================



\\tau^2

\\sum\_{u\\in B\_{pub}}

\\sum\_{c=1}^{C}

\\bar{m}\_{i,c}

T(c|u)

\\log

\\frac{

T(c|u)

}{

p\_i(c|u)

}

]



这一步的含义是：



```text

客户端不是盲目学习全局 teacher；

而是优先学习自己缺失或薄弱的类别知识。

```



这就是数据异构功能的核心。



\---



\# 12. 客户端最终优化目标



每个客户端最终训练目标为：



\[

\\mathcal{L}\_{k}

===============



\\mathcal{L}\*{local}^{k}

\+

\\lambda\*{d2c}

\\mathcal{L}\_{D2C}^{k}

]



展开：



\[

\\mathcal{L}\_{k}

===============



\\mathcal{L}\*{ce}^{k}

\+

\\lambda\*{jsd}\\mathcal{L}\*{jsd}^{k}

\+

\\lambda\*{d2c}\\mathcal{L}\_{D2C}^{k}

]



其中：



\[

\\mathcal{L}\_{ce}^{k}

====================



CE(f\_k(x), y)

\+

CE(f\_k(\\text{PRIME}(x)), y)

]



\[

\\mathcal{L}\_{jsd}^{k}

=====================



JS

\\left(

p\_k(y|x),

p\_k(y|\\text{PRIME}\_1(x)),

p\_k(y|\\text{PRIME}\_2(x))

\\right)

]



\[

\\mathcal{L}\_{D2C}^{k}

=====================



\\tau^2

\\sum\_{u\\in B\_{pub}}

g\_k(u)

\\sum\_{c=1}^{C}

\\bar{m}\_{k,c}

T(c|u)

\\log

\\frac{

T(c|u)

}{

p\_k(c|u)

}

]



\---



\# 13. 完整算法描述



\## 13.1 Server 端算法



```text

Algorithm: FedPRIME-D2C Server



Input:

&#x20; K heterogeneous clients

&#x20; public unlabeled dataset D\_pub

&#x20; communication rounds R

&#x20; temperature τ

&#x20; debias strength β or β0

&#x20; aggregation exponent η

&#x20; complementary exponent ρ



Initialize:

&#x20; each client initializes its own model f\_k



For each round t = 1,...,R:



&#x20; 1. Server samples public batch B\_pub from D\_pub



&#x20; 2. Server sends B\_pub to selected clients S\_t



&#x20; 3. Each client k ∈ S\_t performs:

&#x20;      - local PRIME robust training

&#x20;      - computes logits z\_k(u) for u ∈ B\_pub

&#x20;      - uploads logits z\_k(u)



&#x20; 4. Server computes predictive prior:

&#x20;      π\_hat\_k,c = average\_u softmax(z\_k(u)/τ)\_c



&#x20; 5. Server applies smoothing / clipping:

&#x20;      π\_bar\_k = Normalize(clip(π\_hat\_k, p\_min, p\_max))



&#x20; 6. Server computes debias strength:

&#x20;      fixed version:

&#x20;         β\_k = β

&#x20;      adaptive version:

&#x20;         h\_k = H(π\_bar\_k) / log C

&#x20;         β\_k = β0(1 - h\_k)



&#x20; 7. Server performs prior debiasing:

&#x20;      z\_tilde\_k,c(u) = z\_k,c(u) - β\_k log(π\_bar\_k,c + ε)



&#x20; 8. Server computes debiased probabilities:

&#x20;      p\_tilde\_k(c|u) = softmax(z\_tilde\_k(u)/τ)\_c



&#x20; 9. Server computes class coverage weights:

&#x20;      q\_k,c = (π\_bar\_k,c + ε)^η / Σ\_j(π\_bar\_j,c + ε)^η



&#x20; 10. Server computes sample confidence:

&#x20;      r\_k(u) = 1 - H(p\_tilde\_k(.|u)) / log C



&#x20; 11. Server computes aggregation weights:

&#x20;      a\_k,c(u) = q\_k,c · r\_k(u)



&#x20; 12. Server constructs global teacher:

&#x20;      s(c|u) = Σ\_k a\_k,c(u) p\_tilde\_k(c|u)

&#x20;      T(c|u) = s(c|u) / Σ\_c' s(c'|u)



&#x20; 13. Server sends T(c|u) and π\_bar\_k to each client k



&#x20; 14. Each client performs personalized complementary distillation

```



\---



\## 13.2 Client 端算法



```text

Algorithm: FedPRIME-D2C Client k



Input:

&#x20; local private dataset D\_k

&#x20; local heterogeneous model f\_k

&#x20; public batch B\_pub

&#x20; global teacher T from server



Local PRIME Robust Training:



&#x20; For local epoch e = 1,...,E:

&#x20;     sample private batch B\_k from D\_k



&#x20;     for each (x,y) in B\_k:

&#x20;         x1' = PRIME(x)

&#x20;         x2' = PRIME(x)



&#x20;     compute:

&#x20;         L\_ce = CE(f\_k(x), y) + CE(f\_k(x1'), y)



&#x20;         L\_jsd = JS(

&#x20;             softmax(f\_k(x)),

&#x20;             softmax(f\_k(x1')),

&#x20;             softmax(f\_k(x2'))

&#x20;         )



&#x20;     update θ\_k with:

&#x20;         L\_local = L\_ce + λ\_jsd L\_jsd



Public Logit Upload:



&#x20; compute logits z\_k(u) = f\_k(u), u ∈ B\_pub

&#x20; send z\_k(u) to server



D2C Distillation:



&#x20; receive teacher T(c|u) and π\_bar\_k



&#x20; compute complementary weights:

&#x20;     m\_k,c = (1 - π\_bar\_k,c)^ρ

&#x20;     normalize m\_k,c



&#x20; for u ∈ B\_pub:

&#x20;     p\_k(c|u) = softmax(f\_k(u)/τ)\_c



&#x20;     optional gate:

&#x20;         g\_k(u) = H(p\_k(.|u)) / log C



&#x20; compute:

&#x20;     L\_D2C = τ² Σ\_u g\_k(u) Σ\_c m\_k,c T(c|u)

&#x20;             log\[T(c|u) / p\_k(c|u)]



&#x20; update θ\_k with:

&#x20;     L\_total = L\_local + λ\_d2c L\_D2C

```



\---



\# 14. 为什么这个框架支持模型异构？



因为通信对象是公共数据 logits：



\[

z\_k(u)\\in \\mathbb{R}^{C}

]



只要所有客户端任务类别数一致，模型结构可以完全不同。



FedPRIME-D2C 不需要：



```text

参数聚合

模型结构一致

共享 backbone

共享 projection head

特征维度对齐

原型空间

```



所以它天然支持模型异构。



这和 FedProto 路线完全不同。FedProto 通过类原型实现模型异构，而你这里完全不依赖 prototype。



\---



\# 15. 为什么这个框架支持数据异构？



FedPRIME-D2C 对数据异构的处理不是简单做一个 Non-IID 实验，而是方法本身就显式建模了 Non-IID。



它有三层数据异构机制。



\---



\## 15.1 Prior Debiasing：去除本地先验偏置



Non-IID 造成客户端 public logits 偏向本地多见类别。



D2C 用：



\[

\\tilde{z}\_{k,c}(u)

==================



\## z\_{k,c}(u)



\\beta\_k \\log(\\bar{\\pi}\_{k,c}+\\epsilon)

]



在服务器聚合前先削弱这种偏置。



\---



\## 15.2 Class-balanced Aggregation：类别级知识融合



不是每个客户端整体一个权重，而是每个类别都有不同权重：



\[

q\_{k,c}

]



并结合样本置信度：



\[

r\_k(u)

]



形成：



\[

a\_{k,c}(u)=q\_{k,c}r\_k(u)

]



这比普通 logits 平均更适合 Non-IID。



\---



\## 15.3 Personalized Complementary Distillation：缺什么补什么



客户端 (i) 对类别 (c) 的学习权重：



\[

m\_{i,c}

=======



(1-\\bar{\\pi}\_{i,c})^\\rho

]



本地越缺的类别，越多学习全局 teacher。



这就是个性化互补蒸馏。



\---



\# 16. 为什么这个框架能抗数据损坏？



抗数据损坏主要来自 PRIME 本地鲁棒学习。



PRIME 的 spectral / spatial / color primitives 能生成多种语义保持型扰动，覆盖 noise、blur、weather、digital corruption 等 common corruption 的部分变化空间。PRIME 论文的实验表明，它在 CIFAR-10-C、CIFAR-100-C、ImageNet-100-C、ImageNet-C 等 corruption benchmark 上优于 AugMix。



在 FedPRIME-D2C 中：



```text

每个客户端本地用 PRIME 训练，

使自身模型对损坏输入更稳定；

然后 D2C 让不同异构模型通过去偏后的 logits 交换知识，

避免 Non-IID 先验污染联邦蒸馏。

```



所以鲁棒性来自：



```text

local robustness: PRIME

federated robustness: D2C 去偏通信

```



\---



\# 17. 和 RAHFL 的区别



你的框架一定要和 RAHFL 拉开。



可以这样对比：



| 维度       | RAHFL 风格                                | FedPRIME-D2C                                   |

| -------- | --------------------------------------- | ---------------------------------------------- |

| 底层鲁棒模块   | AugMix-style augmentation               | PRIME                                          |

| 通信对象     | public logits / knowledge               | public logits                                  |

| 通信逻辑     | reliability-aware / asymmetric learning | distribution-decoupled / complementarity-aware |

| 核心问题     | 谁更可靠，谁教谁                                | 哪些 logits 被 Non-IID prior 污染                   |

| 模型异构     | 通过知识蒸馏支持                                | 通过 public logits 支持                            |

| 数据异构     | 多为实验验证                                  | 方法中显式建模                                        |

| 是否用原型    | 不用                                      | 不用                                             |

| 是否需要特征对齐 | 不需要                                     | 不需要                                            |

| 个性化      | 较弱或依赖可靠性                                | 按客户端缺失类别个性化蒸馏                                  |

| 关键公式     | 可靠性加权                                   | prior debias + complementary KD                |



核心区别一句话：



> RAHFL 的通信逻辑是 reliability-aware，FedPRIME-D2C 的通信逻辑是 distribution-decoupled and complementarity-aware。



再说得更直白：



```text

RAHFL 问：哪个客户端更可靠？

FedPRIME-D2C 问：客户端 logits 中哪些部分是本地分布偏置，如何去偏后互补学习？

```



这就是你自己的通信模块。



\---



\# 18. 推荐的 MVP 版本



第一版不要上太多东西，先跑通核心飞轮。



MVP 保留：



```text

1\. Local PRIME training

2\. Public logits communication

3\. Prior debiasing

4\. Class-balanced aggregation

5\. Complementary KD

```



MVP 暂时不加：



```text

1\. self-preserving gate

2\. adaptive β

3\. EMA prior

4\. public PRIME views

5\. DP prior

```



MVP 公式：



\[

\\tilde{z}\_{k,c}(u)

==================



\## z\_{k,c}(u)



\\beta \\log(\\bar{\\pi}\_{k,c}+\\epsilon)

]



\[

T(c|u)

======



\\frac{

\\sum\_k q\_{k,c}\\tilde{p}\*k(c|u)

}{

\\sum\*{c'}\\sum\_k q\_{k,c'}\\tilde{p}\_k(c'|u)

}

]



\[

\\mathcal{L}\_{D2C}^{i}

=====================



\\tau^2

\\sum\_{u}

\\sum\_c

(1-\\bar{\\pi}\_{i,c})^\\rho

T(c|u)

\\log

\\frac{T(c|u)}{p\_i(c|u)}

]



这个版本已经足够证明核心思想。



\---



\# 19. 完整版可以逐步加入的增强



当 MVP 跑通后，再依次加入：



\## 19.1 Adaptive (\\beta\_k)



\[

\\beta\_k=\\beta\_0(1-h\_k)

]



其中：



\[

h\_k=\\frac{H(\\bar{\\pi}\_k)}{\\log C}

]



作用：Non-IID 越严重，去偏越强。



\---



\## 19.2 Self-preserving gate



\[

g\_i(u)=\\frac{H(p\_i(\\cdot|u))}{\\log C}

]



作用：客户端越不确定，越学习 teacher；越确定，越保留自己知识。



\---



\## 19.3 EMA predictive prior



\[

\\hat{\\pi}\_{k}^{(t)}

===================



\\alpha\_{ema}\\hat{\\pi}\*{k}^{(t-1)}

\+

(1-\\alpha\*{ema})\\hat{\\pi}\_{k}^{batch}

]



作用：防止单个 public batch 导致先验估计波动。



\---



\## 19.4 Oracle prior ablation



使用真实 label histogram 做上限实验：



```text

D2C-pred: public prediction prior

D2C-oracle: true local label prior

```



作用：证明 public-predicted prior 和 oracle prior 差距不大。



\---



\# 20. 实验设计



\## 20.1 数据集



第一阶段：



```text

Private training: CIFAR-10

Public data: CIFAR-100 subset / TinyImageNet subset

Test: CIFAR-10 clean + CIFAR-10-C

```



第二阶段：



```text

Private training: CIFAR-100

Public data: TinyImageNet

Test: CIFAR-100 clean + CIFAR-100-C

```



第三阶段可选：



```text

Private training: TinyImageNet

Test: TinyImageNet-C

```



\---



\## 20.2 模型异构设置



建议 10 或 20 个客户端，模型结构混合：



```text

ResNet-18

MobileNetV2

ShuffleNetV2

VGG-small

CNN-small

ViT-Tiny

```



每个客户端固定自己的模型，不参与参数聚合。



\---



\## 20.3 数据异构设置



使用 Dirichlet label partition：



\[

\\alpha \\in {1.0, 0.5, 0.3, 0.1}

]



重点看：



```text

α = 0.5

α = 0.1

```



如果 D2C 在 severe Non-IID 下提升明显，故事就成立。



\---



\## 20.4 Corruption 设置



测试集：



```text

CIFAR-10-C

CIFAR-100-C

```



报告：



```text

Clean Accuracy

Corruption Accuracy

mCE

Noise corruption accuracy

Blur corruption accuracy

Weather corruption accuracy

Digital corruption accuracy

```



\---



\# 21. Baselines



必须包括这些。



\## 21.1 本地训练 baseline



```text

Local

Local + AugMix

Local + PRIME

```



\## 21.2 异构联邦 baseline



```text

FedMD

FedDF

RHFL

AugHFL

RAHFL

```



\## 21.3 关键强对照



```text

RAHFL + PRIME

```



这个非常重要。



它用于证明：



> 性能提升不是简单因为 PRIME 比 AugMix 强，而是因为 D2C 通信模块有效。



\## 21.4 你的方法



```text

FedPRIME-D2C

```



\---



\# 22. 消融实验



至少做：



```text

FedPRIME-D2C

w/o PRIME

w/o prior debiasing

w/o class-balanced aggregation

w/o complementary KD

w/o self-preserving gate

fixed β vs adaptive β

predicted prior vs oracle prior

ordinary logit averaging vs D2C aggregation

```



对应证明：



```text

PRIME 负责 corruption robustness

prior debiasing 负责去除 Non-IID bias

class-balanced aggregation 负责类别级知识融合

complementary KD 负责个性化互补学习

```



\---



\# 23. 超参数建议



初始可以这样设：



```text

clients: 10

rounds: 100

local epochs: 1 or 5

public batch size: 256 or 512

temperature τ: 3

λ\_jsd: 1.0

λ\_d2c: 1.0

β: 0.5

β0: 0.5

η: 0.5

ρ: 1.0

p\_min: 1e-3

ε: 1e-6

EMA α: 0.9

optimizer: SGD or AdamW

```



优先跑：



```text

CIFAR-10 + CIFAR-10-C

10 clients

4 model types

Dirichlet α = 0.5 and 0.1

```



\---



\# 24. 论文贡献写法



可以写成四点。



\*\*贡献一：\*\*



> We introduce PRIME into model-heterogeneous federated learning as a plug-and-play local robust learner against common corruptions.



中文：



> 我们将 PRIME 引入模型异构联邦学习，作为即插即用的本地 common corruption 鲁棒学习模块。



\---



\*\*贡献二：\*\*



> We identify that public logits in model-heterogeneous FL are contaminated by local predictive priors under Non-IID data.



中文：



> 我们指出，在 Non-IID 数据下，模型异构联邦学习中的 public logits 会受到客户端本地预测先验偏置的污染。



\---



\*\*贡献三：\*\*



> We propose Distribution-Decoupled Communication, which debiases client logits before aggregation and constructs class-balanced consensus teachers.



中文：



> 我们提出分布解耦通信模块，在聚合前对客户端 logits 进行先验去偏，并构建类别均衡的全局 teacher。



\---



\*\*贡献四：\*\*



> We design personalized complementary distillation, enabling each heterogeneous client to selectively learn globally shared knowledge for locally underrepresented classes.



中文：



> 我们设计个性化互补蒸馏，使每个异构客户端优先学习本地稀缺类别的全局知识，从而更好适配数据异构。



\---



\# 25. Method Section 建议结构



论文方法部分可以这样组织：



```text

3\. Method



3.1 Problem Formulation

&#x20;   - model heterogeneity

&#x20;   - data heterogeneity

&#x20;   - common corruptions



3.2 Overview of FedPRIME-D2C

&#x20;   - local PRIME robust learning

&#x20;   - D2C communication



3.3 Local PRIME Robust Learning

&#x20;   - PRIME augmentation

&#x20;   - CE + JSD loss



3.4 Public Logit Communication for Model Heterogeneity

&#x20;   - no parameter aggregation

&#x20;   - no prototypes

&#x20;   - only public logits



3.5 Distribution-Decoupled Communication

&#x20;   3.5.1 Predictive Prior Estimation

&#x20;   3.5.2 Local-prior Logit Debiasing

&#x20;   3.5.3 Class-balanced Consensus Aggregation

&#x20;   3.5.4 Personalized Complementary Distillation



3.6 Overall Optimization

&#x20;   - final loss

&#x20;   - algorithm

```



\---



\# 26. 最终定位



你的框架最终要表达的是：



> 现有 robust heterogeneous FL 方法大多关注“哪个客户端更可靠”，但在 severe Non-IID 下，public logits 的核心问题是本地类别先验污染。FedPRIME-D2C 不再做可靠性排序，而是先对客户端 logits 做分布解耦，再进行类别均衡聚合和个性化互补蒸馏。



最核心的一句话：



> \*\*FedPRIME-D2C uses PRIME as a robust local learner and D2C as a distribution-decoupled logit communication mechanism to jointly handle model heterogeneity, data heterogeneity, and common corruptions.\*\*



中文：



> \*\*FedPRIME-D2C 用 PRIME 作为本地鲁棒底座，用分布解耦的 logit 通信机制同时处理模型异构、数据异构和 common corruption 鲁棒性。\*\*



